"""Source-supervised regional intervention fusion; frozen experts, no online gradients."""
import os,sys,json,time,random,signal,csv,importlib.util
from pathlib import Path
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(os.environ['WORK_ROOT']);STOP=False
COUNTS={'fusion_forward_images':0,'outer_updates':0,'inner_grad':0}
def stop(*_):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop)
def write(p,obj):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(p)
def log(obj):print(json.dumps(obj),flush=True)
def size():return sum(p.stat().st_blocks*512 for p in ROOT.rglob('*') if p.is_file())
class Block(nn.Sequential):
 def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
class Fusion(nn.Module):
 def __init__(self):
  super().__init__();self.enc=Block(11,16);self.mid=Block(16,32);self.deep=Block(32,64);self.dec=Block(96,32);self.out=Block(48,16)
  self.score=nn.Conv2d(16,8,1);self.delta=nn.Conv2d(16,2,1);nn.init.normal_(self.delta.weight,std=.005);nn.init.zeros_(self.delta.bias)
  with torch.no_grad():self.score.bias.reshape(2,4)[:,1]=2
 def forward(self,x,probs,control=False):
  COUNTS['fusion_forward_images']+=len(x)
  h=self.enc(torch.cat([x,probs.flatten(1,2)],1));m=self.mid(F.avg_pool2d(h,2));b=self.deep(F.avg_pool2d(m,2))
  u=self.dec(torch.cat([F.interpolate(b,size=m.shape[-2:],mode='bilinear',align_corners=False),m],1));h=self.out(torch.cat([F.interpolate(u,size=h.shape[-2:],mode='bilinear',align_corners=False),h],1))
  logits=F.avg_pool2d(self.score(h),(h.shape[-2]//8,h.shape[-1]//8)).reshape(len(x),2,4,8,8)
  w=logits.softmax(2);identity=torch.zeros_like(w);identity[:,:,1]=1;active=identity if control else w
  wp=F.interpolate(active.flatten(1,2),size=x.shape[-2:],mode='bilinear',align_corners=False).reshape_as(probs)
  proposals=.5*probs+.5*probs[:,:,1:2]
  mixed=(wp*proposals).sum(2);residual=3*torch.tanh(self.delta(h))*residual_support(probs);mixed_z=torch.logit(mixed.clamp(.001,.999))
  return mixed_z+residual,w,mixed_z,residual

def residual_support(probs):
 # Frozen DIRECT uncertainty gates both arms; confident errors may be harder to repair.
 p=probs[:,:,1]
 return 4*p*(1-p)

def logmom(x):
 mask=(x.mean(1,keepdim=True)>.05).float();return (x.clamp_min(.001).log()*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1)
def canonical(x,ref):return x.clamp_min(.001).pow((ref[None,:].to(x.device)/logmom(x).clamp_max(-.001)).clamp(.5,2)[:,:,None,None])
def experts(core,base,expert,x,ref):
 with torch.no_grad():
  z=core.base_forward(base,x);xc=canonical(x,ref);zc=core.base_forward(base,xc)
  x=F.interpolate(x,size=(256,256),mode='bilinear',align_corners=False);xc=F.interpolate(xc,size=(256,256),mode='bilinear',align_corners=False);a=torch.zeros(len(x),16,device=x.device)
  p=expert(x,z,z.sigmoid(),a)[0].sigmoid();pc=expert(xc,zc,zc.sigmoid(),a)[0].sigmoid()
 return x,torch.stack([z.sigmoid(),p,zc.sigmoid(),pc],2)
def intervention_credit(probs,y):
 proposals=.5*probs+.5*probs[:,:,1:2];pred=(proposals>.5).float();base=pred[:,:,1]
 area=(y.shape[-2]//8)*(y.shape[-1]//8)
 def region(t):return F.avg_pool2d(t.flatten(1,2),(y.shape[-2]//8,y.shape[-1]//8)).reshape(len(y),2,4,8,8)*area
 rp=region(pred);rtp=region(pred*y[:,:,None])
 tp=(base*y).sum((-2,-1))[:,:,None,None,None];bp=base.sum((-2,-1))[:,:,None,None,None];gy=y.sum((-2,-1))[:,:,None,None,None]
 altered=100*(2*(tp-rtp[:,:,1:2]+rtp)+1e-6)/(bp-rp[:,:,1:2]+rp+gy+1e-6)
 original=100*(2*tp+1e-6)/(bp+gy+1e-6)
 return altered-original

def utility_loss(weights,probs,y):
 with torch.no_grad():
  gain=intervention_credit(probs,y);target=(gain/.1).softmax(2);identity=torch.zeros_like(target);identity[:,:,1]=1
  active=(gain.max(2,keepdim=True).values>1e-6);target=torch.where(active,target,identity)
  credit=(target*gain).sum(2).mean();fraction=active.float().mean()
 kl=(target*(target.clamp_min(1e-8).log()-weights.clamp_min(1e-8).log())).sum(2).mean()
 return kl,float(credit),float(fraction)
def warp(x,theta,mode='bilinear'):
 grid=F.affine_grid(theta,x.shape,align_corners=False)
 return F.grid_sample(x,grid,mode=mode,padding_mode='zeros',align_corners=False)
def geometry(rng,batch,device):
 mats=[]
 for _ in range(batch):
  a=rng.uniform(-15,15)*math.pi/180;s=rng.uniform(.9,1.1);tx=rng.uniform(-.05,.05);ty=rng.uniform(-.05,.05)
  mats.append([[s*math.cos(a),-s*math.sin(a),tx],[s*math.sin(a),s*math.cos(a),ty]])
 return torch.tensor(mats,device=device,dtype=torch.float32)

def ratio(t):
 hh=[]
 for h in t:
  ids=h.any(1).nonzero().flatten();hh.append(float(ids[-1]-ids[0]+1) if len(ids) else 0.)
 return hh[1]/max(hh[0],1)
def evaluate(model,core,base,expert,dev,ref,arm):
 model.eval();rows=[]
 with torch.no_grad():
  for order in [0,1]:
   ids=list(range(len(dev)));random.Random(8821+order).shuffle(ids)
   for kind in ['clean','gamma','color','blur']:
    for j in ids:
     im,y,group=dev[j];x=core.corruption(im[None].cuda(),kind,.8);y=y[None].cuda();x,ps=experts(core,base,expert,x,ref)
     out,w,mix,res=model(x,ps,arm=='CONTROL')
     preds={'FULL' if arm=='FULL' else 'MATCHED_CONTROL':out,'SOURCE':torch.logit(ps[:,:,0].clamp(.001,.999)),'EXPERT_DIRECT':torch.logit(ps[:,:,1].clamp(.001,.999)),'CANON_SOURCE':torch.logit(ps[:,:,2].clamp(.001,.999)),'CANON_DIRECT':torch.logit(ps[:,:,3].clamp(.001,.999)),'MEAN':torch.logit(ps.mean(2).clamp(.001,.999)),'NO_RESIDUAL':mix}
     if arm=='FULL':preds['DIRECT_WEIGHTS']=torch.logit(ps[:,:,1].clamp(.001,.999))+res
     for name,z in preds.items():
      dd=core.dice(z,y)[0].cpu().tolist();p=(z>0)[0];gt=y[0]>.5
      rows.append({'group':group,'content':j,'order':order,'shift':kind,'arm':name,'od':dd[0],'oc':dd[1],'dice':sum(dd)/2,'empty_heads':int((p.sum((-2,-1))==0).sum()),'outside_cup':int((p[1]&~p[0]).sum()),'vcdr_error':abs(ratio(p)-ratio(gt))})
 metrics={}
 for name in sorted(set(v['arm'] for v in rows)):
  rr=[v for v in rows if v['arm']==name];metrics[name]={k:float(np.mean([v[k] for v in rr])) for k in ['dice','od','oc','empty_heads','outside_cup','vcdr_error']};metrics[name]['clean_dice']=float(np.mean([v['dice'] for v in rr if v['shift']=='clean']))
 model.train();return metrics,rows

def main():
 cfg=json.loads((ROOT/'config.private.json').read_text());torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sp=importlib.util.spec_from_file_location('core',cfg['expert_module']);core=importlib.util.module_from_spec(sp);sp.loader.exec_module(core);signal.signal(signal.SIGTERM,stop)
 torch.manual_seed(7);f=Fusion().cuda();xx=torch.rand(1,3,32,32,device='cuda');pp=torch.rand(1,2,4,32,32,device='cuda');z,w,_,_=f(xx,pp);assert torch.allclose(w.sum(2),torch.ones_like(w[:,:,0]));(core.loss(z,torch.rand_like(z))+.1*utility_loss(w,pp,torch.rand_like(z))[0]).backward();assert f.score.weight.grad is not None and torch.isfinite(f.score.weight.grad).all();assert torch.isfinite(f.delta.weight.grad).all()
 with torch.no_grad():
  gy=(torch.rand(1,2,32,32,device='cuda')>.5).float();gain=intervention_credit(pp,gy)
  original=(pp[:,:,1]>.5).float();replaced=original.clone();candidate=((.5*pp[:,:,0]+.5*pp[:,:,1])>.5).float();replaced[:,:,8:12,12:16]=candidate[:,:,8:12,12:16]
  def exact(p):return 100*(2*(p*gy).sum((-2,-1))+1e-6)/((p+gy).sum((-2,-1))+1e-6)
  assert torch.allclose(gain[:,:,0,2,3],exact(replaced)-exact(original),atol=1e-4,rtol=1e-4)
  support=residual_support(pp);assert support.min()>=0 and support.max()<=1
  extremes=pp.clone();extremes[:,:,1]=0;assert torch.count_nonzero(residual_support(extremes))==0
  extremes[:,:,1]=1;assert torch.count_nonzero(residual_support(extremes))==0
  extremes[:,:,1]=.5;assert torch.equal(residual_support(extremes),torch.ones_like(extremes[:,:,1]))
  _,_,_,res=f(xx,pp);assert (res.abs()<=3*support+1e-6).all()
  identity=torch.tensor([[[1.,0,0],[0,1.,0]]],device='cuda');assert torch.allclose(warp(xx,identity),xx,atol=1e-5)
  p1=f(xx,pp)[0];unused_labels=torch.rand_like(p1);p2=f(xx,pp)[0];assert torch.equal(p1,p2)
 write(ROOT/'SELF_CHECK.json',{'status':'PASS','source_supervised_only':True,'inference_label_free':True,'expert_weight_sum':1,'credit_matches_brute_replacement':True,'affine_identity_check':True,'uncertainty_support_bounds_and_extremes':True,'residual_bound_check':True,'gradient_updates_at_deployment':0,'synthetic_fusion_calls':COUNTS['fusion_forward_images']});del f
 for k in COUNTS:COUNTS[k]=0
 fit,dev=core.load_data(cfg);ref=torch.cat([logmom(v[0][None]) for v in fit],0).median(0).values
 write(ROOT/'CANON_REFERENCE.json',{'source_fit_only':True,'log_means':ref.tolist(),'gamma_bounds':[.5,2]})
 sys.path.insert(0,cfg['backbone_code']);from networks.ResUnet import ResUnet
 base=ResUnet(resnet='resnet34',num_classes=2,pretrained=False).cuda();base.load_state_dict(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True),strict=True);base.eval().requires_grad_(False)
 z=core.base_forward(base,torch.stack([v[0] for v in fit[:2]]).cuda());dd=core.dice(z,torch.stack([v[1] for v in fit[:2]]).cuda());assert dd.mean()>50
 write(ROOT/'SOURCE_INTERFACE_CHECK.json',{'dice':dd.cpu().tolist(),'source_fit_only':True});write(ROOT/'SOURCE_RECEIPT.json',{'fit_count':len(fit),'dev_count':len(dev),'target_images_read':0,'frozen_experts':True})
 for seed in cfg['seeds']:
  expert=core.Model().cuda();ck=torch.load(cfg['expert_checkpoints'][str(seed)],map_location='cpu',weights_only=True);expert.load_state_dict(ck['model'],strict=True);expert.eval().requires_grad_(False)
  for arm in ['CONTROL','FULL']:
   if STOP:break
   torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);model=Fusion().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);run=ROOT/'runs'/f'{arm.lower()}_{seed}';run.mkdir(parents=True,exist_ok=True);best=-float('inf');t0=time.time();step=0
   for step in range(1,cfg['steps']+1):
    if STOP:break
    if step%20==1 and size()>cfg['run_storage_limit_bytes']:raise RuntimeError('storage stop')
    rng=random.Random(seed*100000+step);torch.manual_seed(seed*100000+step);inds=rng.sample(range(len(fit)),3);xx=torch.stack([fit[i][0] for i in inds]).cuda();yy=torch.stack([fit[i][1] for i in inds]).cuda()
    rng.random()  # Preserve augmentation RNG draws; retain native orientation.
    xx=core.corruption(xx,rng.choice(['clean','gamma','color','blur','noise']),rng.uniform(.2,1.4));theta=geometry(rng,len(xx),xx.device);xx2=warp(xx,theta);yy2=warp(yy,theta,'nearest')
    valid=warp(torch.ones_like(yy[:,:1]),theta).detach();xx,ps=experts(core,base,expert,xx,ref);xx2,ps2=experts(core,base,expert,xx2,ref)
    opt.zero_grad(set_to_none=True);out,w,_,res=model(xx,ps,arm=='CONTROL');out2,w2,_,res2=model(xx2,ps2,arm=='CONTROL');ul,credit,fraction=utility_loss(w,ps,yy);ul2,credit2,fraction2=utility_loss(w2,ps2,yy2)
    # Detach the source view before warping: no nondeterministic grid-sample backward.
    target=warp(out.sigmoid().detach(),theta);consistency=((out2.sigmoid()-target).square()*valid).sum()/(2*valid.sum().clamp_min(1))
    objective=.5*(core.loss(out,yy)+core.loss(out2,yy2))+.05*(ul+ul2)+.0005*(res.square().mean()+res2.square().mean())+.1*consistency
    assert torch.isfinite(objective);objective.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();COUNTS['outer_updates']+=1
    if step==1 or step%20==0:
     st={'stage':'TRAINING','arm':arm,'seed':seed,'step':step,'steps':cfg['steps'],'loss':objective.item(),'utility_loss':ul.item(),'geometric_consistency':consistency.item(),'source_credit_pp':credit,'source_positive_region_fraction':fraction,'elapsed_seconds':time.time()-t0,'counts':{**core.COUNTS,**COUNTS},'gpu_peak_bytes':torch.cuda.max_memory_allocated()};write(ROOT/'RUN_STATE.json',st);log(st)
    if step%cfg['eval_every']==0 or step==cfg['steps']:
     metrics,rows=evaluate(model,core,base,expert,dev,ref,arm);score=metrics['FULL' if arm=='FULL' else 'MATCHED_CONTROL']['dice'];record={'step':step,'metrics':metrics,'score':score};write(run/f'validation_{step}.json',record);log({'validation':record,'arm':arm,'seed':seed})
     if score>best:best=score;torch.save({'model':model.state_dict(),'seed':seed,'arm':arm,'step':step},run/'best.pt');write(run/'best_metrics.json',record);write(run/'best_rows.private.json',rows)
     torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'arm':arm,'step':step},run/'last.pt')
   torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'arm':arm,'step':step},run/'last.pt');write(run/'completion.json',{'status':'STOPPED' if STOP else 'COMPLETE','steps':step,'best_dev_dice':best,'seconds':time.time()-t0})
 write(ROOT/'TRAINING_FINISHED.json',{'status':'STOPPED' if STOP else 'COMPLETE','counts':{**core.COUNTS,**COUNTS},'at':time.time()})
if __name__=='__main__':main()
