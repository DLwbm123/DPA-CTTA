"""Source-trained regional adaptation. Runtime configuration is inherited via environment."""
import os,sys,json,time,random,signal,csv,math,copy
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(os.environ.get('WORK_ROOT','.'))
STOP=False
COUNTS={'backbone_forward_images':0,'refiner_forward_images':0,'inner_grad':0,'outer_updates':0}
def stop(*_):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop)

def write_json(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def log(x):
 print(json.dumps(x,ensure_ascii=False),flush=True)
def disk_bytes():
 return sum(p.stat().st_blocks*512 for p in ROOT.rglob('*') if p.is_file())
def loss(z,y):
 p=z.sigmoid();dims=(-2,-1)
 dice=(2*(p*y).sum(dims)+1)/((p+y).sum(dims)+1)
 return F.binary_cross_entropy_with_logits(z,y)+(1-dice).mean()
def dice(z,y):
 p=(z.sigmoid()>.5).float();return ((2*(p*y).sum((-2,-1))+1e-6)/((p+y).sum((-2,-1))+1e-6))*100

class Block(nn.Sequential):
 def __init__(self,a,b):
  super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.enc=Block(7,16);self.mid=Block(16,32);self.deep=Block(32,64)
  self.state=nn.Linear(16,64,bias=False);self.dec=Block(96,32);self.out=Block(48,16)
  self.delta=nn.Conv2d(16,2,1);self.score=nn.Conv2d(16,2,1);self.fields=nn.Conv2d(16,32,1)
  nn.init.normal_(self.delta.weight,std=.005);nn.init.zeros_(self.delta.bias)
  self.basis=nn.Parameter(torch.eye(16)[:,:4].clone());self.loglr=nn.Parameter(torch.tensor(-1.4))
  self.reg=nn.Linear(16,1);self.critic=nn.Sequential(nn.Linear(20,32),nn.SiLU(),nn.Linear(32,1))
 def forward(self,x,z,q,a):
  COUNTS['refiner_forward_images']+=len(x)
  h=self.enc(torch.cat([x,z.sigmoid(),q],1));m=self.mid(F.avg_pool2d(h,2));b=self.deep(F.avg_pool2d(m,2))
  b=b+self.state(a)[:,:,None,None]
  u=self.dec(torch.cat([F.interpolate(b,size=m.shape[-2:],mode='bilinear',align_corners=False),m],1))
  h=self.out(torch.cat([F.interpolate(u,size=h.shape[-2:],mode='bilinear',align_corners=False),h],1))
  field=torch.tanh(self.fields(h)).reshape(len(x),2,16,*h.shape[-2:])
  out=torch.logit(q.clamp(.001,.999))+6*torch.tanh(self.delta(h))+(field*a[:,None,:,None,None]).sum(2)
  s=F.avg_pool2d(self.score(h),(h.shape[-2]//8,h.shape[-1]//8))
  f=h.mean((-2,-1));reg=.001+.02*torch.sigmoid(self.reg(f)).mean()
  return out,s,f,reg

def weights(scores,mode,training):
 flat=scores.flatten(2)
 if mode=='random':key=torch.rand_like(flat)
 else:key=flat
 idx=key.topk(32,dim=-1).indices;hard=torch.zeros_like(flat).scatter_(2,idx,1)
 if training and mode!='random':
  soft=torch.sigmoid(flat-flat.mean(-1,keepdim=True));mask=hard+soft-soft.detach()
 else:mask=hard
 w=mask/(mask.mean(-1,keepdim=True)+1e-8)
 return w.reshape_as(scores)

def adapt(model,x,z,a,training=False,mode='learned'):
 """No label argument; inference cannot access source or target ground truth."""
 with torch.enable_grad():
  a=a.detach().requires_grad_(True)
  before,scores,feat,reg=model(x,z,z.sigmoid(),a)
  with torch.no_grad():target=model(x,z,before.sigmoid().detach(),torch.zeros_like(a))[0].sigmoid()
  w=weights(scores,mode,training);wm=F.interpolate(w,size=z.shape[-2:],mode='nearest')
  energy=((before.sigmoid()-target).square()*wm).mean()+reg*a.square().mean()
  g=torch.autograd.grad(energy,a,create_graph=training)[0];COUNTS['inner_grad']+=1
  Q=torch.linalg.qr(model.basis,mode='reduced').Q
  direction=(g@Q)@Q.T
  step=.5*model.loglr.sigmoid()*direction/(direction.norm(dim=1,keepdim=True)+1e-8)
  # Smooth bounded step and state, with no optimizer history at deployment.
  step=step/(1+step.norm(dim=1,keepdim=True)/.5)
  nxt=a-step;nxt=nxt/(1+nxt.norm(dim=1,keepdim=True)/2)
  after=model(x,z,z.sigmoid(),nxt)[0]
  stats=torch.cat([feat,(before.sigmoid()-target).abs().mean((-2,-1)),step.norm(dim=1,keepdim=True),a.norm(dim=1,keepdim=True)],1)
  gain=model.critic(stats).squeeze(-1)
  return before,after,nxt,gain,energy,w

def corruption(x,kind,level=1.):
 if kind=='clean':return x
 if kind=='gamma':return x.clamp_min(1e-6).pow(1+.65*level)
 if kind=='color':return (x*torch.tensor([1+.25*level,1-.15*level,1-.2*level],device=x.device)[None,:,None,None]).clamp(0,1)
 if kind=='blur':return F.avg_pool2d(x,5,1,2)
 if kind=='noise':return (x+.035*level*torch.randn_like(x)).clamp(0,1)
 raise ValueError(kind)

def load_data(cfg):
 dr=Path(cfg['data_root']);rows=list(csv.DictReader((dr/'RIM_ONE_r3_train.csv').open()))
 assert len(rows)==99, len(rows)
 def group(r):return Path(r['image']).stem.rsplit('-',1)[0]
 groups=sorted(set(map(group,rows)));rng=random.Random(10223);rng.shuffle(groups)
 devgroups=set(groups[:max(1,round(len(groups)*.2))])
 fit=[r for r in rows if group(r) not in devgroups];dev=[r for r in rows if group(r) in devgroups]
 assert not ({r['image'] for r in fit}&{r['image'] for r in dev})
 write_json(ROOT/'split.private.json',{'fit':fit,'dev':dev,'group_rule':'filename without final L/R; unverified patient identity','backbone_exposure':'unknown; may include dev','excluded':['RIM_ONE_r3_test.csv','all other domains']})
 def get(r):
  x=np.array(Image.open(dr/r['image']).convert('RGB').resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32).transpose(2,0,1)
  x=(x-x.min())/max(float(x.max()-x.min()),1e-6)
  y=np.array(Image.open(dr/r['mask']).convert('L').resize((256,256),Image.Resampling.NEAREST))
  return torch.tensor(x),torch.tensor(np.stack([y<255,y==0]).astype('float32')),group(r)
 return [get(r) for r in fit],[get(r) for r in dev]

def base_forward(base,x):
 with torch.no_grad():z=base(x)[0]
 COUNTS['backbone_forward_images']+=len(x);return F.interpolate(z.detach(),size=(256,256),mode='bilinear',align_corners=False)

def evaluate(model,base,dev,device,full=True):
 model.eval();allrows=[];state0=torch.zeros(1,16,device=device)
 with torch.random.fork_rng(devices=[0]):
  torch.manual_seed(9191)
  for order in [0,1]:
   ids=list(range(len(dev)));random.Random(8821+order).shuffle(ids)
   states={m:state0.clone() for m in ['FULL','RANDOM']}
   for kind in ['clean','gamma','color','blur']:
    for j in ids:
     im,y,group=dev[j];x=corruption(im[None].to(device),kind,.8);y=y[None].to(device);z=base_forward(base,x);x=F.interpolate(x,size=(256,256),mode='bilinear',align_corners=False)
     with torch.no_grad():direct=model(x,z,z.sigmoid(),state0)[0]
     preds={'SOURCE':z,'NO_UPDATE':direct}
     if full:
      for mode in ['EPISODIC','FULL','RANDOM']:
       a=state0 if mode=='EPISODIC' else states[mode]
       _,out,nxt,gain,_,_=adapt(model,x,z,a,False,'random' if mode=='RANDOM' else 'learned')
       preds[mode]=out.detach()
       if mode!='EPISODIC':states[mode]=(nxt.detach() if gain.item()>0 else state0.clone())
     for mode,out in preds.items():
      dd=dice(out.detach(),y)[0].cpu().tolist();p=(out.detach().sigmoid()>.5)[0]
      gt=y[0]>.5
      def ratio(t):
       heights=[]
       for h in t:
        rows=h.any(dim=1).nonzero().flatten();heights.append(float(rows[-1]-rows[0]+1) if len(rows) else 0.)
       return heights[1]/max(heights[0],1)
      allrows.append({'group':group,'content':j,'order':order,'shift':kind,'arm':mode,'od':dd[0],'oc':dd[1],'dice':sum(dd)/2,'empty_heads':int((p.sum((-2,-1))==0).sum()),'outside_cup':int((p[1]&~p[0]).sum()),'vcdr_error':abs(ratio(p)-ratio(gt))})
 result={}
 for mode in sorted(set(r['arm'] for r in allrows)):
  rr=[r for r in allrows if r['arm']==mode];result[mode]={k:float(np.mean([r[k] for r in rr])) for k in ['dice','od','oc','empty_heads','outside_cup','vcdr_error']}
  result[mode]['clean_dice']=float(np.mean([r['dice'] for r in rr if r['shift']=='clean']))
 model.train();return result,allrows

def selfcheck(device):
 torch.manual_seed(7);m=Model().to(device);x=torch.rand(1,3,32,32,device=device);z=torch.randn(1,2,32,32,device=device);a=torch.zeros(1,16,device=device)
 b,o,n,g,e,w=adapt(m,x,z,a,True);assert torch.isfinite(o).all();assert (w.detach().flatten(2)>0).sum(-1).eq(32).all();assert torch.allclose(w.detach().mean((-2,-1)),torch.ones(1,2,device=device))
 outer=loss(o,torch.rand_like(o));outer.backward();assert m.basis.grad is not None and torch.isfinite(m.basis.grad).all();assert m.score.weight.grad is not None
 # Changing irrelevant external labels cannot alter the label-free function.
 m.eval();r1=adapt(m,x,z,a,False)[1];unused_labels=torch.rand_like(z);r2=adapt(m,x,z,a,False)[1];delta=float((r1-r2).abs().max());log({'synthetic_repeat_max_logit_delta':delta});assert torch.allclose(r1,r2,atol=1e-5,rtol=1e-5) and torch.equal(r1>0,r2>0)
 return {'status':'PASS','label_free_signature':True,'fixed_region_count':32,'weight_mean':1,'meta_gradient_finite':True}

def main():
 cfg=json.loads((ROOT/'config.private.json').read_text());torch.set_num_threads(4);device='cuda'
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 ck=selfcheck(device);ck['synthetic_calls']=COUNTS.copy();write_json(ROOT/'SELF_CHECK.json',ck);log(ck)
 for key in COUNTS:COUNTS[key]=cfg.get('prior_counts',{}).get(key,0)
 fit,dev=load_data(cfg);log({'fit':len(fit),'dev':len(dev)})
 sys.path.insert(0,cfg['backbone_code']);from networks.ResUnet import ResUnet
 base=ResUnet(resnet='resnet34',num_classes=2,pretrained=False).to(device)
 sd=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True);base.load_state_dict(sd,strict=True);base.eval().requires_grad_(False)
 write_json(ROOT/'SOURCE_RECEIPT.json',{'strict_load':True,'fit_count':len(fit),'dev_count':len(dev),'target_images_read':0,'backbone_resolution':512,'refiner_resolution':256,'trainable_backbone':0})
 with torch.no_grad():
  xx=torch.stack([v[0] for v in fit[:2]]).to(device);yy=torch.stack([v[1] for v in fit[:2]]).to(device);zz=base_forward(base,xx)
  dd=dice(zz,yy).cpu().tolist();write_json(ROOT/'SOURCE_INTERFACE_CHECK.json',{'dice':dd,'source_fit_only':True,'backbone_resolution':512,'refiner_resolution':256});assert float(np.mean(dd))>50,'source interface failed'
 for seed in cfg['seeds']:
  for arm in cfg.get('train_arms',['DIRECT','FULL']):
   if STOP:break
   torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
   model=Model().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
   run=ROOT/'runs'/f'{arm.lower()}_{seed}';run.mkdir(parents=True,exist_ok=True);best=-float('inf');t0=time.time();step=0
   for step in range(1,cfg['steps']+1):
    if STOP:break
    if step%20==1 and disk_bytes()>cfg['run_storage_limit_bytes']:raise RuntimeError('run storage hard stop')
    # The source sampling/corruption schedule is identical across train arms.
    rng=random.Random(seed*100000+step);torch.manual_seed(seed*100000+step)
    inds=rng.sample(range(len(fit)),3);xx=torch.stack([fit[i][0] for i in inds]).to(device);yy=torch.stack([fit[i][1] for i in inds]).to(device)
    if rng.random()<.5:xx=xx.flip(-1);yy=yy.flip(-1)
    kind=rng.choice(['clean','gamma','color','blur','noise']);xx=corruption(xx,kind,rng.uniform(.2,1.4))
    zz=base_forward(base,xx);xx=F.interpolate(xx,size=(256,256),mode='bilinear',align_corners=False);a=torch.zeros(3,16,device=device);opt.zero_grad(set_to_none=True)
    initial=model(xx,zz,zz.sigmoid(),a)[0];fixed=model(xx,zz,yy,a)[0]
    noisy=F.max_pool2d(yy,9,1,4) if rng.random()<.5 else 1-F.max_pool2d(1-yy,9,1,4)
    noisy=(.9*noisy+.05+.1*F.avg_pool2d(torch.randn_like(yy),9,1,4)).clamp(.001,.999)
    denoised=model(xx,zz,noisy,a)[0];denoise_loss=loss(denoised,yy)
    objective=loss(initial,yy)+.25*loss(fixed,yy)+.25*denoise_loss;aux={}
    if arm=='FULL' and step>cfg['warmup']:
     # Two actual source updates expose the model to its own carried state.
     _,first,s1,gain1,_,_=adapt(model,xx[:1],zz[:1],torch.zeros(1,16,device=device),True)
     carry=(torch.sigmoid(20*gain1)[:,None]*s1).detach()
     before,updated,nxt,pred_gain,energy,w=adapt(model,xx[1:2],zz[1:2],carry,True)
     afterq=model(xx[2:],zz[2:],zz[2:].sigmoid(),nxt)[0]
     target_gain=(loss(initial[2:],yy[2:])-loss(afterq,yy[2:])).detach().clamp(-1,1)
     target_gain1=(loss(initial[1:2],yy[1:2])-loss(before,yy[1:2])).detach().clamp(-1,1)
     gate=torch.sigmoid(20*pred_gain);committed=gate[:,None]*nxt
     gatedq=model(xx[2:],zz[2:],zz[2:].sigmoid(),committed)[0]
     objective=.5*loss(initial,yy)+.25*loss(fixed,yy)+.25*denoise_loss+.5*loss(first,yy[:1])+loss(updated,yy[1:2])+loss(gatedq,yy[2:])+10*((pred_gain-target_gain).square().mean()+(gain1-target_gain1).square().mean())
     aux={'energy':energy.item(),'state_norm':nxt.norm().item(),'pred_gain':pred_gain.item(),'true_source_query_gain':target_gain.item(),'gate':gate.item()}
    assert torch.isfinite(objective), 'nonfinite objective'
    objective.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();COUNTS['outer_updates']+=1
    if step==1 or step%20==0:
     status={'stage':'TRAINING','arm':arm,'seed':seed,'step':step,'steps':cfg['steps'],'loss':objective.item(),'elapsed_seconds':time.time()-t0,'counts':COUNTS.copy(),'gpu_peak_bytes':torch.cuda.max_memory_allocated(),**aux}
     write_json(ROOT/'RUN_STATE.json',status);log(status)
    if step%cfg['eval_every']==0 or step==cfg['steps']:
     metrics,rows=evaluate(model,base,dev,device,arm=='FULL');score=metrics['FULL' if arm=='FULL' else 'NO_UPDATE']['dice']
     record={'step':step,'metrics':metrics,'score':score};write_json(run/f'validation_{step}.json',record)
     log({'validation':record,'arm':arm,'seed':seed})
     if score>best:
      best=score;torch.save({'model':model.state_dict(),'seed':seed,'arm':arm,'step':step},run/'best.pt');write_json(run/'best_metrics.json',record);write_json(run/'best_rows.private.json',rows)
     torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'arm':arm,'step':step,'counts':COUNTS},run/'last.pt')
   torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'arm':arm,'step':step,'counts':COUNTS},run/'last.pt')
   write_json(run/'completion.json',{'status':'STOPPED' if STOP else 'COMPLETE','steps':step,'best_dev_dice':best,'seconds':time.time()-t0})
 write_json(ROOT/'TRAINING_FINISHED.json',{'status':'STOPPED' if STOP else 'COMPLETE','counts':COUNTS,'at':time.time()})
if __name__=='__main__':main()
