import os,json,time,random,copy,importlib.util
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates,distance_transform_edt
import torch
from torch import nn
import torch.nn.functional as F
R=Path(os.environ['WORK_ROOT']);C=json.loads((R/'config.private.json').read_text());P=Path(C['parent_root']);spec=importlib.util.spec_from_file_location('base',P/'src/train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
COUNT={'updates':0,'forwards':0,'synthetic_forwards':0,'synthetic_backward':0,'train_anchor_fallback':0,'eval_anchor_fallback':0,'decode_clipped_coordinates':0,'encode_clipped_coordinates':0}
A=np.arange(64)*2*np.pi/64;RAD=np.arange(0,np.hypot(256,256)+1,.25);LO=np.r_[[-np.log(511)]*2,[np.log(.25/64)]*64,[-np.log(9999)]*64];HI=np.r_[[np.log(511)]*2,[np.log(np.hypot(256,256)/64)]*64,[np.log(9999)]*64]
def check():
 if m.STOP or time.time()>=C['campaign_deadline_unix']-120:raise InterruptedError('Budget stop; no replay')
def encode(y):
 y=np.asarray(y,dtype=bool);pts=np.argwhere(y[1]);fallback=False
 center=pts.mean(0) if len(pts) else np.array([127.5,127.5]);cy,cx=center;idx=tuple(np.clip(np.round(center).astype(int),0,255));inside=y[0]&y[1]
 if not (len(pts) and inside[idx]):
  fallback=True;support=inside if inside.any() else y[0]
  center=np.array(np.unravel_index(distance_transform_edt(support).argmax(),support.shape),dtype=float) if support.any() else np.array([127.5,127.5]);cy,cx=center
 radii=[]
 for mask in y:
  vals=map_coordinates(mask.astype(float),[cy+np.sin(A)[:,None]*RAD,cx+np.cos(A)[:,None]*RAD],order=1,mode='constant',cval=0);out=vals<.5;first=out.argmax(1);rr=(RAD[first]+RAD[np.maximum(first-1,0)])/2;rr=np.where((first>0)&out.any(1),rr,.25);radii.append(rr.clip(.25,np.hypot(256,256)))
 radii[1]=np.minimum(radii[1],radii[0]);rawc=center/255;rawratio=radii[1]/radii[0];COUNT['encode_clipped_coordinates']+=int(((rawc<1/512)|(rawc>511/512)).sum()+((rawratio<1e-4)|(rawratio>1-1e-4)).sum());c=np.clip(rawc,1/512,511/512);ratio=np.clip(rawratio,1e-4,1-1e-4);return np.r_[np.log(c/(1-c)),np.log(radii[0]/64),np.log(ratio/(1-ratio))].astype(np.float32),fallback

def decode(z):
 z=np.asarray(z,dtype=float);COUNT['decode_clipped_coordinates']+=int(((z<LO)|(z>HI)).sum());z=np.clip(z,LO,HI);cy,cx=255/(1+np.exp(-z[:2]));od=64*np.exp(z[2:66]);oc=od/(1+np.exp(-z[66:]));yy,xx=np.indices((256,256));angle=np.mod(np.arctan2(yy-cy,xx-cx),2*np.pi);radius=np.hypot(yy-cy,xx-cx);return np.stack([radius<=np.interp(angle,A,r,period=2*np.pi) for r in [od,oc]])
def dfield(z):return torch.tensor(np.where(decode(z),-1.,1.),dtype=torch.float32)[None]
def vectors(masks):
 z=[];flags=[]
 for y in masks:zv,f=encode(y);z.append(zv);flags.append(f)
 return torch.tensor(np.stack(z),device='cuda'),flags
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Linear(1477,512),nn.GELU(),nn.Linear(512,256),nn.GELU(),nn.Linear(256,130));nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
 def forward(self,c,anchor,z,t,synthetic=False):
  COUNT['synthetic_forwards' if synthetic else 'forwards']+=1;return self.layers(torch.cat([c,anchor,z,t[:,None]],1))
def condition(x,ctx):return torch.cat([F.adaptive_avg_pool2d(x,8).flatten(1),F.adaptive_avg_pool2d(ctx[0],8).flatten(1)],1)
def selfcheck():
 yy,xx=np.indices((256,256));y=np.stack([(xx-125)**2+(yy-130)**2<70**2,((xx-130)/32)**2+((yy-126)/29)**2<1]);z,f=encode(y);assert not f;d=dfield(z);v=m.metrics(d,torch.tensor(y));assert v['od']>98 and v['oc']>98 and v['cup_outside']==0
 n=Net().cuda();other=copy.deepcopy(n);assert all(torch.equal(a,b) for a,b in zip(n.parameters(),other.parameters()));c=torch.zeros(1,1216,device='cuda');a=torch.tensor(z,device='cuda')[None];v=n(c,a,a,torch.zeros(1,device='cuda'),True);assert torch.equal(v,torch.zeros_like(v));(v-.1).square().mean().backward();COUNT['synthetic_backward']+=1;assert n.layers[-1].weight.grad.abs().sum()>0
 g=torch.Generator(device='cuda').manual_seed(33);u=torch.randn((1,130),generator=g,device='cuda');g.manual_seed(33);assert torch.equal(u,torch.randn((1,130),generator=g,device='cuda'));m.write(R/'SELF_CHECK.json',{'synthetic_only':True,'representation_roundtrip':True,'containment':True,'paired_initialization':True,'gradient':True,'rng':True,'counts':COUNT.copy()})
@torch.no_grad()
def evaluate(base,nets,dev,run,seed):
 before={k:{a:v.clone() for a,v in net.state_dict().items()} for k,net in nets.items()}
 for net in nets.values():net.eval();net.requires_grad_(False)
 rows=[]
 for ci,kind in enumerate(['clean','gamma','color','blur']):
  for i,(xx,yy,_) in enumerate(dev):
   check();x=m.corrupt(xx[None].cuda(),kind);ctx,_=base.encode(x);d0=base.head(ctx,(256,256));a,flags=vectors((d0<0).cpu().numpy());COUNT['eval_anchor_fallback']+=sum(flags);cond=condition(x,ctx);g=torch.Generator(device='cuda').manual_seed(seed+100000+ci*1000+i);starts=[a+.1*torch.randn(a.shape,device='cuda',generator=g) for _ in range(3)];zero=x.new_zeros(1)
   direct=[a+nets['DIRECT'](cond,a,z,zero) for z in starts];state=starts[0];first=nets['FLOW'](cond,a,state,zero);flow1=state+first;state=state+first/3
   for t in [1/3,2/3]:state=state+nets['FLOW'](cond,a,state,x.new_full((1,),t))/3
   latent={'STRUCT_ZERO':a,'DIRECT1':direct[0],'DIRECT3':sum(direct)/3,'FLOW1':flow1,'FLOW3':state};pred={'ZERO':d0,**{k:dfield(z[0].cpu().numpy()) for k,z in latent.items()}}
   rows.append({'content':i,'condition':kind,'anchor_fallback':bool(flags[0]),**{k:m.metrics(d,yy) for k,d in pred.items()}})
  m.write(run/'detailed.private.json',rows)
 assert all(torch.equal(before[k][a],v) for k,net in nets.items() for a,v in net.state_dict().items());return len(rows)
def main():
 assert not (R/'runs').exists(),'Refuse replay';torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;selfcheck();fit,dev=m.load_data();m.write(R/'DATA_RECEIPT.json',{'fit':len(fit),'dev_final_only':len(dev),'target':0,'truth_centers_only_training_targets':True,'patient_identity':'UNKNOWN'});completed=[]
 for seed in [10225,10226]:
  check();torch.manual_seed(seed);rng=random.Random(seed+90000);base=m.Model().cuda().eval();ck=torch.load(P/f'runs/{seed}/latest.pt',map_location='cpu',weights_only=False);assert ck['phase']=='meta' and ck['step']==500;base.load_state_dict(ck['model']);del ck;base.requires_grad_(False);before=[p.clone() for p in base.parameters()];net=Net().cuda();nets={'DIRECT':net,'FLOW':copy.deepcopy(net)};opts={k:torch.optim.Adam(n.parameters(),lr=.001) for k,n in nets.items()};run=R/'runs'/str(seed);run.mkdir(parents=True)
  for step in range(1,1001):
   check();x,y,_=m.batch(fit,rng,4);x,_=m.paired(x)
   with torch.no_grad():
    ctx,_=base.encode(x);d0=base.head(ctx,(256,256));a,flags=vectors((d0<0).cpu().numpy());COUNT['train_anchor_fallback']+=sum(flags);target,tf=vectors(y.cpu().numpy());assert not any(tf),'Source target representation invalid; no substitution';cond=condition(x,ctx);z0=a+.1*torch.randn_like(a);t=torch.rand(len(x),device='cuda');zt=(1-t[:,None])*z0+t[:,None]*target
   losses={}
   for name in ['DIRECT','FLOW']:
    out=nets[name](cond,a,z0 if name=='DIRECT' else zt,torch.zeros_like(t) if name=='DIRECT' else t);loss=(out-(target-a if name=='DIRECT' else target-z0)).square().mean();assert torch.isfinite(loss);opts[name].zero_grad(set_to_none=True);loss.backward();assert all(p.grad is None for p in base.parameters());torch.nn.utils.clip_grad_norm_(nets[name].parameters(),5,error_if_nonfinite=True);opts[name].step();COUNT['updates']+=1;losses[name]=float(loss.detach())
   if step%25==0:
    p=run/'latest.pt';torch.save({'models':{k:n.state_dict() for k,n in nets.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'completed_step':step,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'sampler_rng':rng.getstate(),'counts':COUNT.copy()},p.with_suffix('.tmp'));p.with_suffix('.tmp').replace(p);m.write(R/'RUN_STATE.json',{'seed':seed,'step':step,'losses':losses,'completed_seeds':completed,'counts':COUNT.copy(),'unix':time.time()})
  assert all(torch.equal(v,p) for v,p in zip(before,base.parameters()));coverage=evaluate(base,nets,dev,run,seed);assert coverage==80 and all(torch.equal(v,p) for v,p in zip(before,base.parameters()));m.write(run/'completion.json',{'optimizer_steps_per_arm':1000,'base_frozen':True,'readout_weights_frozen':True,'coverage':80,'parameters_per_arm':sum(p.numel() for p in net.parameters())});completed.append(seed);del net,nets,opts,base,before;torch.cuda.empty_cache()
 assert COUNT['updates']==4000 and COUNT['forwards']==4960;m.write(R/'TRAINING_FINISHED.json',{'status':'COMPLETE','completed_seeds':completed,'counts':COUNT,'base_counts':m.COUNTS,'ended_unix':time.time(),'peak_cuda_allocated':torch.cuda.max_memory_allocated()})
if __name__=='__main__':
 try:main()
 except BaseException as e:m.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'counts':COUNT,'unix':time.time()});raise
