"""Conditional categorical anatomy generator, fixed three-call comparisons."""
import os,json,time,random,importlib.util
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
R=Path(os.environ['WORK_ROOT']);C=json.loads((R/'config.private.json').read_text());P=Path(C['parent_root'])
spec=importlib.util.spec_from_file_location('base',P/'src/train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
COUNTS={'updates':0,'endpoint_forwards':0,'synthetic_forwards':0,'synthetic_backward':0}
def check():
 if m.STOP or time.time()>=C['campaign_deadline_unix']-120:raise InterruptedError('Budget stop, no replay')
def prior(d):
 p=(-16*d).sigmoid();cup=torch.minimum(p[:,0],p[:,1]);q=torch.stack([1-p[:,0],p[:,0]-cup,cup],1).clamp_min(1e-6);return q/q.sum(1,keepdim=True)
def sample(q,g=None):
 u=torch.rand(q.shape[0],*q.shape[2:],device=q.device,generator=g);return (u[:,None]>q.cumsum(1)).sum(1).clamp_max(2)
def kernel(q,z,a):return (1-a)*F.one_hot(z,3).permute(0,3,1,2).to(q)+a*q
def distance(z):return torch.stack([torch.where(z>0,-1.,1.),torch.where(z==2,-1.,1.)],1)
class Endpoint(nn.Module):
 def __init__(self):
  super().__init__();self.a=m.Block(26,32);self.b=m.Block(32,64,2);self.c=m.Block(64,96,2);self.ub=m.Up(96,64,64);self.ua=m.Up(64,32,32);self.out=nn.Conv2d(32,3,3,padding=1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,x,f,z,q,t,synthetic=False):
  COUNTS['synthetic_forwards' if synthetic else 'endpoint_forwards']+=1
  def small(a):return F.interpolate(a,(64,64),mode='bilinear',align_corners=False)
  a=self.a(torch.cat([small(x),small(f),small(F.one_hot(z,3).permute(0,3,1,2).float()),small(q),t[:,None,None,None].expand(-1,1,64,64)],1));b=self.b(a);c=self.c(b);delta=F.interpolate(self.out(self.ua(self.ub(c,b),a)),q.shape[-2:],mode='bilinear',align_corners=False);return q.log()+delta

def selfcheck():
 torch.manual_seed(773);d=torch.randn(2,2,64,64,device='cuda');q=prior(d);assert (q>0).all() and torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-6)
 g=torch.Generator(device='cuda').manual_seed(1);z=sample(q,g);g.manual_seed(1);assert torch.equal(z,sample(q,g))
 for a in [0.,1/3,1/2,1.]:
  k=kernel(q,z,a);assert (k>=0).all() and torch.allclose(k.sum(1),torch.ones_like(k[:,0]),atol=1e-6)
 assert torch.equal(kernel(q,z,1.),q)
 n=Endpoint().cuda();v=n(torch.rand(2,3,64,64,device='cuda'),torch.rand(2,16,64,64,device='cuda'),z,q,torch.zeros(2,device='cuda'),True);assert torch.allclose(v.softmax(1),q,atol=1e-6)
 F.cross_entropy(v,torch.zeros_like(z)).backward();COUNTS['synthetic_backward']+=1;assert n.out.weight.grad.isfinite().all() and n.out.weight.grad.abs().sum()>0
 m.write(R/'SELF_CHECK.json',{'synthetic_only':True,'probability_conservation':True,'endpoint_kernel':True,'reproducible_sampling':True,'initial_prior':True,'nonzero_gradient':True,'counts':COUNTS.copy()})

@torch.no_grad()
def evaluate(base,net,dev,run,seed):
 net.eval();net.requires_grad_(False);before={k:v.detach().clone() for k,v in net.state_dict().items()};rows=[]
 for ci,kind in enumerate(['clean','gamma','color','blur']):
  for i,(xx,yy,_) in enumerate(dev):
   check();x=m.corrupt(xx[None].cuda(),kind);ctx,_=base.encode(x);d0=base.head(ctx,(256,256));q=prior(d0);f=ctx[0];g=torch.Generator(device='cuda').manual_seed(seed+80000+ci*1000+i)
   starts=[sample(q,g) for _ in range(3)];zero=x.new_zeros(1);p0=net(x,f,starts[0],q,zero).softmax(1)
   ensemble=(p0+net(x,f,starts[1],q,zero).softmax(1)+net(x,f,starts[2],q,zero).softmax(1))/3
   z=sample(kernel(p0,starts[0],1/3),g);p1=net(x,f,z,q,x.new_full((1,),1/3)).softmax(1);z=sample(kernel(p1,z,1/2),g);last=net(x,f,z,q,x.new_full((1,),2/3)).softmax(1)
   pred={'ZERO':d0,'STRUCT_ZERO':distance(q.argmax(1)),'DIRECT1':distance(p0.argmax(1)),'ENSEMBLE3':distance(ensemble.argmax(1)),'CHAIN3':distance(last.argmax(1)),'CHAIN3_SAMPLE':distance(sample(last,g))}
   rows.append({'content':i,'condition':kind,**{a:m.metrics(d,yy) for a,d in pred.items()}})
  m.write(run/'detailed.private.json',rows)
 assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
 m.write(run/'metrics.json',{a:{k:sum(row[a][k] for row in rows)/len(rows) for k in ['dice','od','oc','vcdr_error']} for a in pred});return len(rows)

def main():
 assert not (R/'runs').exists(),'Refuse automatic replay'
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;selfcheck();check();fit,dev=m.load_data()
 m.write(R/'DATA_RECEIPT.json',{'fit':len(fit),'dev_final_only':len(dev),'target':0,'patient_identity':'UNKNOWN','repeated_dev':True});completed=[]
 for seed in [10225,10226]:
  torch.manual_seed(seed);rng=random.Random(seed+70000);base=m.Model().cuda().eval();ck=torch.load(P/f'runs/{seed}/latest.pt',map_location='cpu',weights_only=False);assert ck['phase']=='meta' and ck['step']==500;base.load_state_dict(ck['model']);del ck;base.requires_grad_(False);before=[p.detach().clone() for p in base.parameters()]
  net=Endpoint().cuda();opt=torch.optim.Adam(net.parameters(),lr=.001);run=R/'runs'/str(seed);run.mkdir(parents=True)
  for step in range(1,1001):
   check();x,y,_=m.batch(fit,rng,4);x,_=m.paired(x)
   with torch.no_grad():ctx,_=base.encode(x);q=prior(base.head(ctx,(256,256)));z0=sample(q);target=y[:,0].long()+y[:,1].long();t=torch.rand(len(x),device='cuda');z=torch.where(torch.rand_like(z0.float())<t[:,None,None],target,z0)
   value=F.cross_entropy(net(x,ctx[0],z,q,t),target);assert torch.isfinite(value);opt.zero_grad(set_to_none=True);value.backward();assert all(p.grad is None for p in base.parameters());torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();COUNTS['updates']+=1
   if step%25==0:
    p=run/'latest.pt';torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'completed_step':step,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'sampler_rng':rng.getstate(),'counts':COUNTS.copy()},p.with_suffix('.tmp'));p.with_suffix('.tmp').replace(p)
    m.write(R/'RUN_STATE.json',{'seed':seed,'step':step,'loss':float(value),'completed_seeds':completed,'counts':COUNTS.copy(),'unix':time.time()})
  assert all(torch.equal(v,p) for v,p in zip(before,base.parameters()));n=evaluate(base,net,dev,run,seed);assert n==80 and all(torch.equal(v,p) for v,p in zip(before,base.parameters()))
  m.write(run/'completion.json',{'optimizer_steps':1000,'base_frozen':True,'readout_weights_frozen':True,'coverage':80,'parameters':sum(p.numel() for p in net.parameters())});completed.append(seed);del base,net,opt,before;torch.cuda.empty_cache()
 assert COUNTS['updates']==2000 and COUNTS['endpoint_forwards']==2800
 m.write(R/'TRAINING_FINISHED.json',{'status':'COMPLETE','completed_seeds':completed,'counts':COUNTS,'base_counts':m.COUNTS,'ended_unix':time.time(),'peak_cuda_allocated':torch.cuda.max_memory_allocated()})
if __name__=='__main__':
 try:main()
 except BaseException as e:m.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'counts':COUNTS,'unix':time.time()});raise
