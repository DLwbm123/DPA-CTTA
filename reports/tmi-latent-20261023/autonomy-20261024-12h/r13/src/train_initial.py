"""Paired source training; no historical weights or test-time updates."""
import os,json,time,random,copy,importlib.util
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
R=Path(os.environ['WORK_ROOT']); C=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('base',Path(C['parent_root'])/'src/train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ARMS=['INVARIANT','SHARED','SHUFFLED'];COUNT={'updates':0,'training_encoder_images':0,'evaluation_encoder_images':0,'synthetic_forwards':0,'synthetic_backward':0}
def check():
 if m.STOP or time.time()>=C['campaign_deadline_unix']-120: raise InterruptedError('Budget stop; no replay')
class Net(m.Model):
 def __init__(self):
  super().__init__()
  for k in ['relative_style','ru8','ru4','mod16','mod8','mod4','rgb']:delattr(self,k)
  self.style=nn.Linear(64,4)
 def forward(self,x):
  c,s=self.encode(x);return self.head(c,x.shape[-2:]),torch.cat([c[0].mean((-2,-1)),s],1)
def matrix(a,b,mode):
 x=a[:6,16:];y=b[:6,16:];eye=torch.eye(4,device=x.device)
 if mode=='SHUFFLED':y=y.roll(1,0)
 rr=torch.linalg.solve(x.T@x+.01*eye,x.T@y+.01*eye)
 return eye if mode=='INVARIANT' else rr,rr

def contrast(a,b,mode):
 losses=[];stats=[]
 for x,y in [(a,b),(b,a)]:
  mat,raw=matrix(x,y,mode);q=torch.cat([x[6:,:16],x[6:,16:]@mat],1)
  logits=-(q[:,None]-y[None]).square().mean(-1)/.1
  losses.append(F.cross_entropy(logits,torch.tensor([6,7],device=x.device)))
  with torch.no_grad():
   sv=torch.linalg.svdvals(raw);stats.append({'support_rank':int(torch.linalg.matrix_rank(x[:6,16:])),'matrix_smin':float(sv.min()),'matrix_smax':float(sv.max()),'query_mse':float((q-y[6:]).square().mean())})
 return sum(losses)/2,stats

def selfcheck():
 torch.manual_seed(882);net=Net().cuda();x=torch.rand(8,3,64,64,device='cuda');g=torch.tensor([.9,1.,1.1],device='cuda')[None,:,None,None]
 assert torch.allclose(x.pow(g).pow(1/g),x,atol=2e-6)
 d,a=net(x);_,b=net(x.pow(g));COUNT['synthetic_forwards']+=2
 loss,st=contrast(a,b,'SHARED');loss.backward();COUNT['synthetic_backward']+=1
 assert torch.isfinite(loss) and net.style.weight.grad is not None and net.style.weight.grad.abs().sum()>0
 assert net.stem[0].weight.grad.abs().sum()>0
 probe=torch.randn(8,20,device='cuda');known=torch.diag(torch.tensor([.8,1.1,.9,1.2],device='cuda'));dest=probe.clone();dest[:,16:]=probe[:,16:]@known
 mat,_=matrix(probe,dest,'SHARED');wrong,_=matrix(probe,dest,'SHUFFLED')
 assert (probe[6:,16:]@mat-dest[6:,16:]).square().mean()<(probe[6:,16:]@wrong-dest[6:,16:]).square().mean()
 other=copy.deepcopy(net);assert all(torch.equal(a,b) for a,b in zip(net.parameters(),other.parameters()))
 m.write(R/'SELF_CHECK.json',{'synthetic_only':True,'gamma_inverse_tolerance':2e-6,'support_query_and_wrong_pair':True,'nonzero_encoder_style_gradient':True,'paired_initialization':True,'counts':COUNT.copy()});del net,other
 torch.cuda.empty_cache()

@torch.no_grad()
def evaluate(nets,dev,run):
 before={k:{a:v.detach().cpu().clone() for a,v in n.state_dict().items()} for k,n in nets.items()}
 for n in nets.values():n.eval();n.requires_grad_(False)
 rows=[]
 for kind in ['clean','gamma','color','blur']:
  for i,(xx,yy,_) in enumerate(dev):
   check();x=m.corrupt(xx[None].cuda(),kind);row={'content':i,'condition':kind}
   for name,n in nets.items():d,_=n(x);COUNT['evaluation_encoder_images']+=1;row[name]=m.metrics(d,yy)
   rows.append(row)
  m.write(run/'detailed.private.json',rows)
 assert all(torch.equal(before[k][a],v.cpu()) for k,n in nets.items() for a,v in n.state_dict().items())
 return len(rows)

def main():
 assert not (R/'runs').exists(),'Refuse replay of existing runs'
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 selfcheck();fit,dev=m.load_data();m.write(R/'DATA_RECEIPT.json',{'fit':len(fit),'dev_final_only':len(dev),'target':0,'historical_weights_loaded':False,'patient_identity':'UNKNOWN','development_reused':True})
 completed=[]
 for seed in [10225,10226]:
  check();torch.manual_seed(seed);rng=random.Random(seed+110000);gen=torch.Generator(device='cuda').manual_seed(seed+120000)
  net=Net().cuda();nets={ARMS[0]:net,**{k:copy.deepcopy(net) for k in ARMS[1:]}};opts={k:torch.optim.AdamW(n.parameters(),lr=.0003,weight_decay=.0001) for k,n in nets.items()}
  assert all(torch.equal(a,b) for n in nets.values() for a,b in zip(net.parameters(),n.parameters()))
  run=R/'runs'/str(seed);run.mkdir(parents=True);m.write(run/'MODEL.json',{'parameters_per_arm':sum(p.numel() for p in net.parameters()),'architecture_paired':True})
  rank_deficient={k:0 for k in ARMS}
  for step in range(1,2001):
   check();x,y,dt=m.batch(fit,rng,8);gamma=torch.empty((1,3,1,1),device='cuda').uniform_(-.2,.2,generator=gen).exp();xb=x.pow(gamma);losses={};diagnostics={}
   for name in ARMS:
    check();n=nets[name];da,a=n(x);db,b=n(xb);COUNT['training_encoder_images']+=16
    med=(m.segloss(da,y,dt)+m.segloss(db,y,dt))/2;aux,st=contrast(a,b,name);loss=med+.05*aux
    assert torch.isfinite(loss),'Nonfinite loss'
    opts[name].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),5,error_if_nonfinite=True);opts[name].step();COUNT['updates']+=1
    rank_deficient[name]+=sum(v['support_rank']<4 for v in st)
    losses[name]={'medical':float(med.detach()),'contrastive':float(aux.detach()),'total':float(loss.detach())};diagnostics[name]=st
   if step==1 or step%25==0:
    m.write(R/'RUN_STATE.json',{'seed':seed,'step':step,'completed_seeds':completed,'losses':losses,'action_diagnostics':diagnostics,'rank_deficient_direction_count':rank_deficient,'counts':COUNT.copy(),'unix':time.time()})
   if step%100==0:
    p=run/'latest.pt';torch.save({'models':{k:n.state_dict() for k,n in nets.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'completed_step':step,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'gamma_rng':gen.get_state(),'sampler_rng':rng.getstate(),'counts':COUNT.copy()},p.with_suffix('.tmp'));p.with_suffix('.tmp').replace(p)
  coverage=evaluate(nets,dev,run);assert coverage==80
  m.write(run/'completion.json',{'optimizer_steps_per_arm':2000,'source_training_from_scratch':True,'readout_weights_frozen':True,'coverage':80,'rank_deficient_direction_count':rank_deficient,'parameters_per_arm':sum(p.numel() for p in net.parameters())});completed.append(seed)
  del net,nets,opts,n,da,db,a,b,loss,aux,med;torch.cuda.empty_cache()
 assert COUNT['updates']==12000 and COUNT['training_encoder_images']==192000 and COUNT['evaluation_encoder_images']==480
 m.write(R/'TRAINING_FINISHED.json',{'status':'COMPLETE','completed_seeds':completed,'counts':COUNT,'base_counts':m.COUNTS,'ended_unix':time.time(),'peak_cuda_allocated':torch.cuda.max_memory_allocated()})
if __name__=='__main__':
 try:main()
 except BaseException as e:m.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'counts':COUNT,'unix':time.time()});raise
