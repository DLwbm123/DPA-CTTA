"""Medical rollout inference: full versus truncated trajectory gradients."""
import os,json,time,random,copy,importlib.util
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

R=Path(os.environ['WORK_ROOT']);C=json.loads((R/'config.private.json').read_text());P=Path(C['parent_root'])
spec=importlib.util.spec_from_file_location('base',P/'src/train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
EXTRA={'optimizer_steps':0,'correction_forwards':0,'synthetic_optimizer_steps':0,'synthetic_forwards':0}

class Corrector(nn.Module):
    def __init__(self):
        super().__init__();self.a=m.Block(24,32);self.b=m.Block(32,64,2);self.c=m.Block(64,96,2)
        self.ub=m.Up(96,64,64);self.ua=m.Up(64,32,32);self.out=nn.Conv2d(32,2,3,1,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,feature,d,anchor,sigma,synthetic=False):
        EXTRA['synthetic_forwards' if synthetic else 'correction_forwards']+=1
        def small(z):return F.interpolate(z,(64,64),mode='bilinear',align_corners=False)
        t=sigma.reshape(-1,1,1,1).expand(len(x),1,64,64)
        a=self.a(torch.cat([small(x),small(feature),small(d),small(anchor),t],1));b=self.b(a);c=self.c(b)
        delta=F.interpolate(self.out(self.ua(self.ub(c,b),a)),d.shape[-2:],mode='bilinear',align_corners=False)
        return (d+.25*delta.tanh()).clamp(-1,1)

def loss(d,y,dt):
    w=1+4*torch.exp(-8*dt.abs())
    return m.segloss(d,y,dt)+.5*((d-dt).square()*w).mean()

def check():
    if m.STOP:raise InterruptedError('Stop requested; no automatic replay')
    if time.time()>=C['campaign_deadline_unix']-120:raise InterruptedError('Campaign deadline')

def selfcheck():
    torch.manual_seed(771);n=Corrector().cuda();x=torch.rand(1,3,64,64,device='cuda');f=torch.randn(1,16,64,64,device='cuda')
    d=torch.zeros(1,2,64,64,device='cuda');t=torch.ones(1,device='cuda')*.1
    q=torch.arange(64,device='cuda');yy,xx=torch.meshgrid(q,q,indexing='ij');rad=((xx-32).square()+(yy-32).square()).float().sqrt()
    dt=torch.stack([(rad-24)/16,(rad-11)/16])[None].clamp(-1,1);y=(dt<0).float()
    out=n(x,f,d,d,t,True);assert torch.equal(out,d)
    o=torch.optim.SGD(n.parameters(),lr=.01);loss(out,y,dt).backward()
    assert n.out.weight.grad.isfinite().all() and n.out.weight.grad.abs().sum()>0
    o.step();EXTRA['synthetic_optimizer_steps']+=1
    after=n(x,f,d,d,t,True);assert after.isfinite().all() and not torch.equal(after,d)
    left=copy.deepcopy(n);right=copy.deepcopy(n)
    gradients=[];outputs=[]
    for net,truncated in [(left,True),(right,False)]:
        net.zero_grad(set_to_none=True);state=d;values=[]
        for scale in [.20,.10,.05]:
            state=net(x,f,state,d,x.new_full((1,),scale),True);values.append(loss(state,y,dt))
            if truncated:state=state.detach()
        outputs.append(state.detach());(sum(values)/3).backward()
        gradients.append(net.out.weight.grad.detach().clone())
    assert torch.allclose(outputs[0],outputs[1],atol=1e-6,rtol=1e-5)
    assert not torch.allclose(gradients[0],gradients[1],atol=1e-7,rtol=1e-4)
    m.write(R/'SELF_CHECK.json',{'synthetic_only':True,'same_forward_different_trajectory_gradient':True,'identity_at_initialization':True,'nonzero_trainable_gradient':True,'finite_changed_output':True,'extra_counts':EXTRA.copy(),'base_counts':m.COUNTS.copy()})

def save(run,nets,opts,rng,seed,step):
    value={'models':{k:v.state_dict() for k,v in nets.items()},'optimizers':{k:v.state_dict() for k,v in opts.items()},'seed':seed,'completed_paired_episode':step,'rng':rng.getstate(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'base_counts':m.COUNTS.copy(),'extra_counts':EXTRA.copy()}
    p=run/'latest.pt';torch.save(value,p.with_suffix('.tmp'));p.with_suffix('.tmp').replace(p)

@torch.no_grad()
def evaluate(base,nets,dev,run):
    before={name:{k:v.detach().clone() for k,v in net.state_dict().items()} for name,net in nets.items()}
    for net in nets.values():net.eval();net.requires_grad_(False)
    rows=[];ids=list(range(len(dev)));random.Random(8821).shuffle(ids)
    for kind in ['clean','gamma','color','blur']:
        for i in ids:
            check();xx,yy,_=dev[i];x=m.corrupt(xx[None].cuda(),kind);ctx,_=base.encode(x);d0=base.head(ctx,(256,256));f=ctx[0]
            pred={'ZERO':d0};direct=d0;denoise=d0;abl=d0
            for step,sigma in enumerate([.20,.10,.05]):
                direct=nets['DETACHED'](x,f,direct,d0,x.new_full((1,),sigma));denoise=nets['FULL'](x,f,denoise,d0,x.new_full((1,),sigma))
                abl=nets['FULL'](torch.zeros_like(x),torch.zeros_like(f),abl,d0,x.new_full((1,),sigma))
                if step==0:pred.update(DETACHED1=direct,FULL1=denoise)
            pred.update(DETACHED3=direct,FULL3=denoise,FULL3_NO_EXTRA_IMAGE=abl)
            rows.append({'content':i,'condition':kind,**{arm:m.metrics(d,yy) for arm,d in pred.items()}})
        m.write(run/'detailed.private.json',rows)
    assert all(torch.equal(before[name][k],v) for name,net in nets.items() for k,v in net.state_dict().items())
    m.write(run/'metrics.json',{arm:{key:sum(row[arm][key] for row in rows)/len(rows) for key in ['dice','od','oc','vcdr_error']} for arm in pred})
    return len(rows)

def main():
    assert not (R/'runs').exists(),'Refuse automatic replay or overwrite'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selfcheck();check();fit,dev=m.load_data()
    m.write(R/'DATA_RECEIPT.json',{'fit':len(fit),'dev_final_only':len(dev),'target_reads':0,'patient_identity':'UNKNOWN','repeated_dev':True,'old_weights_read_only':True})
    completed=[]
    for seed in C['seeds']:
        check();torch.manual_seed(seed);rng=random.Random(seed+60000)
        base=m.Model().cuda().eval();ck=torch.load(P/f'runs/{seed}/latest.pt',map_location='cpu',weights_only=False)
        assert ck['phase']=='meta' and ck['step']==500;base.load_state_dict(ck['model'],strict=True);del ck
        base.requires_grad_(False);original=[p.detach().clone() for p in base.parameters()]
        direct=Corrector().cuda();nets={'DETACHED':direct,'FULL':copy.deepcopy(direct)}
        opts={k:torch.optim.Adam(net.parameters(),lr=C['correction_lr']) for k,net in nets.items()}
        run=R/'runs'/str(seed);run.mkdir(parents=True,exist_ok=False)
        for step in range(1,C['episodes']+1):
            check();x,y,dt=m.batch(fit,rng,C['correction_batch']);x,_=m.paired(x)
            with torch.no_grad():ctx,_=base.encode(x);d0=base.head(ctx,(256,256));f=ctx[0]
            losses={}
            for name in ['DETACHED','FULL']:
                current=d0;values=[]
                for sigma in [.20,.10,.05]:
                    current=nets[name](x,f,current,d0,x.new_full((len(x),),sigma))
                    values.append(loss(current,y,dt))
                    if name=='DETACHED':current=current.detach()
                value=sum(values)/3
                assert torch.isfinite(value),'Nonfinite medical trajectory loss'
                opts[name].zero_grad(set_to_none=True);value.backward();assert all(p.grad is None for p in base.parameters())
                torch.nn.utils.clip_grad_norm_(nets[name].parameters(),5,error_if_nonfinite=True)
                opts[name].step();EXTRA['optimizer_steps']+=1;losses[name]=float(value.detach())
            if step%25==0:
                save(run,nets,opts,rng,seed,step)
                m.write(R/'RUN_STATE.json',{'seed':seed,'episode':step,'episodes_per_arm':C['episodes'],'losses':losses,'base_counts':m.COUNTS.copy(),'extra_counts':EXTRA.copy(),'completed_seeds':completed,'updated_unix':time.time()})
                print(json.dumps({'seed':seed,'episode':step,'losses':losses}),flush=True)
        assert all(torch.equal(a,p) for a,p in zip(original,base.parameters()))
        coverage=evaluate(base,nets,dev,run)
        assert all(torch.equal(a,p) for a,p in zip(original,base.parameters()))
        m.write(run/'completion.json',{'episodes_per_arm':C['episodes'],'optimizer_steps_per_arm':C['episodes'],'arms':['DETACHED','FULL'],'parameters_per_arm':sum(p.numel() for p in direct.parameters()),'coverage':coverage,'base_frozen':True,'readout_weights_frozen':True})
        completed.append(seed);del original,base,nets,opts,direct;torch.cuda.empty_cache()
    assert EXTRA['optimizer_steps']==4000
    m.write(R/'TRAINING_FINISHED.json',{'status':'COMPLETE','completed_seeds':completed,'base_counts':m.COUNTS,'extra_counts':EXTRA,'ended_unix':time.time(),'peak_cuda_allocated':torch.cuda.max_memory_allocated()})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        m.write(R/'FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'base_counts':m.COUNTS,'extra_counts':EXTRA,'unix':time.time()});raise
