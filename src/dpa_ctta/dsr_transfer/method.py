"""DSR equations 2-7 and appendices A.5/A.6, transferred to dense Bernoulli outputs.

Public submission: https://openreview.net/forum?id=dMBuehC5eR (desk rejected).
Host update/prediction timing stays CW_LSO; this is not full-paper reproduction.
"""
import copy
import torch
from torch.nn import functional as F
from ..r37_cw_orientation.method import Host as RetainedHost, candidates

ARMS=('NATIVE','SOFT','ALWAYS','DSR')


def configuration(arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    c=copy.deepcopy(candidates()[3]);c.update(id='CW_LSO_'+arm,dsr_arm=arm)
    if arm!='NATIVE':
        c['components']={'W':copy.deepcopy(c['params']), 'T':dict(teacher_mix=1.,ema_alpha=.999)}
    return c


class Regulation:
    def __init__(self):
        self.steps=0;self.history=torch.empty((2,0),dtype=torch.float64)
        self.positive=torch.full((2,),.8,dtype=torch.float64)
        self.reference=0.;self.warm_sum=0.

    def observe(self,q,discrepancy):
        # 64 fixed spatial locations/image preserve temporal history at the paper's 4096 cap.
        positions=torch.arange(32,512,64,device=q.device)
        sample=q[0,:,positions[:,None],positions[None,:]].reshape(2,-1).detach().double().cpu()
        self.history=torch.cat((self.history,sample),1)[:,-4096:]
        warm=self.steps<20;before=self.reference
        if warm:
            self.warm_sum+=discrepancy;gate=False
            if self.steps==19:self.reference=self.warm_sum/20
        else:
            candidate=torch.quantile(self.history,.95,dim=1).clamp(.60,.95)
            self.positive.mul_(.9).add_(candidate,alpha=.1)
            gate=discrepancy>1.2*before
            self.reference=.999*before+.001*discrepancy
        self.steps+=1
        return gate,warm,before

    def snapshot(self):return copy.deepcopy(self.__dict__)
    def restore(self,s):self.__dict__=copy.deepcopy(s)


def hard_loss(z,q,positive):
    positive=positive.to(z).reshape(1,2,1,1)
    pos=q>=positive;neg=q<=.3;valid=pos|neg
    p=z.sigmoid();pn=(1-p+.05).clamp(max=1)
    # ASL gamma_pos=0, gamma_neg=2; focal factors differentiated.
    term=-(pos*F.logsigmoid(z)+neg*(1-pn).square()*pn.clamp_min(1e-8).log())
    return term.sum()/valid.sum().clamp_min(1),pos,neg


class Host(RetainedHost):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.regulation=None if self.config['dsr_arm']=='NATIVE' else Regulation()

    def _prepare_teacher(self,x):
        if self.teacher is None:self.q_teacher=None;return
        with torch.no_grad():self.q_teacher=self._readonly(self.teacher,x).sigmoid()

    def _criterion(self,z,q_native):
        base=super()._criterion(z,q_native)
        if self.regulation is None:return base
        q=self.q_teacher.detach().to(z)
        p=self.weak[0].to(z).sigmoid().clamp(1e-6,1-1e-6)
        safe=q.clamp(1e-6,1-1e-6)
        divergence=float((safe*(safe/p).log()+(1-safe)*((1-safe)/(1-p)).log()).mean().clamp_min(0))
        gate,warm,reference=self.regulation.observe(q,divergence)
        arm=self.config['dsr_arm']
        active=not warm and (arm=='ALWAYS' or (arm=='DSR' and gate))
        if self.config.get('qualification_force_hard',False):active=True
        hard,pos,neg=hard_loss(z,q,self.regulation.positive)
        self.diag.update(hard_active=int(active),warmup=int(warm),bernoulli_discrepancy=divergence,
                         discrepancy_reference=reference,positive_threshold=self.regulation.positive.tolist(),
                         positive_fraction=float(pos.float().mean()),negative_fraction=float(neg.float().mean()),
                         hard_loss=float(hard.detach()),hard_coefficient=.25 if active else 0.)
        return base+.25*hard if active else base

    def snapshot(self):
        s=super().snapshot()
        if self.regulation is not None:s['dsr_regulation']=self.regulation.snapshot()
        return s

    def restore(self,s):
        super().restore(s)
        if self.regulation is not None:self.regulation.restore(s['dsr_regulation'])


def self_check():
    torch.set_num_threads(2)
    q=torch.full((1,2,512,512),.5)
    r=Regulation()
    for _ in range(20):
        gate,warm,_=r.observe(q,.1);assert warm and not gate
        assert torch.equal(r.positive,torch.full((2,),.8,dtype=torch.float64))
    assert abs(r.reference-.1)<1e-12
    gate,warm,ref=r.observe(q,.13);assert gate and not warm and abs(ref-.1)<1e-12
    assert torch.allclose(r.positive,torch.full((2,),.78,dtype=torch.float64))
    snap=r.snapshot();r.observe(q,.05);r.restore(snap)
    assert r.steps==21 and r.history.shape==(2,1344)
    for _ in range(50):r.observe(q,.1)
    assert r.history.shape==(2,4096)
    z=torch.zeros((1,2,1,3),requires_grad=True)
    qsmall=torch.tensor([[[[.9,.5,.1]],[[.9,.5,.1]]]])
    loss,pos,neg=hard_loss(z,qsmall,torch.tensor([.8,.8]));loss.backward()
    assert (z.grad[...,0]<0).all() and (z.grad[...,2]>0).all() and (z.grad[...,1]==0).all()
    z2=torch.zeros((1,2,1,1),requires_grad=True)
    zero,_,_=hard_loss(z2,torch.full_like(z2,.5),torch.tensor([.8,.8]));zero.backward()
    assert zero==0 and (z2.grad==0).all()
    assert not any('rdc' in k.lower() or 'engram' in k.lower() for k in configuration('DSR'))
    print('DSR_REGULATION_SELF_CHECK_PASS')
