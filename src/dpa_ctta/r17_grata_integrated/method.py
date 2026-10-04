"""Inject a stopped local residual target into the official consistency backward.

No additional backbone forward, target label, optimizer or output postprocessing.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from ..r16_evidence_correction.methods import ResidualHead,head_input,normalize_input,residual,seeds
from ..r16_evidence_correction.structure import boundary_band,hard
from ..r15_decision_support.view import supported
from ..r10_use_write_rl.factory import construct as native_construct
from ..r10_12h_core.run import sha

SEED=20261004
TARGET_SEED=20260907
ARMS=('C_CONTEXT','G_CONTEXT','G_LOGIT')
LAMBDA=.1


def payload(z0,zf,feature,image):
    quarter=torch.logit((.75*z0.sigmoid()+.25*zf.sigmoid()).clamp(1e-6,1-1e-6))
    baseline,_=supported(z0,quarter)
    reliable=seeds(z0).any(0)
    editable=boundary_band(hard(baseline),6)&~reliable[None]
    return dict(inputs=head_input(baseline,z0,zf,feature,image)[0],baseline=baseline[0],editable=torch.from_numpy(editable))


def correction_loss(student,q0,qd,mask):
    # Subtract the same loss to DS: zero residual is exactly the unmodified objective.
    if not mask.any():return student.sum()*0
    delta=F.binary_cross_entropy_with_logits(student,qd,reduction='none')-F.binary_cross_entropy_with_logits(student,q0,reduction='none')
    return delta[mask].mean()


class Corrector:
    def __init__(self,native,kind=None,head=None,statistics=None,weight=LAMBDA,observe=None):
        self.native=native;self.kind=kind;self.head=head;self.statistics=statistics;self.weight=weight;self.observe=observe
        self.active=False;self.count=0;self.cache={};self.last={};self.original=native.opt.cal_consis_loss
        self.handles=[dict(native.model.named_modules())['up3'].register_forward_hook(self.feature),native.model.register_forward_hook(self.capture)]
        native.opt.cal_consis_loss=self.consistency
    def feature(self,m,args,value):
        if self.active and self.count==0:self.cache['feature']=value.detach().float().cpu()
    def capture(self,m,args,out):
        if not self.active or torch.is_grad_enabled():return
        if self.count==0:self.cache.update(z0=out[0].detach().float().cpu(),image=args[0].detach().float().cpu())
        elif self.count==1:self.cache['zf']=torch.flip(out[0].detach().float().cpu(),(-1,))
        self.count+=1
    def consistency(self,data):
        self.active=True;self.count=0;self.cache={};self.last={}
        try:return self.original(data,criterion=self.criterion)
        finally:self.active=False;self.cache.clear()
    def criterion(self,predictions,teacher):
        if self.count!=6:raise ValueError('exact official six weak views required')
        base=F.binary_cross_entropy_with_logits(predictions,teacher)
        a=payload(**self.cache)
        if self.observe:self.observe(a)
        self.last=dict(local_weight=self.weight,eligible_pixels=int(a['editable'].sum()),correction_loss=0.,mean_probability_change=0.,head_forwards=0)
        if self.head is None or self.weight==0:return base
        with torch.no_grad():
            device=next(self.head.parameters()).device
            h=self.head(normalize_input(a['inputs'][None].to(device),self.statistics,self.kind)).cpu()
            b=a['baseline'][None];z=residual(b,a['editable'].numpy(),h)
            if not torch.equal(z[~a['editable'][None]],b[~a['editable'][None]]):raise ValueError('protected logits changed')
            q0=b.sigmoid().to(predictions);qd=z.sigmoid().to(predictions);mask=a['editable'][None].to(predictions.device)
        local=correction_loss(predictions,q0,qd,mask)
        self.last.update(correction_loss=float(local.detach()),mean_probability_change=float((qd-q0).abs().mean()),head_forwards=1)
        return base+self.weight*local
    def close(self):
        self.native.opt.cal_consis_loss=self.original
        for h in self.handles:h.remove()
        self.cache.clear()


def construct(c,kind,seed,lock,head=None,statistics=None,weight=LAMBDA,observe=None):
    h,close=native_construct(dict(arm='G_CTTA' if kind.startswith('G') else 'C_CTTA',seed=seed,source_job=None,artifact=None),c,c['output_root'],lock)
    correction=Corrector(h.native,'D_LOGIT' if kind.endswith('LOGIT') else 'D_CONTEXT',head,statistics,weight,observe)
    original_step=h.step
    def step(image):
        out,trace=original_step(image);trace.update(correction=correction.last.copy())
        if correction.cache:raise ValueError('current-image feature cache survived')
        return out,trace
    h.step=step
    def done():correction.close();close()
    return h,correction,done


def load_heads(root,selection):
    heads={}
    with torch.random.fork_rng():
        for kind,r in selection.items():
            h=ResidualHead().cuda();h.load_state_dict(torch.load(r['file'],map_location='cpu',weights_only=True));h.eval().requires_grad_(False);heads[kind]=h
    stats=torch.load(Path(root)/'private/head-statistics.pt',weights_only=True)
    return heads,stats
