"""Thin image-only adapter around the frozen native and BN hosts."""
import copy
import torch
from .b1_host import Host as CHost,finite
from .b2_host import Host as IHost
from .m1_run import make_host,Observed
from .host_diagnostic import rng,restore,snapshot
from .offline.adaptation_dd import cpu_tree
from .source_pilot_release import source_unchanged


def direct(arm,state,device):
    if arm=='A':return make_host('fundus','A',state,device=device)
    if arm=='C':return CHost('C',state,device)
    if arm in ('U','S','I'):return IHost(arm,state,device)
    raise ValueError('B3 fixed arm')


class Entry:
    def __init__(self,arm,state,device='cpu'):
        self.arm=arm;self.host=direct(arm,state,device);self.watch=Observed(self.host) if arm=='A' else None;self.rng=rng();self.steps=0
    @property
    def optimizer(self):return self.host.optimizer if self.arm=='A' else self.host.base
    @property
    def counts(self):
        if self.arm!='A':return self.host.counts.copy()
        c=self.watch.counts
        return dict(forwards=c['model_forwards'],backwards=c['backward_calls'],base_adam=c['online_adam'],perturb=0,restore=0)
    def step(self,x):
        h=self.host
        if self.arm!='A':
            z,m=h.step(x);m['native_counts']=None;m['parent_counters']=None;m['parent_memory_size']=None
            m['interval_diagnostics']=m['diagnostics'] if self.arm!='C' else None
            return z,m
        restore(self.rng);before=self.watch.counts.copy();z=h.step(x);self.watch.verify();self.rng=rng();self.steps+=1
        finite(z);finite(h.optimizer.state)
        c={k:v-before[k] for k,v in self.watch.counts.items()}
        if (c['model_forwards'],c['backward_calls'],c['online_adam'],c['memory_pushes'])!=(2,1,1,1):raise ValueError('native physical counters')
        step=int(h.optimizer.state[h.prompt.data_prompt]['step']);counters=sorted({m.sample_num for m in h.model.modules() if isinstance(m,h.adabn)})
        if step!=self.steps or counters!=[step]:raise ValueError('native lifecycle')
        return z.detach(),dict(counts=dict(forwards=2,backwards=1,base_adam=1,perturb=0,restore=0),adam_step=step,lr=.05,native_counts=c,parent_counters=counters,parent_memory_size=len(h.memory_bank.memory),diagnostics=None,interval_diagnostics=None,fft_calls=dict(fft2=c['prompt_forwards'],ifft2=c['prompt_forwards'],basis='one each per observed pinned Prompt.forward'))
    def capture(self):return capture(self.host,self.arm)
    def finish(self,state):
        if self.arm=='A':self.watch.verify();source_unchanged(self.host,state);self.watch.release()
        else:self.host.finish(state)
    def trainable(self):
        h=self.host
        if self.arm=='A':return dict(optimizer_parameters=['prompt.data_prompt'],parameters=h.prompt.data_prompt.numel(),source_grad_behavior='native preserved',lr=.05,betas=[.9,.99])
        if len(h.names)!=82 or sum(p.numel() for p in h.params)!=19136:raise ValueError('BN registration')
        return dict(optimizer_parameters=h.names,parameters=19136,layers=41,lr=1e-4,betas=[.9,.999])


def capture(h,arm):
    if arm=='A':return cpu_tree(snapshot(h))
    return dict(affine=[p.detach().cpu().clone() for p in h.params],adam=cpu_tree(copy.deepcopy(h.base.state_dict())),step=h.steps)
