"""Native GraTa BN update, followed by the unchanged use/write FiLM controller."""
import copy,math
from pathlib import Path
import torch
from ..r10_use_write_rl.controller import Actor,detached
from ..r10_use_write_rl.math import Memory
from ..r10_use_write_rl.factory import construct as old_construct
from ..r10_12h_core.run import read,sha
from ..r7_shared.context import tensor_digest
from ..r7_shared.numerics import COUNTS,finite

SEED=20260924
NATIVE_SEED=20260907
ARMS=('RL_ORIGINAL','RL_GRATA','FIXED_GRATA','WARM_GRATA')


def fixed_actor():
    a=Actor(SEED)
    with torch.no_grad():
        for p in a.parameters():p.zero_()
        a.use[-1].bias[0]=5*math.atanh(math.log(4)/5)
    return a.eval().requires_grad_(False)


def native(c,lock):
    return old_construct(dict(arm='G_CTTA',seed=NATIVE_SEED,source_job=None,artifact=None),c,c['output_root'],lock)


def old(c,lock):
    ref=c['old_actor'];resolved=dict(arm='GR_RET_EMA',method='GR_RET_EMA',seed=SEED,source_job='POST',artifact=ref)
    return old_construct(resolved,c,c['original_source_root'],lock)


@torch.no_grad()
def copy_bn(src,dest):
    params=dict(dest.named_parameters())
    for name,p in zip(src.names,src.params):params[name].copy_(p.detach())


def bn_state(native):return {n:p.detach().cpu().clone() for n,p in zip(native.names,native.params)}


@torch.no_grad()
def set_bn(model,state):
    ps=dict(model.named_parameters())
    for n,x in state.items():ps[n].copy_(x.to(ps[n]))


class Host:
    def __init__(self,c,arm,lock,actor=None):
        self.old,self.old_close=old(c,lock);self.seg=self.old.segmenter;self.controller=self.old.controller
        self.actor=copy.deepcopy(actor if actor is not None else self.old.actor).eval().requires_grad_(False)
        self.g,self.gclose=native(c,lock) if arm!='RL_ORIGINAL' else (None,lambda:None)
        self.arm=arm;self.visits=0;self.failed=False;self.memory=Memory.zero()
        bn=set(self.g.native.names) if self.g else set()
        self.immutable=[p for name,p in self.seg.model.named_parameters() if name not in bn]+list(self.seg.model.buffers())+list(self.controller.carrier.parameters())+list(self.controller.carrier.buffers())+list(self.actor.parameters())+[self.seg.projection,self.controller.scale]
        self.stamp=tuple((id(p),p._version) for p in self.immutable);self.digest=tensor_digest([(str(i),p) for i,p in enumerate(self.immutable)])
        payload=dict(schema='R18_FULL_RL_HOST',code_sha=c['code_sha'],config_sha256=sha(c),lock_sha256=lock['sha256'],arm=arm,policy_seed=SEED,native_seed=NATIVE_SEED,actor_sha256=tensor_digest(list(self.actor.state_dict().items())),old_context=self.old.context['sha256'],native_context=self.g.context['sha256'] if self.g else None)
        self.context=dict(payload=payload,sha256=sha(payload))
    def check_frozen(self,boundary=False):
        if tuple((id(p),p._version) for p in self.immutable)!=self.stamp:raise ValueError('R18 actor/carrier/nonBN modified')
        if boundary and tensor_digest([(str(i),p) for i,p in enumerate(self.immutable)])!=self.digest:raise ValueError('R18 immutable content changed')
        if self.g:self.g.check_frozen(boundary)
    def step(self,image,force_zero=False):
        if self.failed:raise ValueError('failed host cannot advance')
        self.check_frozen();before=COUNTS.copy();gtrace=None
        try:
            if self.g:
                gprediction,gtrace=self.g.step(image);copy_bn(self.g.native,self.seg.model)
            with torch.no_grad():
                z,raw,tokens=self.seg(image,observe=True)
                if self.g and not torch.equal(z,gprediction):raise ValueError('post-GraTa mirrored zero-FiLM logits differ')
                observation=self.controller.observe(raw,tokens);use,nxt,a=self.controller.act(self.actor,observation,self.memory)
                logits=self.seg(image,self.controller.carrier.basis@(torch.zeros_like(use) if force_zero else use));finite(logits)
                self.memory=detached(nxt)
                if force_zero and not torch.equal(logits,z):raise ValueError('zero-FiLM final output mismatch')
            self.visits+=1;self.check_frozen();delta=dict(COUNTS-before)
            expected=11 if self.g else 2
            if delta.get('backbone_forwards')!=expected:raise ValueError('R18 backbone forward contract')
            return logits.detach(),dict(visit=self.visits,state_committed=True,counts=delta,native=gtrace,gain=float(a['raw_action'][0].sigmoid()),write=float(a['raw_action'][9].sigmoid()),action_residual_norm=float(a['raw_action'][1:9].tanh().norm()),use_norm=float(use.norm()),memory_norm=float(nxt.m.norm()),mass=float(nxt.h),force_zero=force_zero)
        except BaseException:self.failed=True;raise
    def snapshot(self):
        if self.failed:raise ValueError('failed host snapshot')
        self.check_frozen(True)
        return dict(schema='R18_FULL_RL_SNAPSHOT',context_sha256=self.context['sha256'],visits=self.visits,memory={k:getattr(self.memory,k).clone() for k in ('m','q','h')},native=self.g.snapshot() if self.g else None)
    def restore(self,s):
        if self.visits or s['schema']!='R18_FULL_RL_SNAPSHOT' or s['context_sha256']!=self.context['sha256']:raise ValueError('R18 restore identity')
        if self.g:self.g.restore(s['native']);copy_bn(self.g.native,self.seg.model)
        self.memory=Memory(**s['memory']);self.memory.check();self.visits=s['visits'];self.check_frozen(True)
    def close(self):self.old_close();self.gclose()


def load_actor(path):
    a=Actor(SEED);a.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['actor']);return a
