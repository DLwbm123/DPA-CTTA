"""Content-keyed augmentation and reversible current-image interventions."""
import copy,random
from contextlib import contextmanager
import numpy as np
import torch
from ..r78_aligned_barycenter.method import Host as Previous,ARMS
from ..r37_cw_orientation.method import Host as LossHost
from ..r20_model_only_search.method import weights_for,boundary
from ..r8_ba.rng import capture,restore
from ..b1_host import official,finite
from ..hosts.vptta import model_input_from_pixels

SEEDS=(20260907,17011,29009)
NEXT={s:SEEDS[(i+1)%3] for i,s in enumerate(SEEDS)}


def style(x,seed,index):
    """Only the original strong sampler; stable private content ordinal, no new replicate seed."""
    saved=capture()
    try:
        key=int(np.random.SeedSequence([seed,index,79]).generate_state(1)[0])
        random.seed(key);np.random.seed(key);torch.manual_seed(key)
        data={'data':model_input_from_pixels(x,'fundus').numpy().copy()}
        out=official().augmentation_strong_style(data)
        if out.shape!=(1,3,512,512) or not np.isfinite(out).all():raise ValueError('strong view contract')
        return out.copy()
    finally:restore(saved)


@contextmanager
def supplied_style(raw):
    module=official();original=module.augmentation_strong_style
    def provide(data):
        if data['data'].shape!=raw.shape:raise ValueError('style shape mismatch')
        return raw.copy()
    module.augmentation_strong_style=provide
    try:yield
    finally:module.augmentation_strong_style=original


class Host(Previous):
    def __init__(self,*args,charge=None,**kwargs):
        self.charge=charge;self.target_cache=None;self.gradient_cache={}
        super().__init__(*args,**kwargs)

    def snapshot(self):
        n=self.native
        return dict(base=super().snapshot(),diagnostics=copy.deepcopy(self.diag),weak=self.weak,collect=self.collect,
            strong=self.strong,extra=self.extra,target_cache=self.target_cache,gradient_cache=self.gradient_cache,
            features=[getattr(h,'features',None) for h in getattr(n.model,'feature_hooks',[])])

    def restore(self,s):
        super().restore(s['base']);self.diag=copy.deepcopy(s['diagnostics']);self.weak=s['weak'];self.collect=s['collect']
        self.strong=s['strong'];self.extra=s['extra'];self.target_cache=s['target_cache'];self.gradient_cache=s['gradient_cache']
        for h,v in zip(getattr(self.native.model,'feature_hooks',[]),s['features']):h.features=v

    @torch.no_grad()
    def before(self,x):
        saved=self.snapshot()
        try:
            self.collect=False
            z=self.native.model(model_input_from_pixels(x,'fundus').to(self.device))[0].detach().float().cpu()
            finite(z);return z
        finally:self.restore(saved)

    def _tag(self,z,name,scale):
        def hook(g):
            finite(g);raw=g.detach()/scale
            self.gradient_cache[name]=raw
            self.diag.setdefault('gradient_sites',{})[name]=dict(observed_loss_scale=scale,
                unscaled_channel_norm=raw.square().sum((0,2,3)).sqrt().tolist(),
                unscaled_channel_mean=raw.mean((0,2,3)).tolist())
            return g
        z.register_hook(hook)

    def _criterion(self,z,q):
        predictions=torch.stack(self.weak).sigmoid()
        w=weights_for(predictions,.1,2.).to(q)
        self.target_cache=(q.detach().cpu().clone(),w.detach().cpu().clone())
        def loss_at(value):
            loss=LossHost._criterion(self,value,q)
            return loss,dict(BCE=self.diag['pseudo_loss'],LSO=self.diag['LSO'])
        if self.arm=='NATIVE':
            self._tag(z,'original',1.);loss,parts=loss_at(z);self.diag['loss_parts']=parts
        else:
            if self.strong is None:raise ValueError('missing original strong tensor')
            self.extra=True
            try:other=torch.flip(self.native.model(torch.flip(self.strong,(-1,)))[0],(-1,))
            finally:self.extra=False;self.strong=None
            if self.arm=='VERTEX':
                self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
                first,p=loss_at(z);second,r=loss_at(other);loss=(first+second)*.5
                self.diag['loss_parts']={k:(p[k]+r[k])*.5 for k in p}
                self.diag['loss_branch0']=p;self.diag['loss_branch1']=r
            else:
                averaged=(z+other)*.5;self._tag(averaged,'barycenter',1.)
                loss,parts=loss_at(averaged);self.diag['loss_parts']=parts
        self.diag['joint_loss']=float(loss.detach());return loss

    def keyed_step(self,x,raw):
        if self.charge:self.charge('attempt')
        self.target_cache=None;self.gradient_cache={}
        with supplied_style(raw):z,detail=super().step(x)
        if self.arm=='VERTEX':
            a=self.gradient_cache['vertex0'];b=self.gradient_cache['vertex1']
            cosine=(a*b).sum((0,2,3))/(a.square().sum((0,2,3)).sqrt()*b.square().sum((0,2,3)).sqrt()).clamp_min(1e-20)
            self.diag['logit_branch_gradient_cosine']=cosine.tolist()
        self.diag.pop('logit_gradient_regions',None)
        self.gradient_cache={}
        detail['diagnostics']=copy.deepcopy(self.diag)
        if self.charge:self.charge('complete')
        return z,detail

    def cycle(self,x,current,alternative):
        """Only the final K update commits; diagnostic counters live outside snapshots."""
        initial=self.snapshot();pre=self.before(x)
        alternate,ud=self.keyed_step(x,alternative);uq,uw=self.target_cache
        self.restore(initial)
        main,kd=self.keyed_step(x,current);kq,kw=self.target_cache
        if not torch.equal(uq,kq) or not torch.equal(uw,kw):raise ValueError('U/K pseudo target or W drift')
        return (pre,alternate,main),dict(U=ud['diagnostics'],K=kd['diagnostics'],q_W_exact=True)


def self_check():
    from ..r19_model_only.method import equal
    # A synthetic image checks the sampler and RNG isolation without any model calls.
    x=torch.linspace(0,1,3*512*512).reshape(1,3,512,512);saved=capture()
    a=style(x,SEEDS[0],17);b=style(x,SEEDS[0],17)
    assert np.array_equal(a,b) and equal(capture(),saved)
    assert NEXT[NEXT[NEXT[SEEDS[0]]]]==SEEDS[0]
    print('KEYED_STYLE_AND_RNG_SELF_CHECK_PASS',flush=True)
