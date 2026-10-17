"""Support-corrected crop pseudo-targets with a matched resampling control."""
import copy
import torch
from torch.nn import functional as F
from ..r37_cw_orientation.method import Host as RetainedHost, candidates
from ..r20_model_only_search.method import weights_for
from ..r19_model_only.method import equal

ARMS = ('NATIVE', 'RESAMPLE', 'CROP')


def configuration(arm):
    if arm not in ARMS: raise ValueError('unregistered arm')
    c = copy.deepcopy(candidates()[3])
    c.update(id='CW_LSO_'+arm, crop_arm=arm)
    return c


def matrices(x):
    return [x.new_tensor([[[.8,0,-.1],[0,.8,0]]]),
            x.new_tensor([[[-.8,0,.1],[0,.8,0]]])]


def inverse(a):
    h = F.pad(a,(0,0,0,1)); h[:,2,2] = 1
    return torch.linalg.inv(h)[:,:2]


def warp(x, a):
    grid = F.affine_grid(a, x.shape, align_corners=False)
    return F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)


def refine(q, back, support):
    # Six native weak views retain one vote each; inverse zero-padding is missing support.
    return ((6*q.detach()+sum(back))/(6+sum(support))).detach()


def delta_loss(z, q, refined, weight):
    return (weight*(F.binary_cross_entropy_with_logits(z,refined.detach(),reduction='none')-
                    F.binary_cross_entropy_with_logits(z,q.detach(),reduction='none'))).mean()


class Host(RetainedHost):
    @torch.no_grad()
    def prepare_crops(self, x):
        saved = self.snapshot(); back = []; supports = []
        try:
            for a in matrices(x):
                view = warp(x,a)
                p = self._readonly(self.native.model,view).sigmoid()
                inv = inverse(a.to(p))
                back.append(warp(p,inv))
                supports.append(warp(torch.ones_like(p[:,:1]),inv))
        finally:
            self.restore(saved)
        if not equal(self.snapshot(),saved): raise ValueError('crop previews mutated host')
        self.crop_back, self.crop_support = back, supports

    def _criterion(self, z, pseudo):
        base = super()._criterion(z,pseudo)
        if self.config['crop_arm']=='NATIVE': return base
        q = pseudo.detach()
        if self.config['crop_arm']=='RESAMPLE':
            back = [warp(warp(q,a),inverse(a)) for a in matrices(q)]
        else: back = self.crop_back
        target = refine(q,back,self.crop_support)
        if not torch.isfinite(target).all() or target.min() < -1e-6 or target.max() > 1+1e-6:
            raise ValueError('invalid refined Bernoulli target')
        target = target.clamp(0,1)
        p = self.modules['W']
        weight = weights_for(torch.stack(self.weak).sigmoid(),p['variance_temperature'],p['boundary_boost']).to(z)
        self.diag.update(target_change=float((target-q).abs().mean()),
                         target_flip_fraction=float(((target>=.5)!=(q>=.5)).float().mean()),
                         mean_crop_support=float(sum(self.crop_support).mean()),preview_forwards=2)
        return base+delta_loss(z,q,target,weight)

    def step(self,x):
        if self.config['crop_arm']=='NATIVE': return super().step(x)
        self.prepare_crops(x)
        try: return super().step(x)
        finally: self.crop_back=None; self.crop_support=None


def self_check():
    torch.set_num_threads(2)
    q = torch.empty(1,2,64,64); q[:,0]=.8; q[:,1]=.2
    support=[];back=[]
    for a in matrices(q):
        inv=inverse(a)
        h=F.pad(a,(0,0,0,1));h[:,2,2]=1
        hi=F.pad(inv,(0,0,0,1));hi[:,2,2]=1
        assert torch.allclose(h@hi,torch.eye(3)[None],atol=1e-6)
        s=warp(torch.ones_like(q[:,:1]),inv);support.append(s)
        back.append(warp(warp(q,a),inv))
    assert torch.allclose(refine(q,back,support),q,atol=2e-7)
    assert (sum(support)==0).any() and (sum(support)>1.9).any()
    yy,xx=torch.meshgrid(torch.linspace(-1+1/64,1-1/64,64),torch.linspace(-1+1/64,1-1/64,64),indexing='ij')
    ramp=torch.stack(((xx+1)/2,(yy+1)/2))[None]
    for a in matrices(ramp):
        got=warp(warp(ramp,a),inverse(a));s=warp(torch.ones_like(ramp[:,:1]),inverse(a))
        valid=(s>.9999).expand_as(got)
        assert (got-ramp).abs()[valid].max()<2e-6
    z=torch.randn_like(q,requires_grad=True);new=(q*.7).requires_grad_();w=torch.ones_like(q)*1.3
    old=(w*F.binary_cross_entropy_with_logits(z,q,reduction='none')).mean()
    got=old+delta_loss(z,q,new,w);want=(w*F.binary_cross_entropy_with_logits(z,new.detach(),reduction='none')).mean()
    assert torch.allclose(got,want,atol=1e-6)
    g=torch.autograd.grad(got,(z,new),allow_unused=True,retain_graph=True)
    assert g[1] is None and torch.allclose(g[0],torch.autograd.grad(want,z)[0],atol=1e-7)
    assert delta_loss(z,q,q,w)==0
    print('CROP_TARGET_SELF_CHECK_PASS')
