"""Frozen, image-only methods. Source training is in a separate module."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from ..r15_decision_support.view import supported
from .structure import hard,boundary_band,Tracker,largest_filled

STATIC=('C0','H025','DS','P','S','P_SIMPLE','P_VERIFY','D_LOGIT','D_CONTEXT','D_VERIFY')
CONTEXT_OFFSET=8
INPUT_CHANNELS=8+256+3+2

def resize(x,size,mode='bilinear'):
    return F.interpolate(x,size=size,mode=mode,**({} if mode=='nearest' else {'align_corners':False}))

def seeds(z0):
    p=z0.float().sigmoid().numpy()[0];h=hard(z0);r=math.ceil(.01*min(h.shape[-2:]))
    far=~np.any(boundary_band(h,r),axis=0)
    raw=np.stack(((p[0]<=.1)&(p[1]<=.1),(p[0]>=.9)&(p[1]<=.1),(p[0]>=.9)&(p[1]>=.9)))
    # Fixed 2-pixel square erosion before nearest-neighbor projection to up3.
    reliable=np.stack([ndi.binary_erosion(a&far,structure=np.ones((5,5),bool),border_value=0) for a in raw])
    return reliable

def prototypes(feature,reliable,shuffle=False):
    device=feature.device
    seed=resize(torch.from_numpy(reliable.astype(np.float32))[None],feature.shape[-2:],'nearest')[0].bool().to(device)
    f=F.normalize(feature[0].float(),dim=0,eps=1e-8)
    if shuffle:
        gen=torch.Generator().manual_seed(20261003)
        perm=torch.randperm(f.shape[-1]*f.shape[-2],generator=gen).to(device)
        f=f.flatten(1)[:,perm].reshape_as(f)
    counts=seed.sum((1,2));diag=dict(seed_counts=counts.tolist(),seed_coverage=[float(a.mean()) for a in reliable],fallback=None)
    if (counts<16).any():diag['fallback']='missing_seed';return None,diag
    mu=torch.stack([f[:,a].mean(1) for a in seed]);norm=mu.norm(dim=1)
    if not torch.isfinite(f).all() or not torch.isfinite(mu).all() or (norm<1e-8).any():diag['fallback']='nonfinite_or_zero_norm';return None,diag
    mu=F.normalize(mu,dim=1,eps=1e-8);q=torch.softmax(torch.einsum('kc,chw->khw',mu,f)/.1,dim=0)
    q=torch.stack((q[1]+q[2],q[2]))[None]
    diag['prototype_cosines']=(mu@mu.T).tolist()
    return resize(q,(512,512)).detach().cpu(),diag

def head_input(b,z0,zf,feature,image):
    prob=b.float().sigmoid();entropy=-(prob*torch.log(prob.clamp_min(1e-6))+(1-prob)*torch.log((1-prob).clamp_min(1e-6)))
    h=hard(b);distance=[]
    for a in h:
        distance.append(((ndi.distance_transform_edt(~a)-ndi.distance_transform_edt(a))/math.hypot(*a.shape)).astype('float32'))
    side=torch.cat((b,z0-zf,entropy,torch.from_numpy(np.stack(distance))[None]),1)
    rgb=resize(image.float(),(128,128));gray=rgb.mean(1,keepdim=True)
    dx=F.pad(gray[:,:,:,1:]-gray[:,:,:,:-1],(0,1,0,0));dy=F.pad(gray[:,:,1:,:]-gray[:,:,:-1,:],(0,0,0,1))
    # The same float16 rounding is used for cached source inputs and live targets.
    return torch.cat((resize(side,(128,128)),feature.detach().cpu(),rgb,dx,dy),1).half()

class ResidualHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(INPUT_CHANNELS,32,1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.ReLU(),nn.Conv2d(32,2,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        if sum(p.numel() for p in self.parameters())>100000:raise ValueError('head capacity')

    def forward(self,x):return self.net(x)

def normalize_input(x,statistics,kind):
    x=(x.float()-statistics['mean'].to(x.device))/statistics['std'].to(x.device)
    if kind=='D_LOGIT':x[:,CONTEXT_OFFSET:]=0
    elif kind!='D_CONTEXT':raise ValueError('unregistered head')
    return x

def residual(b,editable,r):
    return torch.where(torch.from_numpy(editable)[None].to(b.device),b+.25*torch.tanh(resize(r,b.shape[-2:])),b)

class Engine:
    def __init__(self,segmenter):
        self.seg=segmenter;self.feature=None;self.capture=False
        self.hook=dict(segmenter.model.named_modules())['up3'].register_forward_hook(self._capture)

    def _capture(self,module,args,out):
        if self.capture:self.feature=out.detach()
        return out

    @torch.no_grad()
    def prepare(self,image,shuffle=False):
        try:
            self.capture=True;z0=self.seg(image).detach().float();feature=self.feature
            if feature is None or feature.shape!=(1,256,128,128):raise ValueError('up3 feature alignment')
            self.capture=False;zf=torch.flip(self.seg(torch.flip(image,(-1,))),(-1,)).detach().float()
            qz=torch.logit((.75*z0.sigmoid()+.25*zf.sigmoid()).clamp(1e-6,1-1e-6));b,support=supported(z0,qz)
            if not np.array_equal(hard(b),hard(qz)):raise ValueError('DS hard identity')
            reliable=seeds(z0);protected=np.broadcast_to(np.any(reliable,axis=0),(2,512,512)).copy()
            editable=boundary_band(hard(b),6)&~protected
            q,diag=prototypes(feature,reliable,shuffle)
            p=b if q is None else torch.where(torch.from_numpy(editable)[None],b+.25*(torch.logit(q.clamp(1e-6,1-1e-6))-b).clamp(-1,1),b)
            return dict(C0=z0,H025=qz,DS=b,P=p,editable=editable,reliable=protected,q=q,
                        inputs=head_input(b,z0,zf,feature,image),diagnostics=diag)
        finally:self.capture=False;self.feature=None

    @torch.no_grad()
    def outputs(self,image,heads=None,statistics=None,shuffle=False):
        x=self.prepare(image,shuffle);out={k:x[k] for k in ('C0','H025','DS','P')}
        tracker=Tracker(x['DS'],x['editable'],x['reliable'],x['q']);pc=tracker.candidates(x['P'])
        out['P_SIMPLE'],a=tracker.select(pc,'SIMPLE',largest_filled(hard(x['P'])))
        out['P_VERIFY'],b=tracker.select(pc);out['S'],s=tracker.structural()
        diagnostics=dict(prototype=x['diagnostics'],P_candidates=[v.record() for v in pc],P_SIMPLE=a,P_VERIFY=b,S=s)
        if heads:
            for kind,head in heads.items():
                device=next(head.parameters()).device
                r=head(normalize_input(x['inputs'].to(device),statistics,kind)).cpu()
                out[kind]=residual(x['DS'],x['editable'],r)
            dc=tracker.candidates(out['D_CONTEXT']);out['D_VERIFY'],d=tracker.select(dc);diagnostics.update(D_candidates=[v.record() for v in dc],D_VERIFY=d)
        for key,z in out.items():
            if not torch.isfinite(z).all():raise ValueError('nonfinite static output '+key)
            if key not in ('C0','H025','DS') and not torch.equal(z[torch.from_numpy(~x['editable'])[None]],x['DS'][torch.from_numpy(~x['editable'])[None]]):raise ValueError('protected logits changed')
        return out,diagnostics,x

    def close(self):self.hook.remove();self.feature=None
