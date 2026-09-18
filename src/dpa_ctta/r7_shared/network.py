"""Original frozen ResUNet, exact two-site FiLM, transient current-image observer."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .numerics import COUNTS, finite, shape, mlp
from .context import POLICY
from ..hosts.vptta import model_input_from_pixels


def film(h,v):
    shape(v,(512,))
    gamma,beta=v[:256].to(h),v[256:].to(h)
    return h + torch.expm1(.1*gamma.tanh())[None,:,None,None]*h + .1*beta.tanh()[None,:,None,None]


class Segmenter(nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model.cpu().eval().requires_grad_(False)
        self.inference_policy=copy.deepcopy(POLICY)
        self.v=None; self.capture=False; self.cache={}; self.forwards=0
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):
                m.track_running_stats=False; m.running_mean=None; m.running_var=None; m.num_batches_tracked=None
        # Original convenience hooks retain whole patient activations; remove only
        # this newly owned model's hooks. The backbone forward still uses local sfs.
        for hook in getattr(model,'feature_hooks',[]): hook.remove(); hook.features=None
        if hasattr(model,'feature_hooks'): model.feature_hooks=[]
        modules=dict(model.named_modules())
        if not {'up1','up3','res.conv1'}.issubset(modules): raise ValueError('original intervention sites missing')
        self.handles=[modules['up1'].register_forward_hook(self._up1),modules['up3'].register_forward_hook(self._up3),modules['res.conv1'].register_forward_hook(self._early)]
        with torch.random.fork_rng(devices=[]):
            gen=torch.Generator().manual_seed(20260918)
            q,r=torch.linalg.qr(torch.randn(256,64,generator=gen))
        q=q*torch.where(r.diagonal()<0,-1.,1.)
        self.register_buffer('projection',q)
    def _up1(self,m,args,h):
        if h.shape[1]!=256: raise ValueError('up1 channels')
        return film(h,self.v[:512])
    def _up3(self,m,args,h):
        if h.shape[1]!=256: raise ValueError('up3 channels')
        if self.capture:
            norm=(h-h.mean((-2,-1),keepdim=True))/h.std((-2,-1),unbiased=False,keepdim=True).clamp_min(1e-6)
            e=F.adaptive_avg_pool2d(norm,(8,8))[0].flatten(1).T@self.projection
            self.cache['tokens']=F.layer_norm(e,(64,),eps=1e-6)
        return film(h,self.v[512:])
    def _early(self,m,args,h):
        if self.capture:
            self.cache['early']=torch.cat((h.mean((-2,-1)),h.std((-2,-1),unbiased=False)),1)[0]
    def forward(self,pixels,v=None,observe=False):
        if self.v is not None: raise RuntimeError('reentrant backbone')
        x=model_input_from_pixels(pixels,'fundus')
        if x.shape[0]!=1: raise ValueError('batch1')
        if v is None: v=x.new_zeros(1024)
        shape(v,(1024,)); self.v=v; self.capture=observe
        try:
            self.forwards+=1; COUNTS['backbone_forwards']+=1
            result=self.model(x); z=result[0] if isinstance(result,(tuple,list)) else result
            shape(z,(1,2,512,512))
            if not observe: return z
            rgb=torch.cat((x.mean((-2,-1)),x.std((-2,-1),unbiased=False)),1)[0]
            raw=torch.cat((rgb,self.cache['early'])); tokens=self.cache['tokens']
            shape(raw,(134,)); shape(tokens,(64,64))
            return z,raw,tokens
        finally:
            self.v=None; self.capture=False; self.cache.clear()
    def close(self):
        for h in self.handles: h.remove()
        self.handles=[]; self.cache.clear(); self.v=None


class Observer(nn.Module):
    def __init__(self):
        super().__init__(); self.head=mlp(134,64,32)
        self.register_buffer('mean',torch.zeros(134)); self.register_buffer('std',torch.ones(134))
        self.register_buffer('fitted',torch.tensor(False))
    @torch.no_grad()
    def fit_scaler(self,raw,fold):
        if fold!='fit' or self.fitted: raise ValueError('fit-only one-time scaler')
        if raw.ndim!=2 or raw.shape[1]!=134 or len(raw)<2: raise ValueError('scaler shape')
        finite(raw); self.mean.copy_(raw.mean(0));self.std.copy_(raw.std(0,unbiased=False).clamp_min(1e-6));self.fitted.fill_(True)
    def forward(self,raw):
        if not self.fitted: raise ValueError('fit scaler required')
        shape(raw,(134,)); COUNTS['appearance_MLP']+=1
        return self.head((raw-self.mean)/self.std)
