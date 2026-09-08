"""Fixed FDA-style partial amplitude mixing for M3; native M1/M2 are unchanged."""
import math
import torch
from .hosts.vptta import VPTTAHost
from .offline.adaptation_dd import OfflineEpisode,preprocess,native_adam,cpu_tree
from .offline.prompt_gradient_matching import cosine_objective
from .medical_losses import medical_loss
from .source_pilot import seed_all


def low_frequency_mask(h,w,beta,device):
    radius=int(beta*min(h,w));ky=torch.arange(h,device=device);kx=torch.arange(w,device=device)
    return ((torch.minimum(ky,h-ky)<=radius)[:,None] & (torch.minimum(kx,w-kx)<=radius)[None,:])[None,None]


def condition_proxy(S,donor,strength=.5,band_fraction=.01,eps=1e-8,*,statistics=False):
    if S.ndim!=4 or donor.ndim!=4:raise ValueError('BCHW required')
    if S.shape[1]!=3 or donor.shape[0]!=1 or donor.shape[1:]!=S.shape[1:]:raise ValueError('one RGB donor with matching final grid required')
    if S.device!=donor.device or S.dtype!=donor.dtype or not S.is_floating_point():raise ValueError('matching floating dtype/device required')
    if not (0<=strength<=1 and 0<=band_fraction<.5 and math.isfinite(eps) and eps>0):raise ValueError('invalid transform parameters')
    if not torch.isfinite(S).all() or not torch.isfinite(donor).all() or ((S<0)|(S>1)).any() or ((donor<0)|(donor>1)).any():raise ValueError('finite RGB [0,1] required')
    if strength==0:return (S,dict(clipping_fraction=0.,change_l2=0.)) if statistics else S
    mask=low_frequency_mask(*S.shape[-2:],band_fraction,S.device)
    fs=torch.fft.fft2(S,norm='backward');fd=torch.fft.fft2(donor.detach(),norm='backward');a=fs.abs()
    unit=torch.where(a>eps,fs/a.clamp_min(eps),torch.ones_like(fs))
    mixed=fs+strength*mask*(fd.abs()-a)*unit
    raw=torch.fft.ifft2(mixed,norm='backward').real
    result=raw.clamp(0,1)
    if statistics:return result,dict(clipping_fraction=float(((raw<0)|(raw>1)).float().mean().detach()),change_l2=float((result-S).norm().detach()))
    return result


class ConditionedHost(VPTTAHost):
    def __init__(self,*args,strength=.5,**kwargs):
        if not 0<=strength<=1:raise ValueError('strength outside [0,1]')
        self.strength=strength;self.condition_calls=0;self.condition_stats=dict(clipping_fraction=0.,change_l2=0.)
        super().__init__(*args,**kwargs)

    def step(self,pixel_rgb):
        if self.extra_weight and self.strength:
            # Preserve the fixed proxy body; the previous conditioned output is never a donor/base.
            if pixel_rgb.device.type!='cpu' or pixel_rgb.dtype!=torch.float32 or pixel_rgb.requires_grad:raise ValueError('fixed CPU float32 current pixels required')
            with torch.no_grad():
                P,self.condition_stats=condition_proxy(self.proxy.pixel_rgb,pixel_rgb.to(self.device),self.strength,statistics=True)
                self._proxy_input=preprocess(P,self.task)
            self.condition_calls+=1
        return super().step(pixel_rgb)


def make_host(task,arm,state,proxy,device='cuda:0',strength=.5):
    if arm not in ('R3','D3','O3','O2T'):raise ValueError('M3 new arms only')
    seed_all(20260907)
    return ConditionedHost(task,source_state=state,device=device,mode='proxy_rehearsal',extra_weight=.1,beta_boundary=0,proxy_factory=lambda:proxy,strength=strength)


class ConditionedEpisode(OfflineEpisode):
    """Same D/O operations, with T before proxy preprocessing and optional smoke capture."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.counts['condition_transforms']=0;self.inner_result=None

    def objective(self,method,S,masks,pixels,query_mask,state,*,capture_inner=False):
        if method not in {'D','O'} or not S.requires_grad:raise ValueError('trainable D/O pixels required')
        P,self.condition_stats=condition_proxy(S,pixels.to(self.device),statistics=True);self.counts['condition_transforms']+=1
        x,phi,opt=self.initialize(pixels,state)
        masks=masks.to(self.device);query_mask=query_mask.to(self.device)
        logits=self.forward(self.host.model,x,phi)
        if method=='D':reference=self.gradient(medical_loss(logits,query_mask,beta_boundary=0).region,phi).detach()
        else:
            scope=self.host.model.resnet if self.task=='polyp' else self.host.model
            losses=[m.bn_loss for m in scope.modules() if isinstance(m,self.host.adabn)]
            if not losses:raise ValueError('empty native loss coverage')
            reference=self.gradient(sum(losses)/len(losses),phi).detach()
        del logits
        syn=self.forward(self.clone,preprocess(P,self.task),phi,True)
        g=self.gradient(medical_loss(syn,masks,beta_boundary=0).region,phi,True)
        if method=='D':loss=cosine_objective(g,reference)
        else:
            plus,adam_plus=native_adam(phi,reference+.1*g,opt,self.host.optimizer.param_groups[0]['lr'])
            self.counts['differentiable_inner']+=1
            if capture_inner:self.inner_result=cpu_tree(dict(prompt=plus,adam=adam_plus,gradient=reference+.1*g,initial_prompt=phi))
            after=self.forward(self.host.model,x,plus)
            loss=medical_loss(after,query_mask,beta_boundary=0).region
        if not torch.isfinite(loss):raise ValueError('nonfinite outer objective')
        return loss
