"""Protect native confident logits; fixed quarter mixing only at native uncertainty."""
import torch
from ..r10_12h_core.run import sha
from ..r12_horizontal_weight.view import WeightedHost,construct as original
NAME='U10_H025'
LOW=.4
HIGH=.6
WEIGHTS=(.75,.25)

def gated(z0,zflip):
    if (LOW,HIGH,WEIGHTS)!=(.4,.6,(.75,.25)):raise ValueError('frozen uncertainty interval/weight changed')
    p=z0.float().sigmoid();uncertain=(p>=LOW)&(p<=HIGH)
    mixed=torch.logit((WEIGHTS[0]*p+WEIGHTS[1]*zflip.float().sigmoid()).clamp(1e-6,1-1e-6))
    return torch.where(uncertain,mixed,z0),uncertain

class UncertaintyHost(WeightedHost):
    def payload(self):return dict(super().payload(),gate_interval=[LOW,HIGH],confident_logits='preserve exact native logits')
    @torch.no_grad()
    def step(self,image):
        if self.failed:raise RuntimeError('gate host stopped')
        self.check_frozen()
        try:
            z0,_=self.base.step(image);zf,_=self.base.step(torch.flip(image,(-1,)));zf=torch.flip(zf,(-1,));out,mask=gated(z0,zf)
            if not torch.isfinite(out).all():raise FloatingPointError('nonfinite gated prediction')
            self.visits+=1
            return out,dict(visit=self.visits,state_committed=True,view_count=2,history_used=False,flip_weight=.25,uncertain_fraction=float(mask.float().mean()),protected_fraction=float((~mask).float().mean()))
        except Exception:self.failed=True;raise

def construct(name,c,lock):
    if name!=NAME:return original(name,c,lock)
    h,close=original('C0',c,lock)
    return UncertaintyHost(h.base,'CV_H025'),close
