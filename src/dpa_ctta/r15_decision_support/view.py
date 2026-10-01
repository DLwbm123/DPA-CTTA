"""A parameter-free eligibility rule that preserves the quarter mixture hard decisions."""
import torch
from ..r12_horizontal_weight.view import WeightedHost,construct as original
NAME='DS_H025'
RULE='use quarter logits iff native and quarter float32 sigmoid>=0.5 decisions differ; otherwise exact native logits'

def supported(z0,zq):
    change=(z0.float().sigmoid()>=.5)!=(zq.float().sigmoid()>=.5)
    return torch.where(change,zq,z0),change

class SupportHost(WeightedHost):
    def payload(self):return dict(super().payload(),decision_support=RULE)
    @torch.no_grad()
    def step(self,image):
        if self.failed:raise RuntimeError('decision support host stopped')
        self.check_frozen()
        try:
            z0,_=self.base.step(image);zf,_=self.base.step(torch.flip(image,(-1,)));zf=torch.flip(zf,(-1,));zq=torch.logit((.75*z0.float().sigmoid()+.25*zf.float().sigmoid()).clamp(1e-6,1-1e-6));out,change=supported(z0,zq)
            if not torch.equal(out.sigmoid()>=.5,zq.sigmoid()>=.5):raise ValueError('hard-decision preservation failed')
            self.visits+=1
            return out,dict(visit=self.visits,state_committed=True,view_count=2,history_used=False,uncertain_fraction=float(change.float().mean()),quarter_hard_exact=True)
        except Exception:self.failed=True;raise

def construct(name,c,lock):
    if name!=NAME:return original(name,c,lock)
    h,close=original('C0',c,lock)
    return SupportHost(h.base,'CV_H025'),close
