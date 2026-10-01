"""Two prespecified current-image horizontal-flip controls; no learned gate."""
import torch
from ..r10_12h_core.run import sha
from ..r11_current_view.view import ViewHost,construct as original
VIEWS={'C0':((),),'CV_H2':((),(-1,)),'CV_H025':((),(-1,)),'H_ONLY':((-1,),)}
WEIGHTS={'C0':(1.,),'CV_H2':(.5,.5),'CV_H025':(.75,.25),'H_ONLY':(1.,)}
AGGREGATION='fixed inverse-aligned float32 sigmoid probability weights; clamp 1e-6 then logit for mixtures; preserve original inverse-aligned logits for one view'

class WeightedHost(ViewHost):
    def __init__(self,base,condition):
        if condition not in ('CV_H025','H_ONLY'):raise ValueError('registered new condition')
        super().__init__(base,'CV_H2' if condition=='CV_H025' else 'IDENTITY')
        self.condition=condition;self.views=VIEWS[condition];self.weights=WEIGHTS[condition]
        payload=self.payload();self.context=dict(payload=payload,sha256=sha(payload))
    def payload(self):return dict(base_context_sha256=self.base.context['sha256'],condition=self.condition,views=self.views,weights=self.weights,aggregation=AGGREGATION)
    def check_frozen(self,boundary=False):
        self.base.check_frozen(boundary)
        if self.views!=VIEWS[self.condition] or self.weights!=WEIGHTS[self.condition] or self.context['sha256']!=sha(self.payload()) or self.base.visits!=self.visits*len(self.views):raise ValueError('fixed weighting identity or offset changed')
    @torch.no_grad()
    def step(self,image):
        if self.failed:raise RuntimeError('weighted host stopped')
        self.check_frozen();ps=[]
        try:
            for dims in self.views:
                z,_=self.base.step(torch.flip(image,dims) if dims else image)
                if dims:z=torch.flip(z,dims)
                ps.append(z.float().sigmoid())
            out=z if len(ps)==1 else torch.logit((self.weights[0]*ps[0]+self.weights[1]*ps[1]).clamp(1e-6,1-1e-6))
            if not torch.isfinite(out).all():raise FloatingPointError('nonfinite weighted prediction')
            self.visits+=1
            return out,dict(visit=self.visits,state_committed=True,view_count=len(ps),flip_weight=self.weights[-1] if self.condition!='C0' else 0.,history_used=False,view_probability_disagreement=float(torch.stack(ps).std(0,unbiased=False).mean()))
        except Exception:self.failed=True;raise

def construct(name,c,lock):
    if name in ('C0','CV_H2'):return original('IDENTITY' if name=='C0' else name,c,lock)
    h,close=original('IDENTITY',c,lock)
    return WeightedHost(h.base,name),close
