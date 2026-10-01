"""Fixed probability ensembles of inverse-aligned current-image flips."""
import torch
from ..r10_12h_core.run import sha
from ..r10_use_write_rl.factory import construct as native
VIEWS={'IDENTITY':((),),'CV_H2':((),(-1,)),'CV_FLIP4':((),(-1,),(-2,),(-2,-1))}
AGGREGATION='mean float32 sigmoid probabilities after inverse flips; clamp 1e-6; logit; unchanged one-view logits'

class ViewHost:
    def __init__(self,base,name):
        if name not in VIEWS or base.method is not None:raise ValueError('registered frozen C0 views required')
        self.base=base;self.segmenter=base.segmenter;self.name=name;self.views=VIEWS[name];self.visits=0;self.failed=False
        payload=dict(base_context_sha256=base.context['sha256'],condition=name,views=self.views,aggregation=AGGREGATION)
        self.context=dict(payload=payload,sha256=sha(payload))
    def check_frozen(self,boundary=False):
        self.base.check_frozen(boundary)
        payload=dict(base_context_sha256=self.base.context['sha256'],condition=self.name,views=self.views,aggregation=AGGREGATION)
        if self.views!=VIEWS[self.name] or self.context['sha256']!=sha(payload) or self.base.visits!=self.visits*len(self.views):raise ValueError('frozen view identity or offset changed')
    @torch.no_grad()
    def step(self,image):
        if self.failed:raise RuntimeError('view host stopped')
        self.check_frozen();ps=[]
        try:
            for dims in self.views:
                z,_=self.base.step(torch.flip(image,dims) if dims else image)
                if dims:z=torch.flip(z,dims)
                ps.append(z.float().sigmoid())
            out=z if len(ps)==1 else torch.logit(torch.stack(ps).mean(0).clamp(1e-6,1-1e-6))
            if not torch.isfinite(out).all():raise FloatingPointError('nonfinite view prediction')
            self.visits+=1
            return out,dict(visit=self.visits,state_committed=True,view_count=len(ps),history_used=False,view_probability_disagreement=float(torch.stack(ps).std(0,unbiased=False).mean()))
        except Exception:self.failed=True;raise
    def snapshot(self):
        if self.failed:raise ValueError('failed snapshot')
        self.check_frozen(True)
        return dict(schema='R11_CURRENT_VIEW_SNAPSHOT_V1',context_sha256=self.context['sha256'],visits=self.visits,base=self.base.snapshot())
    def restore(self,s):
        if s.get('schema')!='R11_CURRENT_VIEW_SNAPSHOT_V1' or s.get('context_sha256')!=self.context['sha256'] or type(s.get('visits')) is not int or s['visits']<0 or s['base']['visits']!=s['visits']*len(self.views):raise ValueError('view snapshot identity')
        self.base.restore(s['base']);self.visits=s['visits'];self.failed=False;self.check_frozen(True)

def construct(name,c,lock):
    if name not in VIEWS:raise ValueError('unknown condition')
    r=dict(arm='C0',method='C0',seed=20260924,source_job=None,artifact=None,diagnostic=None)
    h,close=native(r,c,c['output_root'],lock)
    return ViewHost(h,name),close
