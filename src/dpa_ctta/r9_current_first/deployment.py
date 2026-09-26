"""Output-only FiLM residual scaling and same-weight frozen R9 diagnostics."""
import copy
import torch
from torch.nn import functional as F
from ..r8_ba.methods import R8Segmenter,film
from ..r8_ba.host import OnlineHost as LegacyHost
from ..r8_ba.context import capture
from ..r7_shared.numerics import COUNTS,finite,temperature_initial,temperature,variance
from .protocol import SPEC_SHA,digest


class Segmenter(R8Segmenter):
    def __init__(self,*args,alpha=1.,**kwargs):
        if alpha not in (0.,.25,.5,1.):raise ValueError('registered output-alpha')
        self.alpha=alpha;self.diagnostics={}
        super().__init__(*args,**kwargs)

    def _film(self,h,v,name):
        if h.shape[1]!=256:raise ValueError('FiLM channels')
        changed=film(h,v,self.amplitude)
        residual=changed-h
        self.diagnostics[name]=float(residual.detach().square().mean().sqrt())
        # Endpoints retain the original floating-point paths, not a cancellation approximation.
        if self.alpha==1:return changed
        if self.alpha==0:return h
        return h+self.alpha*residual

    def _up1(self,module,args,h):return self._film(h,self.v[:512],'up1_residual_rms')

    def _up3(self,module,args,h):
        if self.capture:
            norm=(h-h.mean((-2,-1),keepdim=True))/h.std((-2,-1),unbiased=False,keepdim=True).clamp_min(1e-6)
            e=F.adaptive_avg_pool2d(norm,(8,8))[0].flatten(1).T@self.projection
            self.cache['tokens']=F.layer_norm(e,(64,),eps=1e-6)
        return self._film(h,self.v[512:],'up3_residual_rms')


def prepare(method,diagnostic):
    """Mutate only a private copy before minting its R9 deployment identity."""
    method=copy.deepcopy(method)
    if diagnostic=='PRECAL':
        with torch.no_grad():method.cal_raw.copy_(temperature_initial())
        method.freeze()  # B recomputes frozen_eta with kappa=1.
    return method


class Host(LegacyHost):
    def __init__(self,segmenter,method,config,source,expected_context,diagnostic=None):
        if config.get('r9_spec_sha256')!=SPEC_SHA or config.get('output_alpha')!=segmenter.alpha:
            raise ValueError('R9 deployment identity')
        if config.get('diagnostic')!=diagnostic:raise ValueError('diagnostic binding')
        self.diagnostic=diagnostic;self.alpha=segmenter.alpha
        ablation={'RESET':'RESET_HISTORY','PRED_ONLY':'PRED_ONLY','ISTA20':'ISTA_20'}.get(diagnostic)
        if diagnostic not in (None,'RESET','PRECAL','PRED_ONLY','ISTA20','OUTSCALE'):raise ValueError('R9 diagnostic')
        if diagnostic=='PRECAL' and not torch.allclose(temperature(method.cal_raw),torch.tensor(1.)):
            raise ValueError('PRECAL temperature must be one')
        super().__init__(segmenter,method,config,source,expected_context,ablation)

    def check_frozen(self,boundary=False):
        if self.segmenter.alpha!=self.alpha:raise ValueError('R9 alpha changed')
        super().check_frozen(boundary)

    @torch.no_grad()
    def step(self,image):
        if self.failed:raise RuntimeError('R9 host stopped')
        before=COUNTS.copy()
        try:
            self.check_frozen()
            trace={}
            if self.method is None:logits=self.segmenter(image);state=None
            else:
                _,raw,tokens=self.segmenter(image,observe=True)
                state,a=self.method.update(raw,tokens,self.state,ablation=self.ablation)
                self.method.validate_state(state)
                v=self.method.ambient(state);logits=self.segmenter(image,v)
                trace.update(state_norm=float(self.method.code(state).norm()),v_norm=float(v.norm()),
                             tanh_saturation=float((v.abs()>3).float().mean()),**self.segmenter.diagnostics)
                if self.method.group=='A':
                    trace.update(traceP=float(state['P'].trace()),traceQ=float(variance(self.method.qraw.double(),1.).sum()),traceR=float(a['R'].sum()))
                elif hasattr(self.method,'cal_raw'):
                    trace.update(kappa=float(temperature(self.method.cal_raw)),prior_norm=float(a['prior'].norm()),
                                 correction_norm=float(a['delta'].norm()),sparse_fraction=float((a['delta']==0).double().mean()),
                                 energy=None if a['energy'] is None else a['energy'].tolist())
            finite(logits)
            if logits.shape!=(1,2,512,512):raise ValueError('R9 logits shape')
            self.state=state;self.visits+=1
            return logits.detach(),dict(visit=self.visits,state_committed=True,counts=dict(COUNTS-before),diagnostics=trace)
        except BaseException:self.failed=True;raise
