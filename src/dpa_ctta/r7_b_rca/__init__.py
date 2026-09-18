"""B: finite five-step adapter correction; no sparse-recovery guarantee."""
import torch
from torch import nn
from ..r7_shared.host import Method
from ..r7_shared.numerics import COUNTS,shape,finite,stable,mlp,normalize,temperature,temperature_initial,coherence

def energy(delta,h,o,prior,kappa):
    return .5*(o-h@(prior+delta)).square().sum()/kappa**2+.01*delta.abs().sum()+.05*delta.square().sum()

def correct(h,o,prior,kappa=1.):
    h,o,prior=[x.double() for x in (h,o,prior)];finite(h,o,prior)
    shape(h,(len(o),len(prior)))
    kappa=torch.as_tensor(kappa,dtype=torch.float64);finite(kappa)
    if kappa<=0:raise ValueError('positive kappa')
    COUNTS['spectral_norms']+=1
    eta=1/(torch.linalg.matrix_norm(h,2).square()/kappa.square()+.1)
    delta=torch.zeros_like(prior);hist=[energy(delta,h,o,prior,kappa)]
    for _ in range(5):
        g=h.T@(h@(prior+delta)-o)/kappa.square()+.1*delta
        v=delta-eta*g;delta=v.sign()*(v.abs()-.01*eta).clamp_min(0)
        COUNTS['ISTA_iterations']+=1;hist.append(energy(delta,h,o,prior,kappa))
    finite(delta);return prior+delta,dict(delta=delta,eta=eta,energy=torch.stack(hist))

class RCA(Method):
    group='B';rank=32
    def __init__(self,basis,static=False):
        super().__init__(basis,static)
        if not torch.allclose(self.basis.T@self.basis,torch.eye(32,dtype=torch.float64),atol=1e-8,rtol=1e-8):raise ValueError('U orthonormal')
        self.W=nn.Parameter(.9*torch.eye(32));self.G=nn.Parameter(torch.randn(32,32)*.01)
        self.bias=mlp(32,64,32);self.head=mlp(32,64,64);self.Hraw=nn.Parameter(torch.randn(64,32))
        self.cal_raw=nn.Parameter(temperature_initial(),requires_grad=False)
        self.register_buffer('frozen_eta',(1/(torch.linalg.matrix_norm(normalize(self.Hraw.detach().double(),0),2).square()+.1)).clone())
    def initial(self):return dict(z=torch.zeros(32,dtype=torch.float64),d=torch.zeros(32,dtype=torch.float64),counter=0)
    def set_stage(self,stage):
        super().set_stage(stage);self.cal_raw.requires_grad_(stage=='cal')
        if stage=='online':
            with torch.no_grad():self.frozen_eta.copy_(1/(torch.linalg.matrix_norm(normalize(self.Hraw.double(),0),2).square()/temperature(self.cal_raw).double().square()+.1))
    def observe(self,raw,tokens):
        shape(tokens,(64,64));d=self.observer(raw);COUNTS['method_MLP']+=2
        return dict(d=d.double(),o=self.head(d).double(),bias=self.bias(d).double(),H=normalize(self.Hraw.double(),0))
    def update(self,raw,tokens,state,ablation=None):
        self.validate_state(state);old_counter=state['counter'];state=self.initial() if self.static else state
        a=self.observe(raw,tokens);d=a['d'];diff=torch.zeros_like(d) if state['counter']==0 else d-state['d']
        prior=stable(self.W.double())@state['z']+a['bias']+stable(self.G.double(),1.)@diff
        kappa=1. if self.stage=='fit' else temperature(self.cal_raw).double()
        if ablation=='B_PRED_ONLY':z=prior;a.update(delta=torch.zeros_like(z),eta=None,energy=None)
        elif ablation is None:
            z,diag=correct(a['H'],a['o'],prior,kappa);a.update(diag)
            if self.stage=='online' and not torch.allclose(a['eta'],self.frozen_eta,rtol=1e-10,atol=1e-12):raise ValueError('frozen eta mismatch')
        else:raise ValueError('B ablation')
        a.update(prior=prior,difference=diff)
        return dict(z=z,d=d,counter=old_counter+1),a
    def fit_loss(self,a,clean,zstar,state):
        return .1*(state['z']-zstar).square().mean()+.1*(a['o']-a['H']@zstar).square().mean()+.001*coherence(a['H'].T)
    def cal_loss(self,a,zstar,state):return (state['z']-zstar).square().mean()

def build(basis,static=False,seed=20260918):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return RCA(basis,static)
