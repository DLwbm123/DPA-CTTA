"""C: conditional heteroscedastic evidence, not front-door causal identification."""
import torch
from torch import nn
from ..r7_shared.host import Method
from ..r7_shared.numerics import COUNTS,shape,finite,solve,mlp,attention,normalize,variance,coherence

def energy(z,h,o,r,prior):
    e=(o-h@z)/r.sqrt();a=e.abs()
    return torch.where(a<=1,.5*e.square(),a-.5).mean()+.05*(z-prior).square().sum()

def correct(h,o,r,prior):
    h,o,r,prior=[x.double() for x in (h,o,r,prior)];finite(h,o,r,prior)
    shape(h,(len(o),len(prior)));shape(r,o.shape)
    if len(o)==0 or (r<=0).any():raise ValueError('positive evidence variance')
    z=prior;hist=[energy(z,h,o,r,prior)];weights=[]
    for _ in range(3):
        e=(o-h@z)/r.sqrt();nu=1/e.abs().clamp_min(1);w=nu/r
        lhs=(h.T*w)@h/len(o)+.1*torch.eye(len(prior),dtype=torch.float64)
        rhs=h.T@(w*o)/len(o)+.1*prior
        z=solve(lhs,rhs);COUNTS['IRLS_iterations']+=1
        hist.append(energy(z,h,o,r,prior));weights.append(w)
    return z,dict(energy=torch.stack(hist),weights=weights)

class RBE(Method):
    group='C';rank=32
    def __init__(self,basis,static=False):
        super().__init__(basis,static)
        if not torch.allclose(self.basis.T@self.basis,torch.eye(32,dtype=torch.float64),atol=1e-8,rtol=1e-8):raise ValueError('U orthonormal')
        self.Pc=nn.Parameter(torch.randn(32,64)*.1);self.Pa=nn.Parameter(torch.randn(16,64)*.1);self.O=nn.Parameter(torch.randn(16,8)*.1)
        self.query=mlp(96,64,16);self.mapping=mlp(64,64,256);self.reliability=mlp(96,64,8)
        self.reliability.requires_grad_(False)
        self.register_buffer('constant_R',torch.ones(8));self.register_buffer('constant_ready',torch.tensor(False))
    def initial(self):return dict(z=torch.zeros(32,dtype=torch.float64),counter=0)
    def set_stage(self,stage):
        super().set_stage(stage);self.reliability.requires_grad_(stage=='cal')
    def observe(self,raw,tokens):
        shape(tokens,(64,64));d=self.observer(raw);kc=attention(tokens,self.Pc);c=kc@self.Pc
        x=torch.cat((tokens-c,d.expand(64,-1)),1)
        ka=self.query(x).softmax(-1);o=ka@self.O;h=normalize(self.mapping(c).reshape(64,8,32))
        r=torch.ones_like(o) if self.stage=='fit' else variance(self.reliability(x))
        COUNTS['method_MLP']+=2+(self.stage!='fit')
        return dict(kc=kc,c=c,ka=ka,o=o.double(),H=h.double(),R=r.double(),E=tokens,reconstruction=c+ka@self.Pa)
    def update(self,raw,tokens,state,ablation=None):
        self.validate_state(state);old_counter=state['counter'];state=self.initial() if self.static else state
        a=self.observe(raw,tokens);r=a['R']
        if ablation=='C_CONST_R':
            if not self.constant_ready:raise ValueError('source-cal constant R missing')
            r=self.constant_R.double().expand_as(r)
        elif ablation is not None:raise ValueError('C ablation')
        z,diag=correct(a['H'].reshape(512,32),a['o'].flatten(),r.flatten(),.9*state['z']);a.update(diag)
        return dict(z=z,counter=old_counter+1),a
    def fit_loss(self,a,clean,zstar,state):
        residual=a['o']-a['H']@zstar
        coh=(coherence(self.Pc)+coherence(self.Pa))/2
        return .1*(state['z']-zstar).square().mean()+.1*residual.square().mean()+.05*(a['reconstruction']-a['E']).square().mean()+.05*(a['kc']-clean['kc'].detach()).square().mean()+.001*coh
    def cal_loss(self,a,zstar,state):return .5*(a['R'].log()+(a['o']-a['H']@zstar).square()/a['R']).mean()
    @torch.no_grad()
    def set_constant_variance(self,r,fold):
        if fold!='cal' or self.stage!='online' or self.constant_ready:raise ValueError('one source-cal constant R only')
        shape(r,(8,))
        if (r<1e-4).any() or (r>10).any():raise ValueError('constant variance range')
        self.constant_R.copy_(r);self.constant_ready.fill_(True)

def build(basis,static=False,seed=20260918):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return RBE(basis,static)
