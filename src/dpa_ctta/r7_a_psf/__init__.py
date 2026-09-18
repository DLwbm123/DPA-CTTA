"""A: source-proxy Gaussian filtering, not a full RP-GSSM reproduction."""
import math
import torch
from torch import nn
from ..r7_shared.host import Method
from ..r7_shared.numerics import (COUNTS,shape,finite,solve,symmetric,stable,mlp,attention,variance,temperature,temperature_initial,gaussian_nll)

def gaussian_filter(m,p,f,q,o,r):
    n=len(m)
    for x in (p,f,q,r):shape(x,(n,n))
    shape(o,(n,));finite(m)
    eye=torch.eye(n,dtype=torch.float64)
    # Validate the prior and process covariance, even when the predicted sum is SPD.
    torch.linalg.cholesky(p);torch.linalg.cholesky(q);torch.linalg.cholesky(r)
    pm=f@m;pp=symmetric(f@p@f.T+q)
    ip=solve(pp,eye);ir=solve(r,eye);precision=symmetric(ip+ir)
    pn=symmetric(solve(precision,eye));mn=pn@(ip@pm+ir@o)
    finite(mn,pn);return mn,pn

class PSF(Method):
    group='A';rank=16
    def __init__(self,basis,static=False):
        super().__init__(basis,static)
        self.style=nn.Parameter(torch.randn(8,32)*.1);self.content=nn.Parameter(torch.randn(32,64)*.1)
        self.head=mlp(128,64,32);self.W=nn.Parameter(torch.eye(16)*(.9/.95))
        u=(.01-1e-4)/(1-1e-4);self.qraw=nn.Parameter(torch.full((16,),math.log(u/(1-u))))
        self.cal_raw=nn.Parameter(temperature_initial(),requires_grad=False)
    def initial(self):return dict(m=torch.zeros(16,dtype=torch.float64),P=torch.eye(16,dtype=torch.float64),counter=0)
    def set_stage(self,stage):
        super().set_stage(stage);self.cal_raw.requires_grad_(stage=='cal')
    def observe(self,raw,tokens):
        shape(tokens,(64,64));d=self.observer(raw);ks=attention(d,self.style);dh=ks@self.style
        kc=attention(tokens,self.content);c=kc@self.content
        COUNTS['method_MLP']+=1;value=self.head(torch.cat((d,dh,c.mean(0))))
        r=variance(value[16:]).double()
        if self.stage!='fit':r=r*temperature(self.cal_raw).double().square()
        return dict(d=d,dhat=dh,k=kc,c=c,E=tokens,o=value[:16].double(),R=r)
    def update(self,raw,tokens,state,ablation=None):
        self.validate_state(state);old_counter=state['counter'];state=self.initial() if self.static else state
        a=self.observe(raw,tokens);r=a['R']
        if ablation=='A_ISO_OBS':r=r.mean().expand_as(r)
        elif ablation is not None:raise ValueError('A ablation')
        m,p=gaussian_filter(state['m'],state['P'],stable(self.W.double()),torch.diag(variance(self.qraw.double(),1.)),a['o'],torch.diag(r))
        return dict(m=m,P=p,counter=old_counter+1),a
    def fit_loss(self,a,clean,zstar,state):
        content=((a['c']-clean['E'].detach()).square().mean()+(a['k']-clean['k'].detach()).square().mean())/2
        return .1*(a['o']-zstar).square().mean()+.01*gaussian_nll(zstar,state['m'],state['P'])+.05*(a['dhat']-a['d'].detach()).square().mean()+.05*content
    def cal_loss(self,a,zstar,state):return gaussian_nll(zstar,state['m'],state['P'])

def build(basis,static=False,seed=20260918):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return PSF(basis,static)
