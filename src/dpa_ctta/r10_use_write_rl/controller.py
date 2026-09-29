"""Frozen real B read, separate current use and atomic persistent write."""
import copy
import math
import torch
from torch import nn
from .math import Memory, policy_observation, use_and_write
from .protocol import digest
from ..r7_shared.context import tensor_digest
from ..r8_ba.methods import correct
from ..r7_shared.numerics import stable, temperature, finite


class Actor(nn.Module):
    def __init__(self, seed):
        super().__init__()
        def network(widths):
            layers=[]
            for i,(a,b) in enumerate(zip(widths,widths[1:])):
                layers.append(nn.Linear(a,b))
                if i<len(widths)-2:layers.append(nn.SiLU())
            return nn.Sequential(*layers)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.use=network([193,128,64,9]);self.write=network([193,64,32,1])
            with torch.no_grad():
                self.use[-1].weight.zero_();self.use[-1].bias.zero_()
                self.use[-1].bias[0]=5*math.atanh(math.log(4)/5)
                self.write[-1].weight.zero_();self.write[-1].bias.zero_()
    def forward(self, observation):
        if observation.shape[-1]!=193:raise ValueError('policy interface')
        x=observation.float();finite(x)
        return 5*torch.cat((self.use(x),self.write(x)),dim=-1).div(5).tanh()
    def train_writer(self, enabled):
        self.write.requires_grad_(enabled)


def detached(memory):
    return Memory(*(x.detach().clone() for x in (memory.m,memory.q,memory.h)))


class Controller:
    def __init__(self, carrier, scale):
        if carrier.rank!=64 or carrier.observation!='global' or carrier.amplitude!=.3:
            raise ValueError('fixed B64/global/.3 carrier')
        self.carrier=copy.deepcopy(carrier);self.carrier.freeze()
        self.carrier.requires_grad_(False)
        self.scale=torch.as_tensor(scale,dtype=torch.float64).clone()
        if self.scale.shape!=(64,) or not torch.isfinite(self.scale).all() or (self.scale<1e-3).any():
            raise ValueError('verified fit scale')
        self.carrier.assert_deployment_eta()
    @torch.no_grad()
    def observe(self, raw, tokens):
        a=self.carrier.observe(raw,tokens)
        return {k:v.detach().clone() for k,v in a.items()}
    def propose(self, observation, memory):
        memory.check();a=observation;d=a['d']
        difference=memory.h*d-memory.q
        prior=stable(self.carrier.W.double())@memory.m+a['bias']+stable(self.carrier.G.double(),1.)@difference
        proposed,diag=correct(a['H'],a['o'],prior,temperature(self.carrier.cal_raw).double(),5)
        if not torch.allclose(diag['eta'],self.carrier.frozen_eta,rtol=1e-10,atol=1e-12):raise ValueError('frozen solver step')
        obs=policy_observation(d,proposed,memory,self.scale)
        return proposed,obs
    def act(self, actor, observation, memory, epsilon=None, forced_write=None, identity_use=False):
        proposed,features=self.propose(observation,memory)
        mu=actor(features);raw=mu if epsilon is None else mu+.35*epsilon.to(mu)
        use,nxt=use_and_write(proposed,observation['d'],memory,self.scale,raw,
                             forced_write=forced_write,identity_use=identity_use)
        return use,nxt,dict(observation=features,raw_action=raw,mean=mu)


class Host:
    """Image-only deployment. Only m/q/h plus an audit counter cross visits."""
    def __init__(self, segmenter, controller, actor, identity, diagnostic=None):
        if diagnostic not in (None,'RESET_ALL','FORCE_WRITE','CONST_HALF','WARM_STATIC'):raise ValueError('diagnostic')
        self.segmenter,self.controller,self.actor=segmenter,controller,copy.deepcopy(actor).eval()
        self.actor.requires_grad_(False);self.identity=identity;self.diagnostic=diagnostic
        self.state=Memory.zero();self.visits=0;self.failed=False
        self.context=dict(payload=identity,sha256=digest(identity))
        self.frozen_stamp=self._stamp()
    def _stamp(self):
        return tuple((id(p),p._version) for model in (self.actor,self.controller.carrier,self.segmenter.model) for p in list(model.parameters())+list(model.buffers())) if hasattr(self.segmenter,"model") else tuple((id(p),p._version) for p in self.actor.parameters())
    def check_frozen(self,boundary=False):
        if self._stamp()!=self.frozen_stamp:raise ValueError("frozen deployment weights changed")
    @torch.no_grad()
    def step(self,image):
        if self.failed:raise RuntimeError('stopped host')
        try:
            self.check_frozen()
            state=Memory.zero() if self.diagnostic in ('RESET_ALL','WARM_STATIC') else self.state
            _,raw,tokens=self.segmenter(image,observe=True)
            observation=self.controller.observe(raw,tokens)
            w={'FORCE_WRITE':1.,'CONST_HALF':.5,'WARM_STATIC':.5}.get(self.diagnostic)
            use,nxt,a=self.controller.act(self.actor,observation,state,forced_write=w)
            logits=self.segmenter(image,self.controller.carrier.basis@use);finite(logits)
            if logits.shape!=(1,2,512,512):raise ValueError('prediction shape')
            self.state=detached(nxt);self.visits+=1
            return logits.detach(),dict(visit=self.visits,state_committed=True,gain=float(a['raw_action'][0].sigmoid()),
                 write=float(a['raw_action'][9].sigmoid()) if w is None else w,state_norm=float(nxt.m.norm()),mass=float(nxt.h))
        except BaseException:self.failed=True;raise
    def snapshot(self):
        return dict(schema='R10_HOST_V1',identity=self.identity,diagnostic=self.diagnostic,visits=self.visits,
                    memory={k:getattr(self.state,k).clone() for k in ('m','q','h')})
    def restore(self,s):
        if s['schema']!='R10_HOST_V1' or s['identity']!=self.identity or s['diagnostic']!=self.diagnostic:raise ValueError('host identity')
        if set(s['memory'])!={'m','q','h'}:raise ValueError('persistent state allowlist')
        m=Memory(**s['memory']);m.check();self.state=detached(m);self.visits=s['visits'];self.failed=False
