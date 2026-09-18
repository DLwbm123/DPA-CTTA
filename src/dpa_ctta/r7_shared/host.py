"""Transactional image-only host. Labels/evaluator are outside this object."""
import copy
import hashlib
import torch
from torch import nn
from .network import Observer
from .context import (STATE_SCHEMA,make_context,require_context,validate_context,stamp)
from .numerics import COUNTS,finite,shape,project

class Method(nn.Module):
    def __init__(self,basis,static=False):
        super().__init__()
        shape(basis,(1024,self.rank))
        if torch.linalg.matrix_rank(basis.double())!=self.rank:raise ValueError('basis full rank')
        self.register_buffer('basis',basis.detach().double().clone());self.observer=Observer();self.static=bool(static);self.stage='fit'
    def project(self,v):
        if self.group=='A':return project(self.basis,v.double())[0].detach()
        return (self.basis.T@v.double()).detach()
    def code(self,state):return state['m'] if self.group=='A' else state['z']
    def ambient(self,state):return (self.basis@self.code(state)).float()
    def freeze(self):
        self.requires_grad_(False);self.eval();self.set_stage('online')
    def set_stage(self,stage):
        if stage not in ('fit','cal','online'):raise ValueError('method stage')
        self.stage=stage
    def validate_state(self,state):
        initial=self.initial()
        if set(state)!=set(initial) or type(state['counter']) is not int or state['counter']<0:raise ValueError('state schema')
        for k,v in initial.items():
            if isinstance(v,torch.Tensor):
                shape(state[k],v.shape)
                if state[k].dtype!=torch.float64:raise ValueError('state float64')
        if self.group=='A':
            if not torch.allclose(state['P'],state['P'].T,rtol=1e-10,atol=1e-12):raise ValueError('symmetric covariance')
            torch.linalg.cholesky(state['P'])
    def digest(self):
        h=hashlib.sha256()
        h.update(f'{self.group}|{self.static}'.encode())
        for name,value in sorted(self.state_dict().items()):
            h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

class OnlineHost:
    def __init__(self,segmenter,method=None,*,ablation=None,expected_context=None):
        self.segmenter=segmenter;self.method=method;self.ablation=ablation;self.failed=False;self.first_error=None
        self.state=None if method is None else method.initial();self.visits=0
        if method:
            if any(p.requires_grad for p in method.parameters()) or method.stage!='online' or not method.observer.fitted:raise ValueError('frozen prepared method required')
            allowed={'A':('A_ISO_OBS',),'B':('B_PRED_ONLY',),'C':('C_CONST_R',)}[method.group]
            if ablation is not None and (ablation not in allowed or method.static):raise ValueError('FULL-checkpoint deployment ablation only')
            if ablation=='C_CONST_R' and not method.constant_ready:raise ValueError('missing source-cal constant R')
        elif ablation:raise ValueError('C0 has no ablation')
        if expected_context is not None:validate_context(expected_context)
        source=None if expected_context is None else expected_context['payload']['source']
        self._context=make_context(segmenter,method,ablation,source)
        if expected_context is not None:require_context(self._context,expected_context)
        self._stamp=stamp(segmenter,method,ablation)
    def _check_frozen(self,boundary=False):
        if stamp(self.segmenter,self.method,self.ablation)!=self._stamp:raise ValueError('frozen inference environment changed')
        if boundary:
            actual=make_context(self.segmenter,self.method,self.ablation,self._context['payload']['source'])
            require_context(actual,self._context)
    @torch.no_grad()
    def step(self,current_image):
        if self.failed:raise RuntimeError('hard stopped after first error')
        before=COUNTS.copy()
        try:
            self._check_frozen()
            if self.method is None:z=self.segmenter(current_image);next_state=None
            else:
                _,raw,tokens=self.segmenter(current_image,observe=True)
                next_state,_=self.method.update(raw,tokens,self.state,ablation=self.ablation)
                self.method.validate_state(next_state)
                z=self.segmenter(current_image,self.method.ambient(next_state))
                finite(z)
            # No state commit or evaluator release until final prediction succeeds.
            self.state=next_state;self.visits+=1
            return z.detach(),dict(counts=dict(COUNTS-before),visit=self.visits,state_committed=True)
        except Exception as exc:
            self.failed=True;self.first_error=dict(type=type(exc).__name__,message=str(exc),counts=dict(COUNTS-before))
            raise
    def save_state(self):
        self._check_frozen(boundary=True)
        return copy.deepcopy(dict(schema=STATE_SCHEMA,context=self._context,binding=self._context['sha256'],ablation=self.ablation,state=self.state,visits=self.visits,failed=self.failed,first_error=self.first_error))
    def load_state(self,packet):
        if not isinstance(packet,dict) or packet.get('schema')!=STATE_SCHEMA:raise ValueError('legacy/incomplete state packet unsupported')
        self._check_frozen(boundary=True)
        require_context(self._context,packet.get('context'))
        if self.failed or packet.get('failed') or packet.get('first_error') is not None:raise ValueError('failed state is not resumable')
        if packet.get('binding')!=self._context['sha256'] or packet.get('ablation')!=self.ablation:raise ValueError('state binding')
        if type(packet.get('visits')) is not int or packet['visits']<0:raise ValueError('visit counter')
        if self.method:
            self.method.validate_state(packet['state'])
            if packet['state']['counter']!=packet['visits']:raise ValueError('state visit mismatch')
        elif packet['state'] is not None:raise ValueError('C0 state')
        self.state=copy.deepcopy(packet['state']);self.visits=packet['visits']
