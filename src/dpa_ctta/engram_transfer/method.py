"""ENGRAM equations (2)-(4), Algorithm 2, default Table 10 settings.

Independent module implementation: https://openreview.net/pdf?id=bkwlR0ND2g.
The frozen source convolutional stem replaces the paper's VGG descriptor.
"""
import copy
import torch
from torch.nn import functional as F
from ..r37_cw_orientation.method import Host as RetainedHost, candidates
from ..hosts.vptta import model_input_from_pixels

ARMS = ('NATIVE', 'ENGRAM')


def configuration(arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    c = copy.deepcopy(candidates()[3])
    c.update(id='CW_LSO_'+arm, memory_arm=arm)
    return c


class Memory:
    def __init__(self, source, key_dim):
        self.source = source.detach().double().clone()
        self.U = self.source.new_empty((len(source), 0))
        self.M = self.source.new_empty((0, key_dim))
        self.average = torch.zeros_like(self.source)
        self.previous = self.source.new_zeros(key_dim)
        self.n = 0

    def correction(self, key):
        return self.U @ (self.M @ (key.double()-self.previous))

    @torch.no_grad()
    def write(self, current, key):
        key = key.double(); y = current.double()-self.source
        self.M.add_(.1 * torch.outer(self.U.T @ y - self.M @ key, key))
        # Appendix A permits the difference-of-reads form, with growth from P_perp y.
        outside = y-self.U @ (self.U.T @ y)
        self.average.mul_(.9).add_(outside, alpha=.1); self.n += 1
        z = (self.average-self.U @ (self.U.T @ self.average))/(1-.9**self.n)
        grew = False
        if self.n >= 10 and self.U.shape[1] < min(64, len(y)) and z.norm() > .25*y.norm():
            for _ in range(2):
                z = z-self.U @ (self.U.T @ z)
            if z.norm() > 1e-12:
                self.U = torch.cat((self.U, (z/z.norm())[:, None]), dim=1)
                self.M = torch.cat((self.M, self.M.new_zeros((1, len(key)))), dim=0)
                self.average.zero_(); self.n = 0; grew = True
        self.previous = key.detach().clone()
        if not all(torch.isfinite(v).all() for v in (self.U,self.M,self.average)):
            raise ValueError('nonfinite associative memory')
        return grew

    def snapshot(self):
        return dict(U=self.U.clone(), M=self.M.clone(), average=self.average.clone(),
                    previous=self.previous.clone(), n=self.n)

    def restore(self, state):
        for name in ('U','M','average','previous'):
            setattr(self, name, state[name].clone().to(self.source))
        self.n = state['n']


class Host(RetainedHost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = self.stem = None
        if self.config['memory_arm'] == 'ENGRAM':
            self.stem = copy.deepcopy(self.native.model.res.conv1).eval().requires_grad_(False)
            self.memory = Memory(self.vector(), self.stem.out_channels)

    def vector(self):
        return torch.cat([p.detach().flatten() for p in self.native.params])

    @torch.no_grad()
    def key(self, x):
        feature = self.stem(model_input_from_pixels(x, 'fundus').to(self.device))
        k = feature.double().var(dim=(0,2,3), unbiased=False).clamp_min(1e-6).log()
        k = k-k.mean()
        if self.config.get('constant_key', False) or k.norm() < 1e-12:
            k = torch.zeros_like(k); k[0] = 1.
        return F.normalize(k,dim=0)

    def step(self, x):
        if self.memory is None:
            return super().step(x)
        with torch.no_grad():
            k = self.key(x); delta = self.memory.correction(k)
            shift = float(delta.norm()); key_shift = float((k-self.memory.previous).norm())
            if shift:
                offset=0
                for p in self.native.params:
                    n=p.numel(); p.add_(delta[offset:offset+n].reshape_as(p).to(p)); offset+=n
        z, detail = super().step(x)
        current = self.vector().clone()
        grew = self.memory.write(current, k)
        if not torch.equal(current, self.vector()):
            raise ValueError('memory write changed completed host parameters')
        diagnostics = dict(recall_norm=shift, key_shift=key_shift, memory_rank=self.memory.U.shape[1],
                           grew=int(grew), write_conserved=True, stem_forwards=1)
        detail['diagnostics'].update(diagnostics)
        return z, detail

    def snapshot(self):
        state = super().snapshot()
        if self.memory is not None:
            state['associative_memory'] = self.memory.snapshot()
        return state

    def restore(self, state):
        super().restore(state)
        if self.memory is not None:
            self.memory.restore(state['associative_memory'])


def self_check():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    source=torch.zeros(20,dtype=torch.float64); key=F.normalize(torch.randn(5,dtype=torch.float64),dim=0)
    memory=Memory(source,5); current=source.clone()
    for t in range(35):
        assert torch.equal(memory.correction(key),torch.zeros_like(source))
        current += torch.arange(20,dtype=torch.float64)/1000
        before=current.clone();memory.write(current,key)
        assert torch.equal(current,before)
    assert memory.U.shape[1] >= 1 and memory.M.norm()>0
    assert torch.allclose(memory.U.T@memory.U,torch.eye(memory.U.shape[1],dtype=torch.float64),atol=1e-10)
    newkey=F.normalize(torch.randn(5,dtype=torch.float64),dim=0)
    # Explicit compensated read and difference-of-reads must agree.
    residual=current-source-memory.U@(memory.M@key)
    explicit=source+memory.U@(memory.M@newkey)+residual
    assert torch.allclose(explicit,current+memory.correction(newkey),atol=1e-12)
    assert memory.correction(newkey).norm()>0
    snap=memory.snapshot();delta=memory.correction(newkey).clone()
    memory.write(current+.01,newkey);memory.restore(snap)
    assert torch.equal(delta,memory.correction(newkey))
    # Delta rule reduces coefficient read error by exactly (1-beta) for a unit key.
    old=memory.U.T@(current-source)-memory.M@newkey
    memory.write(current,newkey)
    new=memory.U.T@(current-source)-memory.M@newkey
    assert torch.allclose(new,.9*old,atol=1e-12)
    print('ENGRAM_MEMORY_SELF_CHECK_PASS')
