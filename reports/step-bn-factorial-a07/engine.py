"""Unchanged domain operator with a fixed parameter/buffer factorial intervention."""
import copy
import numpy as np
import torch
import torch.nn.functional as F




def clone_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: clone_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(clone_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def seeded(key):
    torch.manual_seed(20261022 + key)
    np.random.seed((20261022 + key) % (2**32))


def uncertainty(logits):
    p = logits.sigmoid()
    u = -p * torch.log(p + 1e-30)
    assert torch.isfinite(u).all()
    span = u.max() - u.min()
    assert span > 0, 'constant one-sided entropy: not silently regularized'
    return (u - u.min()) / span


def rectangles(height, width, centers, side):
    """Exact union of the official clipped integer squares, without a GPU loop."""
    diff = np.zeros((height + 1, width + 1), np.int32)
    centers = np.asarray(centers, dtype=np.int64).reshape(-1, 2)
    lo = np.maximum(centers - side // 2, 0)
    hi = np.minimum(centers + side // 2 + 1, (height, width))
    np.add.at(diff, (lo[:, 0], lo[:, 1]), 1)
    np.add.at(diff, (hi[:, 0], lo[:, 1]), -1)
    np.add.at(diff, (lo[:, 0], hi[:, 1]), -1)
    np.add.at(diff, (hi[:, 0], hi[:, 1]), 1)
    return diff.cumsum(0).cumsum(1)[:height, :width] > 0


def structure_mask(logits):
    u = uncertainty(logits)
    result = torch.zeros_like(u)
    for b in range(u.shape[0]):
        for h in range(u.shape[1]):
            valid = u[b, h] > 1 - torch.quantile(u[b, h].flatten(), .1)
            centers = valid.nonzero()
            n = len(centers)
            if not n:
                continue
            side = max(int(torch.sqrt(valid.sum().float()).item()), 1)
            take = max(int(n * .1), 1)
            centers = centers[torch.randperm(n)[:take]].cpu().numpy()
            result[b, h] = torch.from_numpy(rectangles(*u.shape[-2:], centers, side)).to(u.device)
    return result


class StyleMemory:
    def __init__(self):
        self.items = {}

    def apply_and_push(self, x):
        fft = torch.fft.fft2(x)
        amplitude = torch.fft.fftshift(fft.abs())
        h, w = x.shape[-2:]
        slices = (slice(None), slice(None), slice(h//2-2, h//2+3), slice(w//2-2, w//2+3))
        current = amplitude[slices].detach().cpu().numpy().astype(np.float32)
        output = x.clone()
        if self.items:
            keys = np.stack([np.frombuffer(k, np.float32) for k in self.items])
            query = current.reshape(-1)
            denom = np.linalg.norm(keys, axis=1) * np.linalg.norm(query)
            assert np.all(denom > 0)
            similarity = keys @ query / denom
            selected = np.argpartition(similarity, -1)[-1]
            amplitude[slices] = torch.as_tensor(list(self.items.values())[selected], device=x.device)
            amplitude = torch.fft.ifftshift(amplitude)
            phase = fft.angle()
            output = torch.fft.ifft2(torch.complex(torch.cos(phase)*amplitude, torch.sin(phase)*amplitude)).real
        # Preserve the reference insertion condition (effective maximum 41).
        if len(self.items) > 40:
            self.items.pop(next(iter(self.items)))
        self.items[current.reshape(-1).tobytes()] = current.copy()
        return output


def segmentation_loss(logits, target):
    p = logits.sigmoid()
    return F.binary_cross_entropy(p, target) + 1 - (2*(p*target).sum()+1)/(p.sum()+target.sum()+1)


class Arm:
    def __init__(self, source, name, meter):
        self.name = name
        self.keep_parameters = name != 'BN_ONLY'
        self.source = source
        self.keep_buffers = True
        self.model = copy.deepcopy(source).requires_grad_(True)
        self.params = list(self.model.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=1e-4, betas=(.9,.999), eps=1e-8,
                                    weight_decay=.01, foreach=False)
        self.support = [p.detach().abs() < 2e-4 for p in self.params]
        if name == 'RANDOM':
            generator=torch.Generator().manual_seed(20261022)
            random_support=[]
            for p,m in zip(self.params,self.support):
                ix=torch.randperm(p.numel(),generator=generator)[:int(m.sum())].to(p.device)
                chosen=torch.zeros(p.numel(),dtype=torch.bool,device=p.device);chosen[ix]=True
                random_support.append(chosen.reshape_as(p))
            self.support=random_support
        self.style = StyleMemory()
        self.meter = meter
        self.support_checks = 0

    def forward(self, x):
        captured = []
        handle = self.model.up5.register_forward_hook(lambda mod, args, out: captured.append(out))
        try:
            self.meter['forward'] += 1
            z = self.model(x)[0]
        finally:
            handle.remove()
        assert len(captured) == 1 and captured[0].shape[1] == 32
        return z, captured[0]

    def step(self, loss, masks):
        assert torch.isfinite(loss)
        self.opt.zero_grad(set_to_none=True)
        self.meter['backward'] += 1
        loss.backward()
        before = [p.detach().clone() for p in self.params]
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.params)
        self.opt.step()
        with torch.no_grad():
            for p, old, mask in zip(self.params, before, masks):
                p.copy_(torch.where(mask, p, old))
                assert torch.equal(p[~mask], old[~mask])
        self.meter['updates'] += 1
        self.support_checks += 1

    def buffers(self):
        return {n:b.detach().clone() for n,b in self.model.named_buffers()}

    def restore_buffers(self,values):
        with torch.no_grad():
            for n,b in self.model.named_buffers():b.copy_(values[n])
        assert all(torch.equal(b,values[n]) for n,b in self.model.named_buffers())

    def run(self,x,teacher_logits,teacher_features,key):
        if self.name in ('RESET_ALL','MEMORY_ONLY'):
            self.model.load_state_dict(self.source.state_dict(),strict=True)
            self.opt.state.clear();self.opt.zero_grad(set_to_none=True)
            if self.name=='RESET_ALL':self.style=StyleMemory()
        self.model.eval()
        with torch.no_grad():
            pre=teacher_logits.clone() if self.name in ('RESET_ALL','MEMORY_ONLY') else self.forward(x)[0]
        before=self.buffers()
        # Rebuild this input's statistics from source; retain parameters and optimizer.
        self.restore_buffers(dict(self.source.named_buffers()))
        seeded(key*10)
        self.model.train()
        with torch.no_grad():self.forward(x)
        observed=self.buffers()
        if self.keep_parameters:
            self.restore_buffers(before)
            seeded(key*10)
            with torch.no_grad():
                u=uncertainty(teacher_logits)
                pseudo=((teacher_logits.sigmoid()>=.5)|(u>=.75)).float()
                transformed=self.style.apply_and_push(x)
                mask=structure_mask(teacher_logits);transformed[:,:2]*=1-mask
            z,features=self.forward(transformed)
            loss=segmentation_loss(z,pseudo)+.1*F.l1_loss(torch.fft.fft2(features).abs(),torch.fft.fft2(teacher_features).abs())
            self.step(loss,self.support);value=float(loss.detach())
            del z,features,loss,transformed,pseudo
        else:
            assert not self.opt.state and all(p.grad is None for p in self.params)
            value=0.
        self.restore_buffers(observed)
        self.model.eval()
        with torch.no_grad():post=self.forward(x)[0]
        parity=None
        if self.name=='REPAIR':
            source_buffers=dict(self.source.named_buffers())
            self.restore_buffers(source_buffers)
            with torch.no_grad():parity=self.forward(x)[0]
            self.restore_buffers(observed)
        assert all(torch.equal(b,observed[n]) for n,b in self.model.named_buffers())
        return pre.detach(),post.detach(),value,parity


def check():
    m=torch.nn.BatchNorm2d(2);state=clone_cpu(m.state_dict())
    m.train();m(torch.arange(32,dtype=torch.float32).reshape(1,2,4,4))
    assert not equal_tree(state,m.state_dict())
    m.load_state_dict(state);assert equal_tree(state,m.state_dict())
