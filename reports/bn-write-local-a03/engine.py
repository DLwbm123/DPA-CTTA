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
    def __init__(self, source, meter):
        self.model=copy.deepcopy(source).requires_grad_(False)
        self.style=StyleMemory();self.meter=meter

    def forward(self,x):
        self.meter['forward']+=1
        return self.model(x)[0]

    def buffers(self):
        return {n:b.detach().clone() for n,b in self.model.named_buffers()}

    def restore(self,values):
        with torch.no_grad():
            for n,b in self.model.named_buffers():b.copy_(values[n])
        assert all(torch.equal(b,values[n]) for n,b in self.model.named_buffers())

    def run(self,x,teacher,key,selected):
        seeded(key*10)
        before=self.buffers()
        style=self.style.apply_and_push(x)
        mask=structure_mask(teacher)
        full=style.clone();full[:,:2]*=1-mask
        random_before=(torch.get_rng_state(),torch.cuda.get_rng_state())
        self.model.train();self.forward(full)
        random_after=(torch.get_rng_state(),torch.cuda.get_rng_state())
        after=self.buffers()
        self.model.eval();anchor=self.forward(x)
        outputs=[]
        if selected:
            self.restore(before);self.model.eval();hold=self.forward(x)
            assert all(torch.equal(b,before[n]) for n,b in self.model.named_buffers())
            masked=x.clone();masked[:,:2]*=1-mask
            outputs=[hold]
            for augmented in (x,style,masked):
                self.restore(before)
                torch.set_rng_state(random_before[0]);torch.cuda.set_rng_state(random_before[1])
                self.model.train();self.forward(augmented)
                self.model.eval();outputs.append(self.forward(x))
            outputs.append(anchor)
        self.restore(after)
        torch.set_rng_state(random_after[0]);torch.cuda.set_rng_state(random_after[1])
        return anchor,outputs


def check():
    m=torch.nn.BatchNorm2d(2)
    initial=clone_cpu(m.state_dict())
    m.train();m(torch.arange(32,dtype=torch.float32).reshape(1,2,4,4))
    assert not equal_tree(initial,m.state_dict())
    m.load_state_dict(initial);assert equal_tree(initial,m.state_dict())
