"""Sparse domain updates with explicitly isolated sample-refinement state.

The PixelContrastLoss dependency is loaded from an inspected, private copy of
the official TACT source. No data loader or official training entry is executed.
"""
import copy
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F


QUEUE_FIELDS = ('segment_queue', 'segment_queue_ptr', 'pixel_queue', 'pixel_queue_ptr')


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


def load_contrast(path):
    spec = importlib.util.spec_from_file_location('registered_contrast', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PixelContrastLoss


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


def class_labels(binary):
    # Nested independent heads, with OC taking precedence in contrast labels.
    out = torch.zeros_like(binary[:, :1], dtype=torch.float32)
    out[:, 0][binary[:, 0]] = 1
    out[:, 0][binary[:, 1]] = 2
    return out


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


def transform(x, k):
    if k == 0:
        return x.flip(-1)
    if k == 1:
        return x.flip(-2)
    if k == 2:
        return x.flip(-1).transpose(-2, -1)
    if k == 3:
        return x.flip(-1).flip(-2)
    if k == 4:
        return x.transpose(-2, -1).flip(-1)
    return x


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
    def __init__(self, source, name, contrast_class, meter):
        self.name = name
        self.model = copy.deepcopy(source).requires_grad_(True)
        self.params = list(self.model.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=1e-4, betas=(.9, .999), eps=1e-8,
                                     weight_decay=.01, foreach=False)
        self.support = [p.detach().abs() < 2e-4 for p in self.params]
        self.random_indices = None
        if name == 'RANDOM_OWNED':
            generator = torch.Generator().manual_seed(20261022)
            self.random_indices = [torch.randperm(p.numel(), generator=generator)[:int(m.sum())].to(p.device)
                                   for p, m in zip(self.params, self.support)]
            self.support = []
            for p, index in zip(self.params, self.random_indices):
                mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
                mask[index] = True
                self.support.append(mask.reshape_as(p))
        self.style = StyleMemory()
        seeded(0)
        self.contrast = contrast_class(temperature=.5, max_samples=1024, max_views=5,
                                      memory=True, memory_size=500, pixel_classes=3, dim=32,
                                      device=str(next(source.parameters()).device))
        self.meter = meter
        self.restores = 0
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

    def snapshot(self, full):
        result = {'model': clone_cpu(self.model.state_dict())}
        if full:
            result['optimizer'] = clone_cpu(self.opt.state_dict())
            result['queue'] = {n: clone_cpu(getattr(self.contrast, n)) for n in QUEUE_FIELDS}
        return result

    def restore(self, snap):
        self.model.load_state_dict(snap['model'], strict=True)
        if 'optimizer' in snap:
            self.opt.load_state_dict(snap['optimizer'])
            for name, value in snap['queue'].items():
                setattr(self.contrast, name, value.clone())
        self.opt.zero_grad(set_to_none=True)
        assert equal_tree(self.model.state_dict(), snap['model'])
        if 'optimizer' in snap:
            assert equal_tree(self.opt.state_dict(), snap['optimizer'])
            assert equal_tree({n: getattr(self.contrast, n) for n in QUEUE_FIELDS}, snap['queue'])
        self.restores += 1

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

    def run(self, x, teacher_logits, teacher_features, key, matched_counts=None):
        seeded(key * 10)
        self.model.train()
        with torch.no_grad():
            u = uncertainty(teacher_logits)
            pseudo = ((teacher_logits.sigmoid() >= .5) | (u >= .75)).float()
            transformed = self.style.apply_and_push(x)
            mask = structure_mask(teacher_logits)
            transformed[:, :2] *= 1 - mask
        z, features = self.forward(transformed)
        first_loss = segmentation_loss(z, pseudo) + .1*F.l1_loss(torch.fft.fft2(features).abs(),
                                                                 torch.fft.fft2(teacher_features).abs())
        self.step(first_loss, self.support)
        first_value = float(first_loss.detach())
        del features, z, first_loss, transformed, pseudo
        if self.name == 'DOMAIN_ONLY':
            self.model.eval()
            with torch.no_grad():
                prediction, _ = self.forward(x)
            return prediction, {'domain_loss': first_value, 'refine_loss': 0.}, None

        full = self.name != 'WEIGHT_ONLY'
        snap = self.snapshot(full)
        seeded(key * 10 + 1)
        self.model.eval()
        with torch.no_grad():
            current, _ = self.forward(x)
            targets = [current]
            for k in range(5):
                q, _ = self.forward(transform(x, k))
                targets.append(transform(q, (0, 1, 4, 3, 2)[k]))
            target = torch.stack(targets).mean(0)
            u = uncertainty(current)
            binary = current.sigmoid() >= .5
            low = class_labels(binary & (u < .75))
        self.model.train()
        z, features = self.forward(x)
        contrast = self.contrast(F.normalize(features, p=2, dim=1), low.detach(),
                                 class_labels(z.detach().sigmoid() >= .5),
                                 plabel=True, en_map=u.detach(), low=low.detach())
        # The queue observes stage-1 weights before refinement updates them.
        # Commit these observations; only ALL_RESET discards them.
        if self.name in ('OWNED_STATE', 'RANDOM_OWNED'):
            snap['queue'] = {n: clone_cpu(getattr(self.contrast, n)) for n in QUEUE_FIELDS}
        second_loss = .1*contrast + .1*(-z.sigmoid()*torch.log(target.sigmoid()+1e-8)).sum(1).mean()
        if self.name == 'RANDOM_OWNED':
            assert matched_counts is not None
            masks = []
            for p, index, count in zip(self.params, self.random_indices, matched_counts):
                assert 0 <= count <= len(index)
                mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
                mask[index[:count]] = True
                masks.append(mask.reshape_as(p))
        else:
            masks = [support & (p.detach().abs() < 2e-4) for p, support in zip(self.params, self.support)]
        counts = [int(mask.sum()) for mask in masks]
        self.step(second_loss, masks)
        second_value = float(second_loss.detach())
        del features, z, second_loss, contrast, targets, target
        self.model.eval()
        with torch.no_grad():
            prediction, _ = self.forward(x)
        self.restore(snap)
        return prediction, {'domain_loss': first_value, 'refine_loss': second_value}, counts


def check():
    centers = np.array([[0, 0], [2, 3], [6, 6]])
    actual = rectangles(7, 7, centers, 4)
    expected = np.zeros((7, 7), bool)
    for a, b in centers:
        expected[max(0,a-2):min(7,a+3), max(0,b-2):min(7,b+3)] = True
    assert np.array_equal(actual, expected)
    # Optimizer moments, step and unregistered queues must all restore.
    parameter = torch.nn.Parameter(torch.tensor([1e-5, 1.]))
    opt = torch.optim.AdamW([parameter], lr=1e-4, foreach=False)
    parameter.sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    saved = clone_cpu(opt.state_dict())
    (-parameter.sum()).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    assert not equal_tree(saved, opt.state_dict())
    opt.load_state_dict(saved)
    assert equal_tree(saved, opt.state_dict())
    for k in range(5):
        x = torch.arange(25).reshape(1, 1, 5, 5)
        assert torch.equal(transform(transform(x, k), (0, 1, 4, 3, 2)[k]), x)
