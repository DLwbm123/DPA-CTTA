"""Keep the R20 host; add independent geometry or bounded contribution balance."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import label
from ..r20_model_only_search.method import Host as ExistingHost, weights_for


def candidates():
    w = dict(variance_temperature=.1, boundary_boost=2.)
    baseline = dict(id='W', family='pixel_weight', host='G', params=w,
                    final_lr_multiplier=1.5)
    result = [dict(id='C', family='control', host='C', params={}), baseline]
    for name, module in [('TP', 'persistence'), ('LSO', 'orientation'),
                         ('LSM', 'magnitude'), ('LS', 'both'), ('BAL', 'balance')]:
        c = copy.deepcopy(baseline)
        c.update(id='W_'+name, module=module, module_strength=.5 if name == 'TP' else .1)
        result.append(c)
    return result


def smooth(x, kernel=3):
    return F.avg_pool2d(F.pad(x, (kernel//2,)*4, mode='replicate'), kernel, 1)


def structure(field):
    channels = field.shape[1]
    k = field.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    pad = F.pad(field, (1,)*4, mode='replicate')
    gx = F.conv2d(pad, k[None, None].repeat(channels, 1, 1, 1), groups=channels)
    gy = F.conv2d(pad, k.T[None, None].repeat(channels, 1, 1, 1), groups=channels)
    xx, yy, xy = smooth(gx.square()), smooth(gy.square()), smooth(gx*gy)
    vector = torch.stack((xx-yy, 2*xy), dim=2)
    norm = (vector.square().sum(2)+1e-12).sqrt()
    return vector / norm.unsqueeze(2), norm


def local_structure_loss(logits, target, reliability):
    # Each nested sigmoid structure has its own tensor; OD and OC are not softmax classes.
    pred = smooth(F.avg_pool2d(logits.sigmoid(), 4))
    ref = smooth(F.avg_pool2d(target.detach(), 4))
    po, pm = structure(pred)
    to, tm = structure(ref)
    pn = pm / pm.amax((-2, -1), keepdim=True).clamp_min(1e-6)
    tn = tm / tm.amax((-2, -1), keepdim=True).clamp_min(1e-6)
    weight = (tn * F.avg_pool2d(reliability.detach(), 4) * (tm > 2e-6)).detach()
    denominator = weight.sum().clamp_min(1e-6)
    orientation = ((1-(po*to).sum(2).clamp(-1, 1)) * weight).sum() / denominator
    magnitude = ((pn-tn).abs()*weight).sum() / denominator
    return orientation, magnitude


@torch.no_grad()
def persistence_weight(target):
    """Cross-threshold component overlap, not a reproduction of PH/OT TopoOT."""
    small = F.avg_pool2d(target.detach(), 4).cpu().numpy()
    output = np.zeros_like(small)
    for b in range(small.shape[0]):
        for c in range(small.shape[1]):
            a = small[b, c]
            centre, count = label(a >= .5)
            if not count:
                continue
            areas = np.bincount(centre.ravel(), minlength=count+1)
            stability = np.ones(count+1)
            stability[0] = 0
            for threshold in (.3, .4, .6, .7):
                other, _ = label(a >= threshold)
                for index in range(1, count+1):
                    support = centre == index
                    overlaps = np.bincount(other[support])
                    if len(overlaps) <= 1 or not overlaps[1:].any():
                        stability[index] = 0
                        continue
                    match = overlaps[1:].argmax()+1
                    union = areas[index]+np.count_nonzero(other == match)-overlaps[match]
                    stability[index] = min(stability[index], overlaps[match]/max(1, union))
            output[b, c] = stability[centre]
    weight = torch.from_numpy(output).to(target)
    return F.interpolate(weight, size=target.shape[-2:], mode='nearest')


class Host(ExistingHost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frequency = torch.full((1, 2, 1, 1), .5, device=self.device)
        self.pending_frequency = None

    def _criterion(self, logits, pseudo):
        base = super()._criterion(logits, pseudo)
        module = self.config.get('module')
        strength = float(self.config.get('module_strength', 0))
        if not module or not strength:
            return base
        q = pseudo.detach()
        p = self.modules['W']
        reliability = weights_for(torch.stack(self.weak).sigmoid(),
                                  p['variance_temperature'], p['boundary_boost']).to(logits)
        if module == 'persistence':
            extra = 1+strength*persistence_weight(q)
            extra = extra / extra.mean((-2, -1), keepdim=True)
            self.diag['structure_weight_std'] = float(extra.std())
            return (F.binary_cross_entropy_with_logits(logits, q, reduction='none')*
                    reliability*extra).mean()
        if module == 'balance':
            fg = q >= .5
            self.pending_frequency = fg.float().mean((-2, -1), keepdim=True)
            freq = self.frequency.clamp(.05, .95)
            extra = torch.where(fg, .5/freq, .5/(1-freq)).clamp(.25, 4.)
            extra = 1+strength*(extra-1)
            extra = extra / extra.mean((-2, -1), keepdim=True)
            self.diag['balance_foreground_frequency'] = self.frequency.flatten().tolist()
            self.diag['balance_weight_std'] = float(extra.std())
            return (F.binary_cross_entropy_with_logits(logits, q, reduction='none')*
                    reliability*extra.detach()).mean()
        orientation, magnitude = local_structure_loss(logits, q, reliability)
        extra = {'orientation': orientation, 'magnitude': magnitude,
                 'both': (orientation+magnitude)/2}[module]
        self.diag.update(LSO=float(orientation.detach()), LSM=float(magnitude.detach()),
                         module_loss=float(extra.detach()), module_coefficient=strength)
        return base+strength*extra

    def step(self, x):
        z, detail = super().step(x)
        if self.pending_frequency is not None:
            self.frequency.mul_(.9).add_(self.pending_frequency, alpha=.1)
            self.pending_frequency = None
        return z, detail

    def snapshot(self):
        saved = super().snapshot()
        saved['module_frequency'] = self.frequency.detach().cpu().clone()
        return saved

    def restore(self, saved):
        super().restore(saved)
        self.frequency.copy_(saved['module_frequency'].to(self.frequency))
        self.pending_frequency = None
