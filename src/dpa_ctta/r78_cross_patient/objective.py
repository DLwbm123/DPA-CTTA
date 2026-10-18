"""Relative boundary objective and bounded state; no data or label loading."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def gradient(field):
    padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
    gx = (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]) / 2
    gy = (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1]) / 2
    return gx, gy


def sample(field, xy):
    """Fixed-coordinate bilinear gather retains mixed derivatives of field values."""
    if field.ndim != 3 or xy.shape[-1] != 2 or xy.requires_grad:
        raise ValueError('CxHxW field and fixed (...,2) xy required')
    if not torch.isfinite(xy).all():
        raise ValueError('finite coordinates required')
    height, width = field.shape[-2:]
    x, y = xy.unbind(-1)
    if ((x < 0) | (x > width-1) | (y < 0) | (y > height-1)).any():
        raise ValueError('sample outside field')
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0+1).clamp_max(width-1), (y0+1).clamp_max(height-1)
    dx, dy = x-x0, y-y0
    return ((1-dx)*(1-dy)*field[:, y0, x0] + dx*(1-dy)*field[:, y0, x1]
            + (1-dx)*dy*field[:, y1, x0] + dx*dy*field[:, y1, x1])


@dataclass
class Reference:
    d0: torch.Tensor
    h: torch.Tensor
    inputs: torch.Tensor
    anchors: tuple
    coordinates: tuple
    weights: tuple
    support: torch.Tensor

    @property
    def empty(self):
        return not any(len(a) for a in self.anchors)


@torch.no_grad()
def reference(image, logits, size=256):
    if image.shape[:2] != (1, 3) or logits.shape[:2] != (1, 2):
        raise ValueError('one RGB image and two nested sigmoid logits required')
    if not torch.isfinite(image).all() or not torch.isfinite(logits).all():
        raise ValueError('finite input required')
    rgb = F.interpolate(image, (size, size), mode='bilinear', align_corners=False)
    rgb = (rgb-rgb.mean((-2, -1), keepdim=True))/rgb.std((-2, -1), keepdim=True, correction=0).clamp_min(1e-6)
    z0 = F.interpolate(logits, (size, size), mode='bilinear', align_corners=False)
    gx, gy = gradient(-z0)
    h = (gx.square()+gy.square()+1e-6).sqrt().clamp_min(.1)
    d0 = -z0/h
    dx, dy = gradient(d0)
    length = (dx.square()+dy.square()).sqrt()
    nx, ny = dx/length.clamp_min(1e-6), dy/length.clamp_min(1e-6)
    ix, iy = gradient(rgb)
    edge = (ix.square()+iy.square()).mean(1, keepdim=True).sqrt()
    q = z0.sigmoid()
    weight = 4*q*(1-q)
    weight[:, 1:2] *= torch.sigmoid(-d0[:, :1]/.5)
    yy, xx = torch.meshgrid(torch.arange(size, device=z0.device),
                            torch.arange(size, device=z0.device), indexing='ij')
    border = (xx >= 2.5) & (xx <= size-3.5) & (yy >= 2.5) & (yy <= size-3.5)
    inputs, anchors, coordinates, weights, supports = [], [], [], [], []
    for channel in range(2):
        valid = (d0[0, channel].abs() <= 1) & (length[0, channel] > 1e-6) & border
        supports.append(valid)
        yx = valid.nonzero()
        if len(yx) > 128:
            index = torch.linspace(0, len(yx)-1, 128, device=z0.device).round().long()
            yx = yx[index]
        y, x = yx.unbind(1)
        centre = torch.stack((x, y), -1).to(z0)
        normal = torch.stack((nx[0, channel, y, x], ny[0, channel, y, x]), -1)
        xy = centre[:, None] + z0.new_tensor([-.5, 0, .5])[None, :, None]*normal[:, None]
        inputs.append(torch.cat((rgb, edge, q[:, channel:channel+1],
                                 d0[:, channel:channel+1].clamp(-3, 3),
                                 nx[:, channel:channel+1], ny[:, channel:channel+1],
                                 torch.full_like(q[:, :1], channel)), 1))
        anchors.append(yx)
        coordinates.append(xy)
        weights.append(weight[0, channel, y, x])
    return Reference(d0.detach(), h.detach(), torch.cat(inputs).detach(),
                     tuple(anchors), tuple(coordinates), tuple(weights),
                     torch.stack(supports).unsqueeze(0))


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, inputs, geometry_only=False):
        if geometry_only:
            inputs = torch.cat((torch.zeros_like(inputs[:, :4]), inputs[:, 4:]), 1)
        return self.layers(inputs).tanh()


def energy(logits, ref, delta):
    if delta.shape != (2, 1, *ref.d0.shape[-2:]):
        raise ValueError('one signed field per anatomical channel required')
    z = F.interpolate(logits, ref.d0.shape[-2:], mode='bilinear', align_corners=False)
    d = -z/ref.h
    terms = []
    for channel, yx in enumerate(ref.anchors):
        if len(yx) == 0:
            continue
        y, x = yx.unbind(1)
        relative = sample(d[0, channel:channel+1]-ref.d0[0, channel:channel+1],
                          ref.coordinates[channel])[0]
        error = relative+delta[channel, 0, y, x][:, None]
        loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.5, reduction='none')
        weight = ref.weights[channel]
        terms.append((loss.mean(1)*weight).sum()/weight.sum().clamp_min(1e-12))
    return torch.stack(terms).mean() if terms else logits.sum()*0+delta.sum()*0


def project(state):
    return state * (.05/state.norm().clamp_min(.05))


def update(state, loss, create_graph=False):
    grad, = torch.autograd.grad(loss, state, create_graph=create_graph)
    return project(state-.01*grad)


class ParameterPredictor(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.readout = nn.Linear(16, dimensions)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, inputs, empty=False):
        output = self.readout(self.features(inputs).mean((0, 2, 3)))
        return output*0 if empty else project(output)


def direct_logits(logits, ref, delta):
    residual = ref.h*delta[:, 0].unsqueeze(0)*ref.support
    return logits+F.interpolate(residual, logits.shape[-2:], mode='bilinear', align_corners=False)


@torch.no_grad()
def ribbon(ref):
    """SRDF appearance scoring, with shared relative energy and fixed support."""
    result = torch.zeros((2, 1, *ref.d0.shape[-2:]), device=ref.d0.device, dtype=ref.d0.dtype)
    for channel, yx in enumerate(ref.anchors):
        if not len(yx):
            continue
        y, x = yx.unbind(1)
        centre = torch.stack((x, y), -1).to(ref.d0)
        normal = ref.inputs[channel, 6:8, y, x].T
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), -1)
        descriptor = ref.inputs[channel, :4]
        scores = []
        # Priority 0,-1,+1 resolves all exact ties in advance.
        for delta in (0, -1, 1):
            sides = []
            for offsets in ((-1.5, -.5), (.5, 1.5)):
                positions = torch.stack([centre+(delta+n)*normal+t*tangent
                                         for n in offsets for t in (-1, 0, 1)], 1)
                values = sample(descriptor, positions)
                mean = values.mean(-1)
                dispersion = ((values-mean[..., None]).square().mean((0, 2))+1e-6).sqrt()
                sides.append((mean, dispersion))
            between = ((sides[0][0]-sides[1][0]).square().mean(0)+1e-6).sqrt()
            within = (sides[0][1]+sides[1][1])*.5
            scores.append((between-within)/(between+within).clamp_min(1e-6))
        choice = torch.stack(scores).argmax(0)
        result[channel, 0, y, x] = ref.d0.new_tensor([0, -1, 1])[choice]
    return result


class BNState:
    """Functional up3 affine offsets; current-image BN statistics never persist."""
    def __init__(self, model):
        self.model = model.train().requires_grad_(False)
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if type(module) is not nn.BatchNorm2d:
                    raise ValueError('standard BatchNorm2d required')
                module.track_running_stats = False
                module.running_mean = module.running_var = None
        self.names = ('up3.bn.weight', 'up3.bn.bias')
        self.parameters = dict(model.named_parameters())
        if any(name not in self.parameters for name in self.names):
            raise ValueError('up3 BN affine parameters missing')
        self.channels = self.parameters[self.names[0]].numel()
        self.dimensions = 2*self.channels

    def zero(self):
        return self.parameters[self.names[0]].new_zeros(self.dimensions, requires_grad=True)

    def __call__(self, image, state):
        if state.shape != (self.dimensions,):
            raise ValueError('wrong adaptation state dimension')
        parameters = dict(self.parameters)
        for name, offset in zip(self.names, state.split(self.channels)):
            parameters[name] = self.parameters[name]+offset
        # Cloned buffers prohibit query reads from writing persistent model state.
        buffers = {name: value.clone() for name, value in self.model.named_buffers()}
        result = torch.func.functional_call(self.model, (parameters, buffers), (image,))
        return result[0] if isinstance(result, (tuple, list)) else result
