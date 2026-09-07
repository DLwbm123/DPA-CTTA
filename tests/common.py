import copy

import torch
from torch import nn
import torch.nn.functional as F

from dpa_ctta.anchors import DistilledPotentialAnchor
from dpa_ctta.atlas import DistilledPotentialAtlas
from dpa_ctta.descriptors import DescriptorScaler, FrozenDescriptor
from dpa_ctta.manifold import MultiScaleLatentFiLM
from dpa_ctta.online import DPAOnlineAdapter


class TinySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.low = nn.Conv2d(4, 4, 3, padding=1)
        self.mid = nn.Conv2d(4, 4, 3, padding=1)
        self.tail = nn.Conv2d(4, 1, 1)

    def forward(self, image):
        feature = self.early(image)
        low = F.relu(self.low(F.avg_pool2d(feature, 4)))
        mid = F.relu(self.mid(F.interpolate(low, scale_factor=2, mode="nearest")))
        return self.tail(F.interpolate(mid, size=image.shape[-2:], mode="nearest"))


class FailingSource(TinySource):
    def forward(self, image):
        self.early(image)
        raise RuntimeError("synthetic failure")


def frozen_feature(image):
    weight = torch.linspace(-0.1, 0.1, 12, device=image.device, dtype=image.dtype).reshape(4, 3, 1, 1)
    return F.conv2d(image, weight)


def make_anchor(task="polyp", offset=0.0, requires_grad=True):
    descriptor = FrozenDescriptor(task, feature_channels=4)
    classes = descriptor.num_classes
    anchor = DistilledPotentialAnchor(
        torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16) + offset,
        torch.linspace(-0.5, 0.5, classes * 16 * 16).reshape(1, classes, 16, 16) - offset,
        torch.linspace(-0.2, 0.2, 16) + offset,
        torch.arange(16 * descriptor.output_dim, dtype=torch.float32).reshape(16, descriptor.output_dim) * 1e-5,
        torch.linspace(-1, 1, 16) + offset,
    )
    if not requires_grad:
        anchor.requires_grad_(False)
    return anchor, descriptor


def make_atlas(task="polyp"):
    first, descriptor = make_anchor(task, -0.2)
    second, _ = make_anchor(task, 0.3)
    scaler = DescriptorScaler(descriptor.output_dim)
    return DistilledPotentialAtlas([first, second], descriptor, scaler, temperature=0.7)


def make_online():
    torch.manual_seed(7)
    source = TinySource().eval()
    wrapper = MultiScaleLatentFiLM(source, {"low": 4, "mid": 4, "tail": 1})
    atlas = make_atlas("polyp")
    return DPAOnlineAdapter(wrapper, atlas, "early", temporal_lambda=1.0, precision_floor=1e-3)


def clone_online(adapter):
    clone = make_online()
    clone.load_state_dict(copy.deepcopy(adapter.state_dict()))
    return clone
