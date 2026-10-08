"""Generated inputs exercise optimizer ownership, gradients and restoration."""
import copy
import torch
from torch import nn
from .method import Host
from ..r30_state_history.method import StateHost, readonly
from ..r19_model_only.method import equal


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        h = self.bn(self.features(torch.nn.functional.avg_pool2d(x, 16)))
        z = torch.nn.functional.interpolate(self.seg_head(h), size=x.shape[-2:], mode='nearest')
        return z, [h], h


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    m = Tiny(); x = torch.linspace(.01, .99, 3*512*512).reshape(1, 3, 512, 512)
    baseline = Host(None, 'C_CONT', 17, 'test', 'cpu', copy.deepcopy(m))
    reference = StateHost(None, 'C_CONT', 17, 'test', 'cpu', copy.deepcopy(m))
    h = Host(None, 'C_HEAD', 17, 'test', 'cpu', copy.deepcopy(m))
    frozen = {k: p.detach().clone() for k, p in h.native.model.named_parameters() if k not in h.native.names}
    for i in range(3):
        z, _ = baseline.step(x); rz, _ = reference.step(x)
        assert torch.equal(z, rz)
        h.step(x)
        assert h.native.steps == i+1 and h.native.counts['base_adam'] == i+1
        assert h.diagnostics()['head_gradient_norm'] > 0
        assert h.diagnostics()['head_drift'] > 0
    assert all(torch.equal(dict(h.native.model.named_parameters())[k], p) for k, p in frozen.items())
    assert h.native.base.param_groups[0]['lr'] == 1e-4 and h.native.base.param_groups[1]['lr'] == 1e-5
    s = h.snapshot(); readonly(h, x); assert equal(s, h.snapshot())
    z, _ = h.step(x); after = h.snapshot(); h.restore(s); rz, _ = h.step(x)
    assert torch.equal(z, rz) and equal(after['adam'], h.snapshot()['adam'])
    for q in (h, baseline, reference):
        q.check_frozen(True); q.close()
    print('PASS: head gradients/66 parameters, frozen encoder, optimizer ownership/clocks, baseline parity, read-only and replay state')


if __name__ == '__main__':
    main()
