"""Small generated-input check of fallback, correction direction and real host parity."""
import copy
import torch
from .method import correction, regional_loss, Host
from ..r20_model_only_search.method import Host as Base
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2)
    q = torch.full((1, 2, 16, 16), .6)
    history = torch.full_like(q, .02)
    unchanged, a, n = correction(q, history, False)
    assert torch.equal(unchanged, q) and n == 0 and not a.any()
    target, a, n = correction(q, history, True)
    assert n == 1 and (target < .5).all() and (target[:, 1] <= target[:, 0]).all()
    z = torch.zeros_like(q, requires_grad=True)
    regional_loss(z, target, torch.ones_like(z), a).backward()
    assert torch.isfinite(z.grad).all() and (z.grad > 0).all()
    torch.manual_seed(7); model = Tiny(); x = torch.rand(1, 3, 512, 512)
    for weighted in (False, True):
        c = dict(family='combo', host='C', params={}, components={})
        if weighted: c['components']['W'] = dict(variance_temperature=.1, boundary_boost=2)
        reference = Base(None, c, 20260907, 'check', device='cpu', model=copy.deepcopy(model))
        candidate = copy.deepcopy(c)
        candidate['components']['P'] = dict(prototype_mix=.5, cosine_temperature=.1)
        h = Host(None, candidate, 20260907, 'check', device='cpu', model=copy.deepcopy(model))
        za, _ = reference.step(x); zb, _ = h.step(x)
        assert torch.equal(za, zb)
        for k in ('parameters', 'adam', 'native_rng'):
            assert equal(reference.snapshot()[k], h.snapshot()[k]), k
        h.check_frozen(True); h.close(); reference.close()
    print('PASS: empty-history C/CW exact parity; supported correction changes direction; finite gradient; frozen weights')


if __name__ == '__main__': main()
