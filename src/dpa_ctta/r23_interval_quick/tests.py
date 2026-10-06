"""Generated-input gradient semantics and exact CW host parity."""
import copy
import torch
from .method import conflict_loss, Host
from ..r20_model_only_search.method import Host as Base
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2)
    q = torch.tensor([.2, .2, .2, .2]); source = torch.tensor([.8, .8, .8, .3])
    for mode in ('MASK', 'INTERVAL'):
        z = torch.logit(torch.tensor([.1, .5, .9, .5])).requires_grad_()
        loss, target, conflict, outside = conflict_loss(z, q, source, torch.ones_like(q), mode)
        loss.backward()
        expected = torch.tensor([-.1, 0., .1, .3])/4 if mode == 'INTERVAL' else torch.tensor([0., 0., 0., .3])/4
        assert torch.allclose(z.grad, expected, atol=1e-7), z.grad
        assert not target.requires_grad and conflict.tolist() == [True, True, True, False]
        assert outside.tolist() == [True, False, True, False]
        z = torch.zeros(4, requires_grad=True)
        a, _, _, _ = conflict_loss(z, q, q, torch.ones_like(q), mode)
        assert torch.equal(a, torch.nn.functional.binary_cross_entropy_with_logits(z, q))
    torch.manual_seed(17); model = Tiny(); x = torch.rand(1,3,512,512)
    cfg = dict(id='CW', family='combo', host='C', params={}, components={'W':dict(variance_temperature=.1, boundary_boost=2)})
    ref = Base(None,cfg,20260907,'check',device='cpu',model=copy.deepcopy(model))
    h = Host(None,cfg,20260907,'check',device='cpu',model=copy.deepcopy(model))
    for _ in range(2):
        za,_ = h.step(x); zb,_ = ref.step(x)
        assert torch.equal(za,zb)
        for k in ('parameters','adam','native_rng'): assert equal(h.snapshot()[k],ref.snapshot()[k])
    h.close();ref.close()
    for mode in ('MASK','INTERVAL'):
        cfg['id']=mode
        h=Host(None,cfg,20260907,'check',device='cpu',model=copy.deepcopy(model))
        z,t=h.step(x)
        assert torch.isfinite(z).all() and 0 <= t['diagnostics']['conflict_coverage'] <= 1
        assert all(not p.requires_grad for p in h.f0.parameters())
        h.check_frozen(True);h.close()
    print('PASS: interval gradient direction and zero interior; mask zero gradient; agreement parity; CW output/Adam/RNG parity; frozen source')


if __name__ == '__main__': main()
