"""CPU synthetic check: C parity, conditional semantics, gradients and resume."""
import copy
import numpy as np
import torch
from .method import Host, ContextAdapter, descriptor
from ..r20_model_only_search.method import Host as Base
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = Tiny()
    x = torch.linspace(.01, .99, 3 * 512 * 512).reshape(1, 3, 512, 512)
    def cfg(mode):
        return dict(id=mode, family='context_adapter' if mode not in ('C', 'C_LR15') else 'control',
                    host='C', params={}, components={})
    def host(mode):
        return Host(None, cfg(mode), 20260907, 'test', device='cpu', model=copy.deepcopy(model))
    h = host('C')
    ref = Base(None, cfg('C'), 20260907, 'test', device='cpu', model=copy.deepcopy(model))
    for _ in range(3):
        za, _ = h.step(x); zb, _ = ref.step(x)
        assert torch.equal(za, zb)
        for k in ('parameters', 'adam', 'native_rng'):
            assert equal(h.snapshot()[k], ref.snapshot()[k]), k
    h.close(); ref.close()
    d = descriptor(torch.rand(1, 4, 8, 8), torch.zeros(1, 4), torch.ones(1, 4))
    assert d.shape == (1, 8) and d.abs().max() <= 1 and not d.requires_grad
    a = ContextAdapter(4, 4, 'cpu', 1); a.context = d
    f = torch.rand(1, 4, 8, 8)
    assert torch.equal(a(f), f), 'zero-U exact identity'
    for mode in ('STATIC', 'VIEW', 'ANCHOR', 'C_LR15'):
        h = host(mode)
        for i in range(3):
            z, t = h.step(x.roll(i, -1))
            assert torch.isfinite(z).all()
            assert t['diagnostics']['BN_lr'] == (1.5e-4 if mode == 'C_LR15' else 1e-4)
            if mode == 'ANCHOR':
                assert t['diagnostics']['strong_used_context_distance'] == 0
            if mode == 'VIEW':
                assert t['diagnostics']['strong_used_context_distance'] == t['diagnostics']['strong_context_distance']
            if mode != 'C_LR15':
                assert t['diagnostics']['adapter_U_gradient'] > 0
                assert h.adapter.gate.weight.grad is not None
                if i:
                    assert h.adapter.V.weight.grad.norm() > 0
                    if mode != 'STATIC':
                        assert h.adapter.gate.weight.grad.norm() > 0
        s = h.snapshot(); za, ta = h.step(x.flip(-1)); end = h.snapshot()
        h.restore(s); zb, tb = h.step(x.flip(-1))
        assert torch.equal(za, zb) and equal(end, h.snapshot()), mode
        assert ta['diagnostics'] == tb['diagnostics']
        h.check_frozen(True); h.close()
    from .run import hard_metrics, stage_results
    empty = np.zeros((2, 4, 4), dtype=bool)
    assert all(m['dice'] == 1 for m in hard_metrics(empty, torch.zeros(1, 2, 4, 4)))
    assert all(m['dice'] == 0 for m in hard_metrics(~empty, torch.zeros(1, 2, 4, 4)))
    rows = []
    for mode, value in [('C', .5), ('C_LR15', .501), ('STATIC', .502), ('VIEW', .502), ('ANCHOR', .51)]:
        for order in (0, 1):
            for i in range(4):
                rows.append(dict(condition=mode, seed=20260907, order=order, content=str(i), domain=str(i//2),
                                 role='SEARCH', seconds=1., metrics=[dict(channel=k, dice=value) for k in ('OD', 'OC')]))
    assert stage_results(rows, 20260907)['advance']
    for row in rows:
        if row['condition'] == 'ANCHOR' and row['order'] == 1:
            row['metrics'] = [dict(channel=k, dice=.49) for k in ('OD', 'OC')]
    assert not stage_results(rows, 20260907)['advance'], 'one good order must not hide a bad one'
    print('PASS: C exact output/Adam/RNG parity; fixed LR; bounded detached descriptor; zero-U identity; anchoring; connected gradients; exact continuation; frozen backbone')


if __name__ == '__main__':
    main()
