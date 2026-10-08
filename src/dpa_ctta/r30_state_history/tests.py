"""Small generated-input checks for state isolation, reset clocks and paired futures."""
import copy
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from .method import StateHost, FrozenHost, DelayHost, config, readonly, set_native_rng
from .report import mask_at
from ..r24_c_context.method import Host as Reference
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal, clone


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    model = Tiny(); x = torch.linspace(.01, .99, 3*512*512).reshape(1, 3, 512, 512)
    checks = []
    for arm, name in [('C_CONT', 'C'), ('ANCHOR', 'ANCHOR')]:
        h = StateHost(None, arm, 17, 'test', device='cpu', model=copy.deepcopy(model))
        r = Reference(None, config(name), 17, 'test', device='cpu', model=copy.deepcopy(model))
        for i in range(3):
            h.prepare_arrival(); z, _ = h.step(x.roll(i, -1)); rz, _ = r.step(x.roll(i, -1))
            assert torch.equal(z, rz)
            for key in ('parameters', 'adam', 'native_rng', 'adapter'):
                assert equal(h.snapshot()[key], r.snapshot()[key]), (arm, key)
        before = h.snapshot(); readonly(h, x); assert equal(before, h.snapshot())
        h.check_frozen(True); h.close(); r.close()
    checks.append('unchanged C/ANCHOR exact parity and readonly isolation')
    for arm in ('C_EPISODIC', 'C_OPT32', 'C_PARAM32', 'C_BOTH32'):
        h = StateHost(None, arm, 17, 'test', device='cpu', model=copy.deepcopy(model))
        for i in range(32):
            h.prepare_arrival(); h.step(x.roll(i, -1))
        before = h.snapshot(); counts = h.native.counts.copy()
        h.prepare_arrival(); after = h.snapshot()
        assert equal(before['native_rng'], after['native_rng']) and equal(before['rng'], after['rng'])
        assert h.visits == 32 and h.native.counts == counts
        assert equal(before['buffers'], after['buffers']) and equal(before['modes'], after['modes'])
        if arm != 'C_PARAM32':
            assert not h.native.base.state and h.native.steps == 0
        else:
            assert equal(before['adam'], after['adam']) and h.native.steps == 32
        if arm != 'C_OPT32':
            assert equal(after['parameters'], h.source_affine)
        else:
            assert equal(before['parameters'], after['parameters'])
        h.step(x)
        assert h.visits == 33 and h.native.steps == (33 if arm == 'C_PARAM32' else 1)
        assert h.native.counts['base_adam'] == 33
        h.check_frozen(True); h.close()
    checks.append('reset before arrival 33; parameter/Adam isolation; global clocks and RNG preserved')
    for arm in ('S_SOURCE', 'S_BATCH'):
        h = FrozenHost(None, arm, 17, device='cpu', model=copy.deepcopy(model))
        h.step(x); h.step(x.flip(-1)); h.check_frozen()
        assert not any(p.requires_grad for p in h.model.parameters())
        assert h.model.up3.bn.track_running_stats == (arm == 'S_SOURCE')
    checks.append('native and batch-normalized frozen states unchanged')
    h = DelayHost(None, config('GREEDY'), 17, 'test', device='cpu', model=copy.deepcopy(model))
    h.step(x); before = h.snapshot(); schedule = []; outputs = {}
    for action in ('FULL', 'ZERO', 'HALF'):
        h.restore(before); out = []
        for t in range(4):
            if action == 'FULL':
                schedule.append(clone(h.native.rng))
            else:
                set_native_rng(h, schedule[t])
            if t == 0:
                n = h.native.counts['base_adam']; z = h.first(x.roll(t+1, -1), action)
                assert h.native.counts['base_adam']-n == (action != 'ZERO')
            else:
                z, _ = h.step(x.roll(t+1, -1))
            out.append(z)
        assert h.visits == before['visits']+4
        assert h.native.steps == before['steps']+3+(action != 'ZERO')
        outputs[action] = out
    h.restore(before)
    for t in range(4):
        z, _ = h.step(x.roll(t+1, -1)); assert torch.equal(z, outputs['FULL'][t])
    h.check_frozen(True); h.close()
    checks.append('committed one-step interventions; future RNG coupling; native FULL replay')
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)/'bits'; arr = np.zeros((2, 2, 512, 512), bool); arr[1, 1] = True
        p.write_bytes(np.packbits(arr).tobytes())
        with p.open('rb') as f:
            assert np.array_equal(mask_at(f, 1), arr[1])
            try:
                mask_at(f, 2)
            except ValueError:
                pass
            else:
                raise AssertionError('truncated masks accepted')
    checks.append('mask indexing and truncated-file rejection')
    from . import report
    from ..r10_12h_core.run import save
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for folder in ('scorer', 'public', 'target'):
            (root/folder).mkdir()
        rows = [dict(image_sha256=str(i), domain='D'+str((i//4) % 4)) for i in range(96)]
        save(root/'scorer/SPLIT.private.json', {str(i): 'SEARCH' for i in range(96)})
        windows = [dict(index=i, visit=i+1, available_future=64) for i in range(16)]
        eligibility = {str(o): [{str(H): dict(score_count=H or 1, eligible=True, domains={'D': H or 1})
                                for H in (0, 8, 32, 64)} for _ in range(16)] for o in (0, 1)}
        save(root/'scorer/WINDOWS.json', eligibility)
        jobs = []
        for seed in (1, 2, 3):
            for order in (0, 1):
                j = dict(id=f'{seed}_{order}', seed=seed, order=order); jobs.append(j)
                save(root/'scorer'/f'FULL_o{order}.json', rows)
                dest = root/'target'/j['id']; dest.mkdir()
                import json
                (dest/'windows.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in windows))
                for i in range(16):
                    for a in ('FULL', 'ZERO', 'HALF'):
                        (dest/f'w{i}_{a}.bits').touch()
        def fake_mask(handle, offset):
            action = Path(handle.name).stem.split('_')[-1]
            value = .6 if action == 'FULL' else float(offset % 2 == (action == 'ZERO'))
            return value
        class Reader:
            def __init__(self, *_): pass
            def read(self, _): return None
        with patch.object(report, 'TargetReader', Reader), patch.object(report, 'mask_at', fake_mask), \
             patch.object(report, 'hard_metrics', lambda m, _: [dict(dice=m), dict(dice=m)]):
            result = report.score_horizons(dict(output_root=tmp, target_root=tmp, a_jobs=jobs, seeds=[1, 2, 3]), lambda: None)
        assert abs(result['primary']['oracle_pp']) < 1e-9
        assert abs(result['primary']['zero_pp']+10) < 1e-9
        assert abs(result['summaries'][0]['oracle_pp']-40) < 1e-9
        assert not result['primary']['allocation_signal']
    checks.append('horizon oracle selects whole branches, excludes current image, and preserves denominators')
    for s in checks:
        print('PASS:', s)


if __name__ == '__main__':
    main()
