from pathlib import Path
from collections import Counter
from core import read, save, unpack, components, choose, matched, stratum, audit_hook


def run(c):
    root, old = Path(c['root']), Path(c['sealed_root'])
    allowed = [old / 'private/scores.json'] + list((old / 'inputs').glob('*.bits'))
    opened = audit_hook(old, root, allowed, [root / 'code/evaluate.py', root / 'code/metric.py', c['asset_root']])
    for p in (old / 'scorer/panel.json', old / 'scorer/utility.private.json', root / 'code/evaluate.py'):
        try:
            p.read_text()
        except PermissionError:
            pass
        else:
            raise AssertionError('selection isolation test')
    frames = read(old / 'private/scores.json')
    assert len(frames) == 32
    out = []
    cycle_changed = cycle_total = 0
    for f in frames:
        i, k = f['case'], f['condition']
        s = unpack(old / 'inputs' / f'{i}_{k}_SOURCE.bits')
        b = unpack(old / 'inputs' / f'{i}_{k}_TOPO_SMOOTH.bits')
        groups = components(s, b)
        rows = [r for r in f['blocks'] if r['common']]
        for r in rows:
            mask = groups == r['block_id']
            assert int((s[1] != b[1])[mask].sum()) == r['oc_pixels'] == r['size']
            assert not (s[0] != b[0])[mask].any()
        cap = sum(r['oc_pixels'] for r in rows) // 10
        decisions = dict(ALL_COMMON=[r['block_id'] for r in rows])
        for name, score in [('CROP_POS10', 'CROP_Q'), ('LOGIT_POS10', 'CROP_LOGIT_Q'), ('GEOM_POS10', 'GEOMETRY_ONLY')]:
            decisions[name] = choose(rows, score, cap)
        for name, score in [('LOGIT_MATCH', 'CROP_LOGIT_Q'), ('GEOM_MATCH', 'GEOMETRY_ONLY'),
                            ('MARGIN_MATCH', 'RAW_MARGIN'), ('CYCLE_MATCH', 'CYCLE')]:
            decisions[name] = matched(rows, decisions['CROP_POS10'], score)
        counts = {}
        for name, ids in decisions.items():
            chosen = set(ids)
            counts[name] = dict(blocks=len(ids), pixels=sum(r['oc_pixels'] for r in rows if r['block_id'] in chosen))
            if name not in ('ALL_COMMON',):
                assert counts[name]['pixels'] <= cap
            if name.endswith('_MATCH'):
                assert counts[name] == counts['CROP_POS10']
        cycle_total += len(decisions['CROP_POS10'])
        cycle_changed += len(set(decisions['CROP_POS10']) - set(decisions['CYCLE_MATCH']))
        out.append(dict(case=i, condition=k, cap=cap, common_pixels=sum(r['oc_pixels'] for r in rows),
                        decisions=decisions, counts=counts))
    assert sum(len(f['decisions']['ALL_COMMON']) for f in out) == 2881
    save(root / 'private/DECISIONS.json', out)
    save(root / 'public/SELECTION_ACCESS.json', dict(status='PASS', frames=32, common_blocks=2881,
         gt_utility_evaluator_reads_denied=True, exact_size_direction_distance_quotas=True,
         original_endpoint_reads=64, model_image_weight_reads=0, labels_read=0,
         actual_opened_files=len(opened), budget_fraction=.1,
         cycle_changed_selected_blocks=cycle_changed, cycle_selected_blocks=cycle_total))
