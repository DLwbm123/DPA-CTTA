"""CPU-only descriptive analysis of already recorded vectors; see PROTOCOL.md."""
import os
import time

START = float(os.environ.get('ANALYSIS_STARTED', time.time()))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(1)
SOURCE = Path(os.environ['INPUT_RUN_ROOT'])
OUT = Path(os.environ['ANALYSIS_OUTPUT_ROOT'])
CELLS = ('VV', 'VB', 'BV', 'BB')
CAP = 64 * 1024**2 - 2 * 1024**2


def guard():
    if time.time() - START >= 570:
        raise RuntimeError('original analysis clock exhausted')
    if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) >= CAP:
        raise RuntimeError('output cap; recovery reserve preserved')


def norm(x):
    return float(torch.linalg.vector_norm(x))


def ratio(a, b):
    return a / b if b != 0 else None


def cosine(a, b):
    return ratio(float(torch.dot(a, b)), norm(a) * norm(b))


def stage(name):
    if name.startswith('res.layer'):
        return name.split('.')[1]
    if name.startswith('res.bn1.'):
        return 'stem'
    if name.startswith('up'):
        return name.split('.')[0]
    if name.startswith('bnout.'):
        return 'bnout'
    raise ValueError('unregistered architecture stage')


def self_check():
    assert cosine(torch.zeros(2), torch.ones(2)) is None
    assert cosine(torch.tensor([1., 0.]), torch.tensor([0., 1.])) == 0
    a, b, c = torch.tensor([1., 2.]), torch.tensor([2., 0.]), torch.tensor([0., 3.])
    assert norm((a+b+c) - (a+b) - (a+c) + a) == 0
    assert ratio(1, 0) is None


def run():
    self_check()
    meta = list(csv.DictReader((SOURCE/'public/SAME_STATE_SCORES.csv').open()))
    domain = {}
    for row in meta:
        key = (int(row['seed']), int(row['order']), int(row['probe_number']))
        if key in domain:
            assert domain[key] == row['domain']
        domain[key] = row['domain']
    assert len(domain) == 48
    position = {}
    for seed in (20260907, 17011, 29009):
        for order in (0, 1):
            for d in ('D0', 'D1', 'D2', 'D3'):
                keys = sorted(k for k, v in domain.items() if k[:2] == (seed, order) and v == d)
                assert len(keys) == 2
                position.update(zip(keys, ('p25', 'p75')))

    layout = None
    groups = {}
    records = []
    for seed in (20260907, 17011, 29009):
        for order in (0, 1):
            guard()
            folder = SOURCE/'jobs'/f'replay_VV_s{seed}_o{order}'
            snapshot = torch.load(folder/'terminal.private.pt', map_location='cpu', weights_only=False)
            current = [(n, tuple(t.shape), t.numel()) for n, t in snapshot['parameters'].items()]
            assert len(current) == 82 and sum(x[2] for x in current) == 19136
            if layout is None:
                layout = current
                offset = 0
                inventory = []
                for name, shape, size in layout:
                    kind = 'gamma' if name.endswith('.weight') else 'beta'
                    assert name.endswith(('.weight', '.bias'))
                    module = name.rsplit('.', 1)[0]
                    architecture_stage = stage(name)
                    inventory.append(dict(parameter=name, shape=list(shape), start=offset, end=offset+size,
                                          module=module, stage=architecture_stage, kind=kind))
                    for family, label in [('global', 'BN'), ('module', module), ('stage', architecture_stage)]:
                        for part in ('all', kind):
                            groups.setdefault((family, label, part), []).extend(range(offset, offset+size))
                    offset += size
                assert len(groups) == 156
                for family in ('module', 'stage'):
                    indices = [i for (f, _, part), ids in groups.items() if f == family and part == 'all' for i in ids]
                    assert sorted(indices) == list(range(19136))
                (OUT/'PARAMETER_LAYOUT.json').write_text(json.dumps(inventory, indent=2)+'\n')
                groups = {k: torch.tensor(v, dtype=torch.long) for k, v in groups.items()}
            else:
                assert current == layout
            del snapshot
            files = sorted(folder.glob('probe_*.private.pt'), key=lambda p: int(p.name.split('_')[1].split('.')[0]))
            assert len(files) == 8
            for number, file in enumerate(files, 1):
                guard()
                key = (seed, order, number)
                raw = torch.load(file, map_location='cpu', weights_only=False)
                assert set(raw) == {'gradients', 'Adam_deltas', 'component_gradients'}
                for values in raw.values():
                    assert all(v.shape == (19136,) and v.dtype == torch.float32 and torch.isfinite(v).all() for v in values.values())
                v = {k: {c: x.double() for c, x in values.items()} for k, values in raw.items()}
                g, delta, components = v['gradients'], v['Adam_deltas'], v['component_gradients']
                channel = {c: [components['B_'+c[0]+str(ch)]+.1*components['S_'+c[1]+str(ch)] for ch in (0, 1)] for c in CELLS}
                for c in CELLS:
                    assert torch.allclose(sum(channel[c]), g[c], atol=2e-6, rtol=1e-4)
                rd = delta['BB']-delta['BV']-delta['VB']+delta['VV']
                rg = g['BB']-g['BV']-g['VB']+g['VV']
                for (family, group, part), ids in groups.items():
                    dr, gr = rd[ids], rg[ids]
                    residual_scale = sum(norm(delta[c][ids]) for c in CELLS)
                    for c in CELLS:
                        x, ref = delta[c][ids], delta['VV'][ids]
                        difference = delta[c]-delta['VV']
                        local = difference[ids]
                        od, oc = [t[ids] for t in channel[c]]
                        records.append(dict(seed=seed, order=order, domain=domain[key], position=position[key],
                            probe_number=number, family=family, group=group, part=part, cell=c, n_parameters=len(ids),
                            delta_l2=norm(x), delta_RMS=norm(x)/math.sqrt(len(ids)),
                            delta_energy_share=ratio(float(x.square().sum()), float(delta[c].square().sum())),
                            difference_l2=norm(local), difference_RMS=norm(local)/math.sqrt(len(ids)),
                            difference_energy_share=ratio(float(local.square().sum()), float(difference.square().sum())),
                            local_delta_cosine_VV=cosine(x, ref), local_delta_norm_ratio_VV=ratio(norm(x), norm(ref)),
                            VV_delta_denominator_l2=norm(ref), difference_denominator_energy=float(difference.square().sum()),
                            channel_cosine=cosine(od, oc), OD_dot_VV_delta=float(torch.dot(od, ref)),
                            OC_dot_VV_delta=float(torch.dot(oc, ref)), residual_l2=norm(dr),
                            residual_RMS=norm(dr)/math.sqrt(len(ids)), residual_scale_denominator=residual_scale,
                            residual_relative_scale=ratio(norm(dr), residual_scale),
                            residual_energy_share=ratio(float(dr.square().sum()), float(rd.square().sum())),
                            gradient_factorial_l2=norm(gr), quantization_bound='NOT_IDENTIFIABLE'))
                # Check full module energy partition independently for all recorded cells.
                module_ids = [ids for (f, _, part), ids in groups.items() if f == 'module' and part == 'all']
                for c in CELLS:
                    assert math.isclose(sum(float(delta[c][ids].square().sum()) for ids in module_ids),
                                        float(delta[c].square().sum()), rel_tol=1e-12, abs_tol=1e-30)
    assert len(records) == 48*156*4
    write_csv('ALL_GROUP_PROBES.csv', records)
    metrics = ('difference_RMS', 'difference_energy_share', 'local_delta_cosine_VV', 'local_delta_norm_ratio_VV',
               'channel_cosine', 'OD_dot_VV_delta', 'OC_dot_VV_delta', 'residual_relative_scale')
    summaries = {}
    for row in records:
        for axis in ('ALL', 'seed', 'order', 'domain', 'position'):
            value = 'ALL' if axis == 'ALL' else str(row[axis])
            key = (row['family'], row['group'], row['part'], row['cell'], axis, value)
            acc = summaries.setdefault(key, {m: [] for m in metrics})
            for m in metrics:
                acc[m].append(row[m])
    rows = []
    for key, acc in summaries.items():
        row = dict(zip(('family', 'group', 'part', 'cell', 'axis', 'axis_value'), key))
        for metric, values in acc.items():
            values_valid = [v for v in values if v is not None]
            row.update({metric+'_NA': len(values)-len(values_valid), metric+'_n': len(values_valid),
                        metric+'_mean': sum(values_valid)/len(values_valid) if values_valid else None,
                        metric+'_min': min(values_valid) if values_valid else None,
                        metric+'_max': max(values_valid) if values_valid else None,
                        metric+'_negative': sum(v < 0 for v in values_valid)})
        rows.append(row)
    write_csv('ALL_GROUP_STRATA.csv', rows)
    guard()
    receipt = dict(status='COMPLETE',started_epoch=START, ended_epoch=time.time(), original_elapsed_seconds=time.time()-START,
                   probes=48, modules=41, stages=10, parameter_tensors=82, vector_dimension=19136, groups=156,
                   detailed_rows=len(records), stratum_rows=len(rows), new_model_calls=0, new_GPU_updates=0,
                   new_image_reads=0, new_label_reads=0, all_6_parameter_layouts_equal=True, self_checks='PASS',
                   precision='float64 summaries of float32 saved vectors; inbound-parameter quantization bound not identifiable',
                   storage_cap_bytes=64*1024**2, timeout_seconds=600,
                   artifact_bytes_before_receipt=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()))
    (OUT/'COMPLETION.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


def write_csv(name, rows):
    with (OUT/name).open('x') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({k: 'NA' if v is None else v for k, v in row.items()})
            if i % 128 == 0:
                file.flush()
                guard()


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        (OUT/'FAILURE.json').write_text(json.dumps(dict(error=type(exc).__name__, reason=str(exc),
                                                      started_epoch=START, ended_epoch=time.time()))+'\n')
        raise
