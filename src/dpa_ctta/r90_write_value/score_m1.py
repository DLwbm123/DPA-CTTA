"""Terminal CPU-only window effects; the fixed gate never chooses a threshold."""
import os
import time
START = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gzip
import json
import runpy
import signal
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from dpa_ctta.r7_target_screen.runner import TargetReader
from dpa_ctta.r85_migration.process import alive

SEEDS = (20260907, 17011, 29009)
PROBES = (8, 32, 56, 80, 104, 128, 152, 176)
WIDTH = 131072


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')


def run(c):
    root = Path(c['output_root']); source = Path(c['source_root']); public = root/'public'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU window score cap')))
    signal.alarm(max(1, int(min(1740, c['deadline']-time.time()-180))))
    states = []
    for name, folder in [('qualify_worker0', 'qualify_0'), ('formal_worker0', 'formal_0'), ('formal_worker1', 'formal_1')]:
        p = read(root/'processes'/f'{name}.json'); s = read(root/'private'/folder/'RUN_STATE.json')
        assert p['exit_code'] == 0 and not p['active'] and not alive(p) and s['status'] == 'GPU_COMPLETE'
        states.append(s)
    totals = {k: sum(s[k] for s in states) for k in ('physical_updates', 'completed_physical_updates', 'physical_forwards', 'training_backwards')}
    assert totals == dict(physical_updates=1927, completed_physical_updates=1927, physical_forwards=19270, training_backwards=1927)
    module = runpy.run_path(str(Path(c['campaign_root'])/'m0.py'))
    post, measure, boundary, csvout = (module[k] for k in ('post', 'metrics', 'boundary', 'csvout'))
    names_metrics = module['METRICS']
    panels = [read(source/'scorer'/f'screen_o{o}.json') for o in (0, 1)]
    index = {r['image_sha256']: i for i, r in enumerate(panels[0])}; assert len(index) == 192
    domain_names = sorted({r['domain'] for r in panels[0]})
    domains = np.array([domain_names.index(r['domain']) for r in panels[0]])
    reader = TargetReader(c['target_root'], 256*1024**2, 'mask')
    truth, edges, distances = [], [], []
    for row in panels[0]:
        t = reader.read(row).numpy()[0].astype(bool); edge = np.stack([boundary(x) for x in t])
        truth.append(t); edges.append(edge); distances.append(np.stack([ndi.distance_transform_edt(~x).astype(np.float32) for x in edge]))
    values = np.zeros((192, 6, 2, 2, 2, len(names_metrics)))
    old = np.load(source/'scorer/values.private.npz')['values']
    def frame(file, i):
        raw = file.read(WIDTH); assert len(raw) == WIDTH
        masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(2, 2, 512, 512).astype(bool)
        return np.stack([np.stack([measure(m, truth[i], edges[i], distances[i]),
                                   measure(post(m), truth[i], edges[i], distances[i])]) for m in masks])
    for j in range(6):
        seed, order = SEEDS[j//2], j % 2; folder = root/'jobs'/f'main_s{seed}_o{order}'
        assert read(folder/'COMPLETE.json')['PRE_K_reference_exact']
        with gzip.open(folder/'predictions.bits.gz', 'rb') as f:
            for row in panels[order]:
                i = index[row['image_sha256']]; values[i, j] = frame(f, i)
                assert np.allclose(values[i, j, :, 0, :, 0], old[i, j, 0], atol=1e-10, rtol=0)
            assert not f.read(1)
    records, effects, anchors, unique_triggers, trigger_domains = [], [], [], set(), set()
    for j in range(6):
        seed, order = SEEDS[j//2], j % 2; folder = root/'jobs'/f'main_s{seed}_o{order}'
        signals = read(folder/'signals.json'); assert [s['arrival'] for s in signals] == list(PROBES)
        R, z = [], []
        for sig in signals:
            arrival = sig['arrival']; row = panels[order][arrival-1]
            if sig['trigger']:
                unique_triggers.add(row['image_sha256']); trigger_domains.add(domains[index[row['image_sha256']]])
            write, use = [], []
            with gzip.open(folder/f'probe_{arrival}.use.bits.gz', 'rb') as f:
                for visit in range(arrival, arrival+16):
                    i = index[panels[order][visit]['image_sha256']]
                    use.append(frame(f, i)); write.append(values[i, j])
                assert not f.read(1)
            delta = np.stack(write)-np.stack(use)
            r = float(delta[:, 1, 0, :, 0].mean())
            R.append(r); z.append(int(sig['trigger']))
            records.append(dict(seed=seed, order=order, **sig,
                                R_K_raw_pp=r, R_PRE_raw_pp=float(delta[:, 0, 0, :, 0].mean()),
                                R_K_post_pp=float(delta[:, 1, 1, :, 0].mean())))
            for output, phase in enumerate(('PRE', 'K')):
                for variant, kind in enumerate(('raw', 'post')):
                    for ch, channel in enumerate(('OD', 'OC')):
                        for mi, metric in enumerate(names_metrics):
                            v = delta[:, output, variant, ch, mi]; valid = v[np.isfinite(v)]
                            effects.append(dict(seed=seed, order=order, arrival=arrival, phase=phase, variant=kind,
                                 channel=channel, metric=metric, WRITE_minus_USE_mean=float(valid.mean()) if len(valid) else None,
                                 n_window=16, n_NA=int((~np.isfinite(v)).sum())))
        R, z = np.array(R), np.array(z)
        rule = float((-z*R).mean()); random = float(z.mean()*(-R).mean())
        anchors.append(dict(seed=seed, order=order, probes=8, trigger_count=int(z.sum()),
                            G_rule_pp=rule, G_random_pp=random, G_identify_pp=rule-random,
                            mean_R_pp=float(R.mean()), harmful_writes=int((R < 0).sum())))
    reader.after_check()
    csvout(public/'ALL_PROBES.csv', records); csvout(public/'ALL_WINDOW_EFFECTS.csv', effects)
    csvout(public/'ANCHOR_GAINS.csv', anchors)
    gains = {}
    for metric in ('G_rule_pp', 'G_identify_pp'):
        a = np.array([x[metric] for x in anchors])
        ok = a.mean() > 0 and (a > 0).sum() >= 5
        ok = ok and all(a[o::2].mean() > 0 for o in (0, 1)) and all(a[s*2:s*2+2].mean() > 0 for s in range(3))
        gains[metric] = dict(mean_pp=float(a.mean()), positive_anchors=int((a > 0).sum()),
                             orders=[float(a[o::2].mean()) for o in (0, 1)],
                             seeds=[float(a[s*2:s*2+2].mean()) for s in range(3)], passed=bool(ok))
    count = sum(a['trigger_count'] for a in anchors)
    coverage = count >= 12 and all(1 <= a['trigger_count'] <= 7 for a in anchors)
    coverage = coverage and len(unique_triggers) >= 6 and len(trigger_domains) >= 3
    passed = coverage and all(g['passed'] for g in gains.values())
    save(public/'M1_GATE.json', dict(status='PASS' if passed else 'NOT_PASSED', gains=gains,
        total_triggers=count, trigger_counts=[a['trigger_count'] for a in anchors],
        unique_trigger_contents=len(unique_triggers), trigger_domains=len(trigger_domains), trigger_coverage_passed=bool(coverage),
        authorize_M2=bool(passed), independent_validation=False, primary='raw K double-head arrival-mean 16-window',
        comparison='within-anchor same-frequency random expectation; no independent-probe CI'))
    save(public/'COMPLETION.json', dict(status='COMPLETE', **totals, original_main_trajectories=6,
        probes=48, future_window=16, extra_branch_updates=768, raw_Dice_parity_checks=2304,
        online_label_reads=0, source_image_reads=0, terminal_label_contents=192,
        all_registered_GPU_workers_exit0_absent_before_labels=True, CPU_score_body_seconds=time.time()-START,
        prior_attempt_updates=c.get('prior_updates', 0), prior_attempt_forwards=c.get('prior_forwards', 0),
        cumulative_R90_updates=totals['physical_updates']+c.get('prior_updates', 0),
        cumulative_R90_forwards=totals['physical_forwards']+c.get('prior_forwards', 0),
        shared_started=c['started'], shared_deadline=c['deadline']))
    print(json.dumps(read(public/'M1_GATE.json')), flush=True)


if __name__ == '__main__':
    run(read(Path(os.environ['CONFIG'])))
