"""Eight fixed short trajectories; score SEARCH only after all workers retire."""
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch
from .method import Host
from ..r20_model_only_search import runtime as base
from ..r10_12h_core.run import read, save, sha
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader

ID = 'R21_REGIONAL_PROTOTYPE_QUICK'
WIDTH = 65536


def online(c, guard, permit, job):
    root = Path(c['output_root'])
    rows = read(root/'private'/f'SCREEN_o{job["order"]}.json')
    if rows_sha(rows) != c['manifests']['SCREEN'][job['order']]['sha256']:
        raise ValueError('changed manifest')
    h = Host(base.weights(c), job['candidate'], job['seed'], sha(job))
    base.attach(guard.meter, h)
    reader = TargetReader(c['target_root'], 256*1024**2, 'image')
    dest = root/'target'/job['id']; dest.mkdir()
    try:
        with (dest/'predictions.bits').open('xb') as bits, (dest/'visits.jsonl').open('x') as trace:
            for i, row in enumerate(rows):
                guard(); x = base.image(reader, row, permit, guard); start = time.perf_counter()
                z, t = h.step(x); torch.cuda.synchronize()
                arrays = [z.sigmoid() >= .5, h.last_before >= .5, h.last_after >= .5, h.last_low_variance]
                for a in arrays: bits.write(np.packbits(a.numpy()).tobytes())
                t.update(visit=i+1, seconds=time.perf_counter()-start)
                trace.write(json.dumps(t, allow_nan=False)+'\n'); trace.flush()
        h.check_frozen(True)
        result = dict(status='COMPLETE', arrivals=len(rows), prediction_bytes=(dest/'predictions.bits').stat().st_size, label_reads=0, source_reads=0)
        assert result['prediction_bytes'] == len(rows)*4*WIDTH
        save(dest/'complete.json', result)
        return result
    finally:
        h.close()


def score(c):
    root = Path(c['output_root'])
    if any(read(p).get('active') for p in (root/'processes').glob('*.json')):
        raise ValueError('online retirement barrier')
    reader = TargetReader(c['target_root'], 256*1024**2, 'mask')
    output = []
    for job in c['jobs']:
        rows = read(root/'scorer'/f'SCREEN_o{job["order"]}.json')
        dest = root/'target'/job['id']; receipt = read(dest/'complete.json')
        assert receipt['arrivals'] == len(rows) == 192
        traces = [json.loads(x) for x in (dest/'visits.jsonl').read_text().splitlines()]
        with (dest/'predictions.bits').open('rb') as bits:
            for row, trace in zip(rows, traces):
                raw = bits.read(WIDTH*4)
                masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(4, 2, 512, 512).astype(bool)
                gt = reader.read(row).numpy()[0].astype(bool)
                pred, before, after, stable = masks
                dice = [(2*(pred[k]&gt[k]).sum()+1e-6)/(pred[k].sum()+gt[k].sum()+1e-6)*100 for k in range(2)]
                corrected = (before != gt) & (after == gt); damaged = (before == gt) & (after != gt)
                output.append(dict(arm=job['candidate']['id'], order=job['order'], content=row['image_sha256'], domain=row['domain'], dice=dice, corrected=int(corrected.sum()), damaged=int(damaged.sum()), stable_corrected=int((corrected&stable).sum()), stable_damaged=int((damaged&stable).sum()), **trace))
            assert not bits.read(1)
    # Private identities stay local. Public summaries contain no patient paths or hashes.
    save(root/'scorer/results.private.json', output)
    summaries = {}
    for arm in ('C', 'CW', 'CR', 'CWR'):
        rr = [r for r in output if r['arm'] == arm]
        cells = {f'{o}/{d}/{k}': statistics.mean(r['dice'][k] for r in rr if r['order']==o and r['domain']==d) for o in (0, 1) for d in sorted({r['domain'] for r in rr}) for k in (0, 1)}
        summaries[arm] = dict(macro_Dice_percent=statistics.mean(cells.values()), cells=cells, order_macro=[statistics.mean(v for k,v in cells.items() if k.startswith(f'{o}/')) for o in (0,1)], imageweighted_Dice_percent=statistics.mean(statistics.mean(r['dice']) for r in rr), arrivals=len(rr), seconds_per_image=statistics.mean(r['seconds'] for r in rr), **{k:sum(r[k] for r in rr) for k in ('corrected','damaged','stable_corrected','stable_damaged')}, active_arrivals=sum(r['diagnostics']['correction_regions']>0 for r in rr), mean_coverage=statistics.mean(r['diagnostics']['correction_coverage'] for r in rr))
    comparisons = {}
    for a, b in [('CW','C'), ('CR','C'), ('CWR','CW'), ('CWR','C')]:
        aa,bb = summaries[a],summaries[b]
        comparisons[f'{a}-{b}'] = dict(macro_delta_pp=aa['macro_Dice_percent']-bb['macro_Dice_percent'], order_delta_pp=[x-y for x,y in zip(aa['order_macro'],bb['order_macro'])], worst_domain_channel_order_pp=min(aa['cells'][k]-bb['cells'][k] for k in aa['cells']), time_ratio=aa['seconds_per_image']/bb['seconds_per_image'])
    tested = comparisons['CWR-CW']; full = summaries['CWR']
    promising = tested['macro_delta_pp'] >= .2 and min(tested['order_delta_pp']) > 0 and tested['worst_domain_channel_order_pp'] >= -2 and full['corrected'] > full['damaged'] and comparisons['CWR-C']['macro_delta_pp'] > 0
    result = dict(status='COMPLETE', summaries=summaries, comparisons=comparisons, pilot_signal='PROMISING_NEEDS_CONFIRMATION' if promising else 'NOT_ESTABLISHED', seed=20260907, unique_images=192, orders=2, trajectories=8, online_label_reads=0, historical_exposure=True, patient_linkage='UNKNOWN', interpretation='One-seed short SEARCH pilot; no independent generalization claim; pseudo-target correction differs from final prediction gain.')
    save(root/'public/RESULTS.json', result)
    return result


def supervise():
    c=base.config(); root=Path(c['output_root']); start=time.time()
    save(root/'RUN_STATE.json', dict(status='RUNNING', started=start))
    completed=[]
    for order in (0,1):
        jobs=[j for j in c['jobs'] if j['order']==order]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(base.run_task,c,'online_'+j['id'],gpu,600,j) for j,gpu in zip(jobs,c['gpu_assignments'])]
            completed.extend(f.result() for f in futures)
        if any(r['status'] != 'COMPLETE' for r in completed):
            save(root/'RUN_STATE.json',dict(status='INCOMPLETE',reason='failed fixed trajectory; no retry or expansion',completed=completed));return
    save(root/'stages/ALL_RETIRED.json',dict(status='ALL_WORKERS_RETIRED',at=time.time()))
    result=score(c)
    result.update(wall_seconds=time.time()-start,gpu_worker_seconds=sum(r['cost']['gpu_seconds'] for r in completed),code_sha=c['code_sha'])
    save(root/'public/RESULTS.json',result)
    save(root/'RUN_STATE.json',dict(status='COMPLETE',ended=time.time(),delivery='PENDING_GITHUB'))


def main():
    base.ID=ID; base.Host=Host; base.online=online
    torch.set_num_threads(2)
    if os.environ['RUN_MODE']=='worker':
        r=base.worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    else:
        try:supervise()
        except BaseException as e:
            c=base.config();save(Path(c['output_root'])/'RUN_STATE.json',dict(status='INCOMPLETE',reason=str(e)));raise


if __name__=='__main__':main()
