"""R67 fixed-image mask-response diagnostic; no target labels or optimizer."""
import json
import os
from pathlib import Path
import time
import numpy as np
from ..r51_same_state.run import read, save, run_child
from ..r51_same_state.score import interval, csvout
from .run import worker
from .response import response, self_check


def produce(features, masks, index):
    return masks[None], response(features, masks[:,2::4,2::4])


def aggregate(c):
    root = Path(c['output_root']); state = read(root/'RUN_STATE.json'); process = read(root/'processes/worker.json')
    if state['status'] != 'GPU_COMPLETE' or state['formal_pairs'] != 96 or process['active'] or process['exit_code'] != 0:
        raise ValueError('successful worker retirement required')
    traces = read(root/'private/pairs.json'); domains = np.array(read(root/'private/domains.json'))
    if [r['pair'] for r in traces] != list(range(96)) or len(domains) != 48 or len(set(domains)) != 4:
        raise ValueError('fixed coverage')
    ratios = np.full((48,2,2,2), np.nan); coverage = np.zeros((48,2,2,2,3))
    for r in traces:
        order, i = divmod(r['pair'],48)
        for arm,name in enumerate(('IMAGE','MASK_ONLY')):
            ratios[i,order,arm] = [np.nan if v is None else v for v in r['arms'][name]['spread_ratio']]
            for shift in range(3):
                for ch in range(2):
                    count = r['arms'][name]['counts'][shift][ch]
                    coverage[i,order,arm,ch,shift] = count['accepted']/count['boundary'] if count['boundary'] else 0
    complete = bool(np.isfinite(ratios).all())
    cells = []
    for order in (0,1):
        for domain in sorted(set(domains)):
            for ch,channel in enumerate(('OD','OC')):
                local = ratios[domains==domain,order,:,ch]
                cells.append(dict(order=order,domain=domain,channel=channel,n_contents=len(local),
                    undefined_pairs=int((~np.isfinite(local).all(1)).sum()),
                    image_ratio=float(local[:,0].mean()) if np.isfinite(local[:,0]).all() else None,
                    mask_only_ratio=float(local[:,1].mean()) if np.isfinite(local[:,1]).all() else None,
                    delta_ratio=float((local[:,0]-local[:,1]).mean()) if np.isfinite(local).all() else None,
                    image_minus_acceptance=float(coverage[domains==domain,order,0,ch,0].mean()),
                    image_center_acceptance=float(coverage[domains==domain,order,0,ch,1].mean()),
                    image_plus_acceptance=float(coverage[domains==domain,order,0,ch,2].mean())))
    summary = None; gate = False
    if complete:
        delta = ratios[:,:,0]-ratios[:,:,1]
        summary = dict(image_ratio=float(ratios[:,:,0].mean()),mask_only_ratio=float(ratios[:,:,1].mean()),
            delta_ratio=float(delta.mean()),ci95=interval(delta,domains),orders=delta.mean((0,2)).tolist(),
            minimum_shift_channel_mean_acceptance=float(coverage[:,:,0].mean((0,1)).min()))
        gate = (summary['delta_ratio'] <= -.1 and summary['ci95'][1] < 0 and max(summary['orders']) < 0
                and summary['minimum_shift_channel_mean_acceptance'] >= .25)
    result = dict(status='COMPLETE',response_gate=bool(gate),all_denominators_defined=complete,summary=summary,
        independent_validation=False,segmentation_efficacy_test=False,target_label_reads=0,model_updates=0)
    save(root/'public/RESULTS.json',result); csvout(root/'public/ALL_CELLS.csv',cells)
    save(root/'public/COMPLETION_AUDIT.json',dict(pairs=96,contents=48,orders=2,channels=2,
         arms=2,perturbations=3,native_proposal_calls=576,source_feature_forwards=49,target_label_reads=0,
         frozen_state_unchanged=True,worker_retired_exit=0))
    save(root/'public/QUALIFICATION.json',read(root/'QUALIFICATION.json'))
    state.update(status='COMPLETE',label_reads=0);save(root/'RUN_STATE.json',state)


def main():
    c = read(os.environ['DIAG_CONFIG']); root = Path(c['output_root']); mode = os.environ['DIAG_MODE']
    if mode == 'check': save(root/'CPU_CHECK.json',self_check());return 0
    if mode == 'worker': worker(c,produce=produce);return 0
    if mode == 'score': aggregate(c);return 0
    if mode != 'controller' or read(root/'CPU_CHECK.json')['status'] != 'PASS': raise ValueError('entry gate')
    with (root/'started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    attempts = []
    try:
        attempts.append(run_child(c,'worker',c['gpu_cap_seconds'],True))
        if attempts[-1]['exit_code']==0:attempts.append(run_child(c,'score',c['cpu_cap_seconds'],False))
    finally:
        ledger=dict(attempts=attempts,all_workers_retired=True,
            gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),
            cpu_worker_seconds=c['prior_cpu_seconds']+sum(a['cpu_worker_seconds'] for a in attempts))
        save(root/'RESOURCE_LEDGER.json',ledger);save(root/'public/RESOURCE_LEDGER.json',ledger)
    return 0 if len(attempts)==2 and all(a['exit_code']==0 for a in attempts) else 1


if __name__=='__main__':raise SystemExit(main())
