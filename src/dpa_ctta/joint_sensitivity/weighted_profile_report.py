"""Anonymous aggregates for the bounded profile; never loads images or predictions."""
import csv
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ['PROFILE_ROOT'])
    frozen = json.loads((root/'FROZEN.json').read_text())
    state = json.loads((root/'RUN_STATE.json').read_text())
    ledger = json.loads((root/'RESOURCE_LEDGER.json').read_text())
    cpu_audit = json.loads((root/'CPU_AUDIT.json').read_text()) if (root/'CPU_AUDIT.json').exists() else []
    results = [json.loads(p.read_text()) for p in sorted((root/'results').glob('*.json'))]
    processes = [json.loads(p.read_text()) for p in (root/'processes').glob('*.json')]
    if any(p['active'] for p in processes):
        raise ValueError('workers have not retired')
    if state['status']=='COMPLETE' and (len(results)!=8 or sum(r['arrivals'] for r in results)!=256):
        raise ValueError('profile coverage mismatch')
    pairs = []
    for result in results:
        if result['condition'] != 'C_JS_ACQ_WEIGHTED': continue
        for baseline in results:
            if (baseline['physical_gpu'], baseline['order']) != (result['physical_gpu'], result['order']): continue
            if baseline['condition']==result['condition']: continue
            pairs.append(dict(candidate=result['condition'], baseline=baseline['condition'],
                physical_gpu=result['physical_gpu'], order=result['order'],
                wall_time_ratio=result['wall_seconds']['mean']/baseline['wall_seconds']['mean'],
                cuda_event_time_ratio=result['cuda_event_seconds']['mean']/baseline['cuda_event_seconds']['mean'],
                allocated_delta_MiB=(result['peak_allocated_bytes']-baseline['peak_allocated_bytes'])/1024**2,
                reserved_delta_MiB=(result['peak_reserved_bytes']-baseline['peak_reserved_bytes'])/1024**2))
    attempts = [dict(condition=frozen['conditions'][a['job']['condition_index']], physical_gpu=a['job']['gpu'],
                order=a['job']['order'], status=a['status'], exit_code=a['exit_code'],
                gpu_worker_seconds=a['gpu_worker_seconds']) for a in ledger['attempts']]
    report = dict(status=state['status'], scope=frozen['scope'], method_source_commit=frozen['method_source_commit'],
        runner_source_commit=frozen['runner_source_commit'], results=results, same_device_ratios=pairs,
        all_worker_processes_retired=True, planned_jobs=8, completed_jobs=state['completed'],
        actual_arrivals=sum(r['arrivals'] for r in results), formal_experiment_started=False,
        source_image_reads=0, label_reads=0, predictions_saved=False, checkpoint_saved=False,
        budget=dict(wall_cap_seconds=frozen['wall_cap_seconds'], gpu_worker_cap_seconds=frozen['gpu_worker_cap_seconds']),
        cost=dict(gpu_worker_seconds=ledger['gpu_worker_seconds'], gpu_worker_hours=ledger['gpu_worker_seconds']/3600,
                  wall_seconds=ledger['wall_seconds'], CPU_qualification_seconds=sum(a['seconds'] for a in cpu_audit),
                  CPU_qualification_attempts=cpu_audit,
                  charging=ledger['charging'], attempts=attempts),
        limitations=[frozen['limitations'],
            'C_JS_SHARED is the matched C-background filter comparison; original R46 CW_LSO_JS_SHARED changes the base loss as well.',
            'Derivative signs describe the current unsupervised objective, not Dice or clinical benefit. No descent constraint is imposed.',
            'Each condition follows its own adaptive trajectory; directional derivatives across conditions are not evaluated at an identical parameter point.',
            'Observer hooks are identical across arms; method-internal derivative logging is retained in the weighted arm timing.',
            'No CPU scorer, full trajectory, efficacy gate or automatic next experiment is part of this profile.'])
    out = root/'public'
    (out/'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (out/'PROTOCOL.json').write_text(json.dumps(frozen, indent=2)+'\n')
    with (out/'PAIRED_COST.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(pairs[0]) if pairs else ['candidate','baseline'])
        writer.writeheader(); writer.writerows(pairs)
    print(json.dumps(dict(status=report['status'], completed_jobs=report['completed_jobs'], actual_arrivals=report['actual_arrivals'],
                         gpu_worker_seconds=ledger['gpu_worker_seconds'], pairs=pairs)))


if __name__ == '__main__': main()
