"""Paired development comparison after all eighteen streams retire."""
import os
import time
import statistics as st
from pathlib import Path
from .method import ARMS
from ..r30_state_history.report import score_stream_rows
from ..r20_model_only_search.report import bootstrap_pair, csvout
from ..r10_12h_core.run import read, save


def gate(r):
    return (r['delta_pp'] >= .3 and min(r['order_delta_pp']) > 0
            and r['positive_trajectories'] >= 5 and r['worst_seed_averaged_cell_pp'] >= -2.)


def score(c, guard):
    root = Path(c['output_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU scorer required')
    records = [read(root/'processes'/f'online_{j["id"]}.json') for j in c['b_jobs']]
    if len(records) != 18 or any(r['active'] or r['exit_code'] != 0 for r in records):
        raise ValueError('all-online retirement barrier')
    save(root/'LABEL_RELEASE.json', dict(at=time.time(), all_workers_retired=True, SEARCH=True, REVIEW='development only'))
    rows, main = score_stream_rows(c, guard)
    pairs = []; intervals = []; negative = []
    for role in ('SEARCH', 'SEALED_REVIEW'):
        for candidate, baseline in [('C_HEAD', 'C_CONT'), ('C_HEAD', 'ANCHOR'), ('ANCHOR', 'C_CONT')]:
            r, ci, neg = bootstrap_pair(rows, candidate, baseline, c['seeds'], role=role)
            if r['status'] != 'COMPLETE':
                raise ValueError('incomplete paired comparison')
            pairs.append(r); intervals.extend(ci); negative.extend(neg)
    csvout(root/'public/PAIRED_CONTRASTS.csv', pairs)
    csvout(root/'public/PAIRED_DOMAIN_INTERVALS.csv', intervals)
    csvout(root/'public/ALL_NEGATIVE_CELLS.csv', negative)
    means = {a: st.mean(r['macro_Dice_percent'] for r in main if r['condition'] == a and r['role'] == 'SEARCH') for a in ARMS}
    gates = {r['baseline']: dict(passed=gate(r), contrast=r) for r in pairs if r['candidate']=='C_HEAD' and r['role']=='SEARCH'}
    result = dict(SEARCH_macro=means, gates=gates, promising=all(r['passed'] for r in gates.values()),
                  independent_confirmation=False, data_status='development-exposed', patient_linkage='UNKNOWN', ROI_provenance='UNKNOWN')
    save(root/'public/NEXT_DECISION.json', result)
    return result


def finish(c, state):
    root = Path(c['output_root']); public = root/'public'
    attempts = [read(p) for p in sorted((root/'attempts').glob('*.json'))]
    csvout(public/'COST.csv', [dict(phase=r['phase'], attempt=r['attempt'], status=r['status'], **r['cost']) for r in attempts])
    complete = state['status']=='COMPLETE'
    audit = dict(status=state['status'], completed_jobs=sum(v=='COMPLETE' for v in state['jobs'].values()),
                 GPU_cap_hours=None, wall_seconds=time.time()-c['origin']['T0_epoch'],
                 gpu_worker_seconds=sum(r['cost'].get('gpu_seconds',0) for r in attempts),
                 cpu_worker_seconds=sum(r['cost'].get('cpu_seconds',0) for r in attempts),
                 failure_count=sum(r['status']!='COMPLETE' for r in attempts),
                 barrier_verified=complete and read(root/'LABEL_RELEASE.json')['all_workers_retired'])
    save(public/'COMPLETION_AUDIT.json', audit)
    text = '# R31: adapting the existing final classifier\n\nStatus: **'+state['status']+'**.\n\n'
    if complete:
        d=state['decision']
        text += '| Arm | SEARCH macro Dice (%) |\n|---|---:|\n'+''.join(f'| {k} | {v:.6f} |\n' for k,v in d['SEARCH_macro'].items())
        for k,v in d['gates'].items():
            text += f"\nC_HEAD versus {k}: {v['contrast']['delta_pp']:+.6f} pp; development gate {v['passed']}.\n"
        text += '\nPass against both references: '+str(d['promising'])+'. No learning-rate sweep is authorized by this result.\n'
    else:
        text += 'Incomplete execution is not scientific negative evidence.\n'
    text += f"\nGPU-worker time {audit['gpu_worker_seconds']/3600:.4f} h includes profiles and failures.\n"
    text += '\nThe 66 existing final-classifier scalars use fixed Adam LR 1e-5; BN affine uses 1e-4. Same one-step consistency objective and current-image readout. No extra model or source training. SEARCH and legacy REVIEW are development-exposed; REVIEW and coupled-content intervals are descriptive, with unknown patient linkage and ROI provenance. This only tests the registered parameter family and learning rate. Public artifacts exclude data, masks, identities, states and weights. Publication verification is a separate delivery step.\n'
    (public/'REPORT.md').write_text(text)
