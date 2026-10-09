"""CPU-only fixed factorial contrasts and unchanged R32/R30 references."""
import os
import time
import json
import statistics as st
from pathlib import Path
from contextlib import ExitStack
from collections import defaultdict
from .online import HISTORIES
from ..r32_history_origin.report import bins
from ..r30_state_history.report import mask_at, WIDTH
from ..r24_c_context.run import hard_metrics
from ..r20_model_only_search.report import csvout
from ..r10_12h_core.run import read, save
from ..r7_target_screen.runner import TargetReader

REFERENCES = ('SAME64_HOLD','CROSS64_HOLD','SOURCE_HOLD','C_CONT','ANCHOR')
PAIRS = [('SC','SS'),('CC','CS'),('SS','CS'),('SC','CC'),('SS','CC')]
PAIRS += [(a,'SAME64_HOLD' if a[0]=='S' else 'CROSS64_HOLD') for a in HISTORIES]
PAIRS += [(a,b) for a in HISTORIES for b in ('C_CONT','ANCHOR','SOURCE_HOLD')]


def contrasts(cells):
    keys = ('domain','role','seed','order','position_bin','channel')
    index = {tuple(r[k] for k in keys)+(r['condition'],):r for r in cells}
    out = []
    for r in cells:
        if r['condition'] != 'SS': continue
        key = tuple(r[k] for k in keys)
        def value(arm):
            x = index[key+(arm,)]
            if x['scored_contents'] != r['scored_contents']: raise ValueError('unpaired coverage')
            return x['Dice_percent']
        for a,b in PAIRS:
            out.append(dict(zip(keys,key),candidate=a,baseline=b,delta_pp=value(a)-value(b),scored_contents=r['scored_contents']))
        out.append(dict(zip(keys,key),candidate='OPTIMIZER_BY_PARAMETER_INTERACTION',baseline='ALGEBRA',delta_pp=(value('SC')-value('SS'))-(value('CC')-value('CS')),scored_contents=r['scored_contents']))
    return out


def score(c,guard):
    root = Path(c['output_root']); jobs = c['jobs']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only scoring')
    processes = [read(root/'processes'/f'online_{j["id"]}.json') for j in jobs]
    if len(processes) != 6 or any(p['active'] or p['exit_code'] != 0 for p in processes):
        raise ValueError('all-online retirement barrier')
    save(root/'LABEL_RELEASE.json',dict(at=time.time(),all_workers_retired=True,SEARCH=True,REVIEW='development only'))
    split = read(root/'scorer/SPLIT.private.json'); plans = read(root/'scorer/PLAN.json')
    reader = TargetReader(c['target_root'],256*1024**2,'mask')
    cells = []; diagnostics = []; observations = 0
    with (root/'scores/QUERY.private.jsonl').open('x') as private:
        for job in jobs:
            rows = read(root/f'scorer/FULL_o{job["order"]}.json'); dest = root/'target'/job['id']
            receipt = read(dest/'COMPLETE.json')
            if receipt['replay']['arrivals'] != 1951 or receipt['history_updates'] != 256 or len(receipt['branches']) != 8:
                raise ValueError('formal job coverage')
            if receipt['diagonal_query_checks'] != 2644 or receipt['first_pre_invariance_checks'] != 4:
                raise ValueError('state exchange/replay checks incomplete')
            for case in plans[str(job['order'])]:
                n = len(case['query']); groups = defaultdict(list)
                with ExitStack() as stack:
                    post = {}; pre = {}
                    for arm in HISTORIES:
                        d = dest/f'{case["id"]}_{arm}_UPDATE'; receipt = read(d/'COMPLETE.json')
                        if receipt['arrivals'] != n or any((d/name).stat().st_size != n*WIDTH for name in ('predictions.bits','pre.bits')):
                            raise ValueError('mask coverage')
                        trace = [json.loads(x) for x in (d/'traces.jsonl').read_text().splitlines()]
                        if [x['visit'] for x in trace] != case['query']: raise ValueError('trace query mismatch')
                        for t in trace:
                            for metric,value in t['diagnostics'].items():
                                diagnostics.append(dict(domain=case['domain'],seed=job['seed'],order=job['order'],condition=arm,metric=metric,value=value))
                        post[arm] = stack.enter_context((d/'predictions.bits').open('rb'))
                        pre[arm] = stack.enter_context((d/'pre.bits').open('rb'))
                    for arm in REFERENCES[:3]:
                        d = Path(c['r32_target_root'])/job['id']/f'{case["id"]}_{arm}'
                        if (d/'predictions.bits').stat().st_size != n*WIDTH: raise ValueError('HOLD reference coverage')
                        post[arm] = stack.enter_context((d/'predictions.bits').open('rb'))
                    for arm in REFERENCES[3:]:
                        d = Path(c['snapshot_input_root'])/f'B_{arm}_s{job["seed"]}_o{job["order"]}'
                        post[arm] = stack.enter_context((d/'predictions.bits').open('rb'))
                    for offset,visit in enumerate(case['query'],1):
                        guard(); row = rows[visit-1]; role = split.get(row['image_sha256'],'CONTEXT')
                        if role not in ('SEARCH','SEALED_REVIEW'): continue
                        if row['domain'] != case['domain']: raise ValueError('scorer domain mismatch')
                        label = reader.read(row)
                        for arm,f in post.items():
                            values = hard_metrics(mask_at(f,visit-1 if arm in REFERENCES[3:] else offset-1),label)
                            before = hard_metrics(mask_at(pre[arm],offset-1),label) if arm in pre else None
                            private.write(json.dumps(dict(condition=arm,domain=case['domain'],seed=job['seed'],order=job['order'],content=row['image_sha256'],role=role,visit=visit,query_offset=offset,metrics=values,pre_metrics=before))+'\n')
                            observations += 1
                            for bucket in bins(offset,n):
                                for k,ch in enumerate(('OD','OC')):
                                    groups[role,bucket,ch,arm].append((values[k]['dice'],None if before is None else before[k]['dice']))
                    for (role,bucket,ch,arm),values in groups.items():
                        cells.append(dict(domain=case['domain'],role=role,seed=job['seed'],order=job['order'],position_bin=bucket,channel=ch,condition=arm,scored_contents=len(values),Dice_percent=100*st.mean(x[0] for x in values),pre_Dice_percent=None if values[0][1] is None else 100*st.mean(x[1] for x in values),immediate_update_pp=None if values[0][1] is None else 100*st.mean(x[0]-x[1] for x in values)))
    if observations != c['expected_scored_observations']: raise ValueError('scored observation coverage')
    csvout(root/'public/QUERY_CELLS.csv',cells); paired = contrasts(cells)
    csvout(root/'public/PAIRED_CELLS.csv',paired);csvout(root/'public/ALL_NEGATIVE_CELLS.csv',[x for x in paired if x['delta_pp']<0])
    groups = defaultdict(list)
    for r in paired: groups[r['domain'],r['role'],r['position_bin'],r['candidate'],r['baseline']].append(r)
    summary = []
    for key,values in sorted(groups.items()):
        trajectories = defaultdict(list)
        for x in values: trajectories[x['seed'],x['order']].append(x['delta_pp'])
        effects = [st.mean(v) for v in trajectories.values()]
        summary.append(dict(zip(('domain','role','position_bin','candidate','baseline'),key),delta_pp=st.mean(effects),order_delta_pp=[st.mean(v['delta_pp'] for v in values if v['order']==o) for o in (0,1)],positive_trajectories=sum(v>0 for v in effects),negative_trajectories=sum(v<0 for v in effects),trajectories=len(effects),worst_seed_channel_pp=min(x['delta_pp'] for x in values)))
    csvout(root/'public/PAIRED_SUMMARY.csv',summary)
    groups = defaultdict(list)
    for x in diagnostics: groups[x['domain'],x['seed'],x['order'],x['condition'],x['metric']].append(x['value'])
    csvout(root/'public/STATE_DIAGNOSTICS.csv',[dict(zip(('domain','seed','order','condition','metric'),k),mean=st.mean(v),minimum=min(v),maximum=max(v),count=len(v)) for k,v in sorted(groups.items())])
    primary = [r for r in summary if r['role']=='SEARCH' and r['position_bin']=='ALL' and ((r['candidate'],r['baseline']) in PAIRS[:5] or r['baseline']=='ALGEBRA')]
    result = dict(status='COMPLETE',scored_observations=observations,primary=primary,independent_confirmation=False,next_decision='INTERPRET_CONDITIONAL_STATE_AND_COMPATIBILITY_EFFECTS')
    save(root/'public/NEXT_DECISION.json',result)
    return result


def finish(c,state):
    root = Path(c['output_root']); attempts = [read(p) for p in sorted((root/'attempts').glob('*.json'))]
    csvout(root/'public/COST.csv',[dict(phase=a['phase'],attempt=a['attempt'],status=a['status'],**a['cost']) for a in attempts])
    audit = dict(status=state['status'],completed_jobs=sum(v=='COMPLETE' for v in state['jobs'].values()),GPU_cap_hours=None,gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),cpu_worker_seconds=sum(a['cost'].get('cpu_seconds',0) for a in attempts),failure_count=sum(a['status']!='COMPLETE' for a in attempts),wall_seconds=time.time()-c['origin']['T0_epoch'],barrier_verified=state['status']=='COMPLETE' and read(root/'LABEL_RELEASE.json')['all_workers_retired'])
    save(root/'public/COMPLETION_AUDIT.json',audit)
    text = '# R33 matched-age parameter/Adam state exchange\n\nStatus: **'+state['status']+'**.\n\n'
    if state['status']=='COMPLETE':
        text += '| Domain | Contrast | SEARCH pp | Order0 / order1 | Positive trajectories |\n|---|---|---:|---|---:|\n'
        for r in state['decision']['primary']:
            text += f"| {r['domain']} | {r['candidate']} - {r['baseline']} | {r['delta_pp']:+.6f} | {r['order_delta_pp']} | {r['positive_trajectories']}/{r['trajectories']} |\n"
    text += '\nFirst letter is BN-affine parameter history; second letter is Adam history. S=SAME64, C=CROSS64. Both histories have64 updates and Adam step64. SC-SS and CC-CS compare Adam donors at fixed parameter states; SS-CS and SC-CC compare parameter donors at fixed Adam states. Interaction=(SC-SS)-(CC-CS). Hybrid losses can indicate parameter/optimizer incompatibility, not universally harmful optimizer memory. Diagonals reproduce R32 pre/post hard masks. Frozen references reuse R32 on the identical query tails.\n\nR32 already established prediction-relevant parameter history differences; this round tests subsequent-update sensitivity to Adam memory. No online domain labels, learned selector, LR grid, new source checkpoint or optimizer age reset. All ordered pairs, channels, fixed first64/quarter bins and negative cells are retained. SEARCH/legacy REVIEW are development-exposed; unknown patient linkage and ROI provenance prohibit independent or clinical claims. State exchange is an offline diagnostic, not an executable policy.\n'
    text += f"\nGPU-worker hours including profiles/failures: {audit['gpu_worker_seconds']/3600:.6f}. Public delivery verification is separate.\n"
    (root/'public/REPORT.md').write_text(text)
