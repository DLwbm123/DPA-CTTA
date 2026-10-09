"""CPU paired counterfactual scoring after all online jobs retire."""
import json
import os
import time
import statistics as st
from pathlib import Path
from contextlib import ExitStack
from collections import defaultdict
from .online import HISTORIES, RULES
from ..r30_state_history.report import mask_at, WIDTH
from ..r24_c_context.run import hard_metrics
from ..r20_model_only_search.report import csvout
from ..r10_12h_core.run import read, save
from ..r7_target_screen.runner import TargetReader


def bins(offset, length):
    if not 1 <= offset <= length:
        raise ValueError('invalid query position')
    return ['ALL','Q'+str(min(3,(offset-1)*4//length)+1)] + (['FIRST64'] if offset <= 64 else [])


def contrasts(cells):
    """Preserve domain/seed/order/channel; never change references by outcome."""
    index = {(r['domain'],r['role'],r['seed'],r['order'],r['position_bin'],r['channel'],r['condition']):r for r in cells}
    pairs = [('SAME64_HOLD','CROSS64_HOLD'),('SAME64_UPDATE','CROSS64_UPDATE'),('NATIVE_HOLD','SAME64_HOLD')]
    pairs += [(h+'_HOLD','SOURCE_HOLD') for h in HISTORIES if h!='SOURCE']
    pairs += [(h+'_UPDATE',h+'_HOLD') for h in HISTORIES]
    pairs += [(h+'_'+u,ref) for h in HISTORIES for u in RULES for ref in ('NATIVE_UPDATE','ANCHOR') if h+'_'+u != ref]
    result=[]
    for r in cells:
        if r['condition'] != 'NATIVE_UPDATE':
            continue
        key=(r['domain'],r['role'],r['seed'],r['order'],r['position_bin'],r['channel'])
        for a,b in pairs:
            x,y=index[key+(a,)],index[key+(b,)]
            if x['scored_contents'] != y['scored_contents']:
                raise ValueError('paired content coverage')
            result.append(dict(zip(('domain','role','seed','order','position_bin','channel'),key),
                               candidate=a,baseline=b,delta_pp=x['Dice_percent']-y['Dice_percent'],scored_contents=x['scored_contents']))
        v=lambda name:index[key+(name,)]['Dice_percent']
        result.append(dict(zip(('domain','role','seed','order','position_bin','channel'),key),candidate='HISTORY_UPDATE_INTERACTION',
                           baseline='ALGEBRA',delta_pp=v('SAME64_UPDATE')-v('SAME64_HOLD')-v('CROSS64_UPDATE')+v('CROSS64_HOLD'),
                           scored_contents=r['scored_contents']))
    return result


def score(c, guard):
    root=Path(c['output_root']); jobs=c['jobs']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only scoring')
    processes=[read(root/'processes'/f'online_{j["id"]}.json') for j in jobs]
    if len(processes)!=6 or any(p['active'] or p['exit_code']!=0 for p in processes):
        raise ValueError('all-online retirement barrier')
    save(root/'LABEL_RELEASE.json',dict(at=time.time(),all_workers_retired=True,SEARCH=True,REVIEW='development only'))
    split=read(root/'scorer/SPLIT.private.json'); plans=read(root/'scorer/PLAN.json')
    reader=TargetReader(c['target_root'],256*1024**2,'mask'); cells=[]; diagnostics=[]; observations=0; expected=0
    with (root/'scores/QUERY.private.jsonl').open('x') as private:
        for job in jobs:
            rows=read(root/f'scorer/FULL_o{job["order"]}.json'); dest=root/'target'/job['id']
            receipt=read(dest/'COMPLETE.json')
            if receipt['replay']['arrivals']!=1951 or len(receipt['branches'])!=16 or receipt['history_updates']!=256:
                raise ValueError('formal replay/history/branch coverage')
            for case in plans[str(job['order'])]:
                n=len(case['query']); groups=defaultdict(list)
                with ExitStack() as stack:
                    handles={}; pre={}
                    for history in HISTORIES:
                        for rule in RULES:
                            name=history+'_'+rule; d=dest/f'{case["id"]}_{name}'
                            q=read(d/'COMPLETE.json')
                            if q['arrivals']!=n or (d/'predictions.bits').stat().st_size!=n*WIDTH or (d/'pre.bits').stat().st_size!=(n*WIDTH if rule=='UPDATE' else 0):
                                raise ValueError('query mask coverage')
                            traces=[json.loads(s) for s in (d/'traces.jsonl').read_text().splitlines()]
                            if [t['visit'] for t in traces]!=case['query']:
                                raise ValueError('query trace position mismatch')
                            for t in traces:
                                for key,value in t['diagnostics'].items():
                                    diagnostics.append(dict(domain=case['domain'],seed=job['seed'],order=job['order'],condition=name,metric=key,value=value))
                            handles[name]=stack.enter_context((d/'predictions.bits').open('rb'))
                            if rule=='UPDATE':pre[name]=stack.enter_context((d/'pre.bits').open('rb'))
                    references={arm:stack.enter_context((Path(c['snapshot_input_root'])/f'B_{arm}_s{job["seed"]}_o{job["order"]}'/'predictions.bits').open('rb')) for arm in ('ANCHOR','C_EPISODIC')}
                    for offset,visit in enumerate(case['query'],1):
                        guard(); row=rows[visit-1]; role=split.get(row['image_sha256'],'CONTEXT')
                        if role not in ('SEARCH','SEALED_REVIEW'):
                            continue
                        if row['domain'] != case['domain']:
                            raise ValueError('scorer domain plan mismatch')
                        expected += 10
                        label=reader.read(row); metrics={name:hard_metrics(mask_at(f,offset-1),label) for name,f in handles.items()}
                        metrics.update({arm:hard_metrics(mask_at(f,visit-1),label) for arm,f in references.items()})
                        before={name:hard_metrics(mask_at(f,offset-1),label) for name,f in pre.items()}
                        for name,values in metrics.items():
                            private.write(json.dumps(dict(condition=name,domain=case['domain'],seed=job['seed'],order=job['order'],content=row['image_sha256'],role=role,visit=visit,query_offset=offset,metrics=values,
                                                          pre_metrics=before.get(name)))+'\n');observations+=1
                            for bucket in bins(offset,n):
                                for k,channel in enumerate(('OD','OC')):
                                    groups[role,bucket,channel,name].append((values[k]['dice'],None if name not in before else before[name][k]['dice']))
                    for (role,bucket,channel,name),values in groups.items():
                        cells.append(dict(domain=case['domain'],role=role,seed=job['seed'],order=job['order'],position_bin=bucket,channel=channel,condition=name,scored_contents=len(values),
                                          Dice_percent=100*st.mean(x[0] for x in values),pre_Dice_percent=None if values[0][1] is None else 100*st.mean(x[1] for x in values),
                                          immediate_update_pp=None if values[0][1] is None else 100*st.mean(x[0]-x[1] for x in values)))
    if observations != expected or expected == 0:
        raise ValueError('scored observation coverage')
    csvout(root/'public/QUERY_CELLS.csv',cells); paired=contrasts(cells);csvout(root/'public/PAIRED_CELLS.csv',paired)
    csvout(root/'public/ALL_NEGATIVE_CELLS.csv',[r for r in paired if r['delta_pp']<0])
    groups=defaultdict(list)
    for r in paired:groups[r['domain'],r['role'],r['position_bin'],r['candidate'],r['baseline']].append(r)
    summary=[]
    for key,values in sorted(groups.items()):
        trajectories=defaultdict(list)
        for v in values:trajectories[v['seed'],v['order']].append(v['delta_pp'])
        effects=[st.mean(v) for v in trajectories.values()]
        summary.append(dict(zip(('domain','role','position_bin','candidate','baseline'),key),delta_pp=st.mean(effects),
                            order_delta_pp=[st.mean(v['delta_pp'] for v in values if v['order']==o) for o in (0,1)],
                            positive_trajectories=sum(x>0 for x in effects),negative_trajectories=sum(x<0 for x in effects),
                            trajectories=6,worst_seed_channel_pp=min(v['delta_pp'] for v in values)))
    csvout(root/'public/PAIRED_SUMMARY.csv',summary)
    groups=defaultdict(list)
    for r in diagnostics:groups[r['domain'],r['seed'],r['order'],r['condition'],r['metric']].append(r['value'])
    csvout(root/'public/STATE_DIAGNOSTICS.csv',[dict(zip(('domain','seed','order','condition','metric'),key),mean=st.mean(v),minimum=min(v),maximum=max(v),count=len(v)) for key,v in sorted(groups.items())])
    result=dict(status='COMPLETE',scored_observations=observations,independent_confirmation=False,
                primary=[r for r in summary if r['role']=='SEARCH' and r['position_bin']=='ALL' and ((r['candidate'],r['baseline']) in [('SAME64_HOLD','CROSS64_HOLD'),('NATIVE_HOLD','SAME64_HOLD')] or r['baseline']=='SOURCE_HOLD' or r['candidate'].endswith('_UPDATE') and r['baseline']==r['candidate'].replace('_UPDATE','_HOLD'))],
                next_decision='INTERPRET_HISTORY_AND_UPDATE_EFFECTS_NO_AUTOMATIC_CONTROLLER')
    save(root/'public/NEXT_DECISION.json',result)
    return result


def finish(c,state):
    root=Path(c['output_root']); attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))]
    csvout(root/'public/COST.csv',[dict(phase=a['phase'],attempt=a['attempt'],status=a['status'],**a['cost']) for a in attempts])
    complete=state['status']=='COMPLETE'
    audit=dict(status=state['status'],completed_jobs=sum(v=='COMPLETE' for v in state['jobs'].values()),
               GPU_cap_hours=None,gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),cpu_worker_seconds=sum(a['cost'].get('cpu_seconds',0) for a in attempts),
               failure_count=sum(a['status']!='COMPLETE' for a in attempts),wall_seconds=time.time()-c['origin']['T0_epoch'],
               barrier_verified=complete and read(root/'LABEL_RELEASE.json')['all_workers_retired'])
    save(root/'public/COMPLETION_AUDIT.json',audit)
    text='# R32: history origin and subsequent within-domain updates\n\nStatus: **'+state['status']+'**.\n\n'
    if complete:
        text+='| Query domain | Contrast | SEARCH delta (pp) | Order0 / order1 | Positive trajectories |\n|---|---|---:|---|---:|\n'
        for r in state['decision']['primary']:
            text+=f"| {r['domain']} | {r['candidate']} - {r['baseline']} | {r['delta_pp']:+.6f} | {r['order_delta_pp']} | {r['positive_trajectories']}/6 |\n"
    text+='\nSAME64 and CROSS64 use exactly64 disjoint past images,64 C updates, matched augmentation RNG and Adam step64. SOURCE has no history. NATIVE retains the full original prefix and its different optimizer age; NATIVE comparisons do not isolate history origin from length. All branches share query images and per-arrival augmentation draws. HOLD freezes learned parameters/moments but still uses current-image BN statistics. UPDATE retains state and applies one native C step per query.\n\n'
    text+='Full query tails start at within-domain image65; the first64 images were assigned to matched same-domain history, not included in the query endpoint. Domain identities define offline interventions and scoring, never an online decision feature. This is an oracle diagnostic, not a deployable domain selector. Earlier loss onsets and K=64 are post-hoc development choices. No independent patient or clinical claim; SEARCH and legacy REVIEW are exposed, patient linkage/ROI provenance UNKNOWN. The experiment does not isolate Adam versus parameter effects or exhaust possible histories. No automatic controller or LR sweep follows.\n\n'
    text+=f"Actual GPU-worker hours including profiles/failures: {audit['gpu_worker_seconds']/3600:.6f}. All cells, negative effects, immediate pre/post changes, state diagnostics and costs are retained. Public artifacts exclude images, labels, masks, identities, snapshots, weights and private paths. GitHub delivery verification is separate.\n"
    (root/'public/REPORT.md').write_text(text)
