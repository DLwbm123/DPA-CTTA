"""Future-only paired action values; window oracles never switch per image."""
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics as st
import time
from .method import PAUSE, HORIZONS, PIVOTS
from ..r10_12h_core.run import read, save
from ..r20_model_only_search.report import csvout
from ..r30_state_history.report import mask_at, WIDTH
from ..r24_c_context.run import hard_metrics
from ..r7_target_screen.runner import TargetReader


def summarize(windows):
    groups=defaultdict(list)
    for w in windows: groups[w['role'],w['horizon'],w['seed'],w['order']].append(w)
    trajectories=[]
    for (role,horizon,seed,order),ws in sorted(groups.items()):
        trajectories.append(dict(role=role,horizon=horizon,seed=seed,order=order,windows=len(ws),
            update=st.mean(w['update'] for w in ws),hold=st.mean(w['hold'] for w in ws),
            oracle=st.mean(w['oracle'] for w in ws)))
    endpoints=[]
    for role,horizon in sorted({(x['role'],x['horizon']) for x in trajectories}):
        xs=[x for x in trajectories if x['role']==role and x['horizon']==horizon]
        update,back,oracle=(st.mean(x[k] for x in xs) for k in ('update','hold','oracle'))
        best='HOLD' if back>update else 'UPDATE'
        orders=[]
        for o in (0,1):
            ys=[x for x in xs if x['order']==o]
            if not ys: continue
            k,b,q=(st.mean(x[z] for x in ys) for z in ('update','hold','oracle'))
            orders.append(dict(order=o,hold_minus_update=b-k,oracle_minus_best_fixed=q-max(k,b)))
        endpoints.append(dict(role=role,horizon=horizon,groups=len(xs),update_Dice_percent=update,hold_Dice_percent=back,
            hold_minus_update=back-update,oracle_Dice_percent=oracle,oracle_minus_update=oracle-update,
            best_fixed_action=best,oracle_minus_best_fixed=oracle-max(update,back),order_effects=orders,
            positive_fixed_groups=sum(x['hold']>x['update'] for x in xs),
            fixed_action_gate=(len(xs)==6 and len(orders)==2 and back-update>=.3 and all(x['hold_minus_update']>0 for x in orders) and sum(x['hold']>x['update'] for x in xs)>=5),
            conditional_headroom_gate=(len(xs)==6 and len(orders)==2 and oracle-max(update,back)>=.3 and all(x['oracle_minus_best_fixed']>=.15 for x in orders))))
    return trajectories,endpoints


def score(c,guard):
    root=Path(c['output_root']);jobs=c['jobs']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU-only scorer')
    if len(jobs)!=6: raise ValueError('registered six-job matrix')
    for j in jobs:
        p=read(root/'processes'/f'online_{j["id"]}.json');a=read(root/'attempts'/f'online_{j["id"]}.0.json')
        if p['active'] or p['exit_code']!=0 or a['status']!='COMPLETE': raise ValueError('all-online retirement barrier')
    save(root/'LABEL_RELEASE.json',dict(at=time.time(),all_workers_retired=True,SEARCH=True,REVIEW='development exposed'))
    split=read(root/'scorer/SPLIT.private.json');reader=TargetReader(c['target_root'],256*1024**2,'mask')
    cells=[];windows=[];domain_cells=[];eligibility=[];observations=0
    with (root/'scores/FUTURE.private.jsonl').open('x') as private:
        for j in jobs:
            rows=read(root/f'scorer/FULL_o{j["order"]}.json');dest=root/'target'/j['id'];receipt=read(dest/'COMPLETE.json')
            if receipt['replay_arrivals']!=1951 or receipt['reference_checks']!=1951 or receipt['hold_future_arrivals']!=3646 or len(receipt['windows'])!=15:
                raise ValueError('formal coverage changed')
            if (dest/'native.bits').stat().st_size!=1951*WIDTH: raise ValueError('native mask coverage')
            with (dest/'native.bits').open('rb') as update:
                for t in PIVOTS:
                    length=min(256,len(rows)-t);values=defaultdict(list)
                    if (dest/f'hold_{t}.bits').stat().st_size!=length*WIDTH: raise ValueError('hold mask coverage')
                    with (dest/f'hold_{t}.bits').open('rb') as back:
                        for offset in range(1,length+1):
                            guard();visit=t+offset;row=rows[visit-1];role=split.get(row['image_sha256'],'CONTEXT')
                            if role not in ('SEARCH','SEALED_REVIEW'): continue
                            label=reader.read(row);ka=hard_metrics(mask_at(update,visit-1),label);ba=hard_metrics(mask_at(back,offset-1),label)
                            private.write(json.dumps(dict(job=j['id'],pivot=t,visit=visit,content=row['image_sha256'],role=role,update=ka,hold=ba))+'\n')
                            observations+=2
                            for channel,idx in (('OD',0),('OC',1)):
                                values[role,channel].append(dict(offset=offset,domain=row['domain'],update=100*ka[idx]['dice'],hold=100*ba[idx]['dice']))
                    for horizon in HORIZONS:
                        for role in ('SEARCH','SEALED_REVIEW'):
                            xs=[x for x in values[role,'OD'] if x['offset']<=horizon]
                            eligible=length>=horizon and bool(xs)
                            meta=dict(seed=j['seed'],order=j['order'],pivot=t,origin_domain=rows[t-1]['domain'],horizon=horizon,role=role)
                            eligibility.append(dict(meta,available_future=length,eligible=eligible,scored_contents=len(xs),reason='ELIGIBLE' if eligible else ('TRUNCATED' if length<horizon else 'NO_ELIGIBLE_CONTENT')))
                            if not eligible: continue
                            channel_rows=[]
                            for channel in ('OD','OC'):
                                ys=[x for x in values[role,channel] if x['offset']<=horizon]
                                k,b=(st.mean(x[z] for x in ys) for z in ('update','hold'))
                                row=dict(meta,channel=channel,scored_contents=len(ys),update=k,hold=b,delta_pp=b-k,
                                         cumulative_delta_pp=sum(x['hold']-x['update'] for x in ys))
                                cells.append(row);channel_rows.append(row)
                                for domain in sorted({x['domain'] for x in ys}):
                                    zs=[x for x in ys if x['domain']==domain];dk,db=(st.mean(x[z] for x in zs) for z in ('update','hold'))
                                    domain_cells.append(dict(meta,channel=channel,future_domain=domain,scored_contents=len(zs),update=dk,hold=db,delta_pp=db-dk))
                            k,b=(st.mean(x[z] for x in channel_rows) for z in ('update','hold'))
                            action='HOLD' if b>k else 'UPDATE'
                            windows.append(dict(meta,scored_contents=len(xs),update=k,hold=b,delta_pp=b-k,oracle=max(k,b),oracle_action=action,
                                cumulative_delta_pp=st.mean(x['cumulative_delta_pp'] for x in channel_rows)))
                            for x in channel_rows: x.update(oracle=x['hold'] if action=='HOLD' else x['update'],oracle_action=action)
    if observations!=c['expected_scored_observations']: raise ValueError('scored observation registration mismatch')
    trajectories,endpoints=summarize(windows)
    primary=next(x for x in endpoints if x['role']=='SEARCH' and x['horizon']==256)
    if primary['groups']!=6 or any(x['windows']!=13 for x in trajectories if x['role']=='SEARCH' and x['horizon']==256):
        raise ValueError('primary coverage incomplete')
    strata=[];groups=defaultdict(list)
    for x in windows: groups[x['role'],x['horizon'],x['origin_domain'],x['order']].append(x)
    for key,xs in sorted(groups.items()):
        strata.append(dict(zip(('role','horizon','origin_domain','order'),key),windows=len(xs),update=st.mean(x['update'] for x in xs),hold=st.mean(x['hold'] for x in xs),delta_pp=st.mean(x['delta_pp'] for x in xs)))
    negatives=[dict(cell_type='WINDOW_CHANNEL',**x) for x in cells if x['delta_pp']<0]
    negatives += [dict(cell_type='FUTURE_DOMAIN_CHANNEL',**x) for x in domain_cells if x['delta_pp']<0]
    negatives += [dict(cell_type='ORIGIN_STRATUM',**x) for x in strata if x['delta_pp']<0]
    for name,rows in [('WINDOW_CELLS',cells),('WINDOW_SUMMARY',windows),('FUTURE_DOMAIN_CELLS',domain_cells),('ELIGIBILITY',eligibility),('TRAJECTORIES',trajectories),('ENDPOINTS',endpoints),('ORIGIN_STRATA',strata),('ALL_NEGATIVE_CELLS',negatives)]:
        csvout(root/'public'/f'{name}.csv',rows)
    decision=('REGISTER_OBSERVABILITY_TEST' if primary['conditional_headroom_gate'] else
              'REGISTER_FIXED_ACTION_FULL_POLICY_CHECK' if primary['fixed_action_gate'] else 'STOP_REGISTERED_PROSPECTIVE_HOLD')
    result=dict(status='COMPLETE',scored_observations=observations,primary=primary,next_decision=decision,
                independent_confirmation=False,RL_authorized_by_result=False)
    save(root/'public/NEXT_DECISION.json',result);return result


def finish(c,state):
    root=Path(c['output_root']);attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))]
    csvout(root/'public/COST.csv',[dict(phase=a['phase'],attempt=a['attempt'],status=a['status'],**a['cost']) for a in attempts])
    audit=dict(status=state['status'],completed_jobs=sum(v=='COMPLETE' for v in state['jobs'].values()),
               gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),cpu_worker_seconds=sum(a['cost'].get('cpu_seconds',0) for a in attempts),
               failure_count=sum(a['status']!='COMPLETE' for a in attempts),wall_seconds=time.time()-c['origin']['T0_epoch'],
               barrier_verified=state['status']=='COMPLETE' and read(root/'LABEL_RELEASE.json')['all_workers_retired'])
    save(root/'public/COMPLETION_AUDIT.json',audit)
    text='# R35 prospective update pause action value\n\nStatus: **'+state['status']+'**.\n\n'
    if state['status']=='COMPLETE':
        p=state['decision']['primary']
        text+=f"Primary H256 SEARCH: HOLD−UPDATE {p['hold_minus_update']:+.6f}pp; oracle−UPDATE {p['oracle_minus_update']:+.6f}pp; oracle−best-fixed {p['oracle_minus_best_fixed']:+.6f}pp. Best fixed action: {p['best_fixed_action']}.\n\nNext decision: **{state['decision']['next_decision']}**. This is a resource-allocation decision, not permission to start RL or claim a deployable selector.\n\n"
    text+='HOLD256 preserves the current learned parameters and Adam state, performs current-image BN inference without updates, and advances arrivals and paired RNG. UPDATE reuses the freshly verified native trajectory. H256 is primary and H64 auxiliary. This tests future freezing, not retrospective rollback. Each branch starts independently; no full-stream controlled policy is evaluated.\n\n'
    text+='The primary mean weights windows within each seed/order and then the six groups equally. Origin/future-domain strata, negative cells and all denominators are published separately. This is not a balanced four-domain or full-policy efficacy estimate. Windows overlap, contents repeat, SEARCH/legacy REVIEW are development-exposed and patient/ROI provenance is unknown. No independent/clinical claim or label-free observability/RL claim is supported by an oracle alone.\n\n'
    text+=f"GPU-worker time including profiles/failures: {audit['gpu_worker_seconds']/3600:.6f}h. Public delivery is verified separately.\n"
    (root/'public/REPORT.md').write_text(text)
