"""Fixed diagnostic snapshots; labels remain closed until all six replays retire."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
import numpy as np
import torch
from .method import Host, actions_for
from ..r20_model_only_search import runtime as base
from ..r20_model_only_search.report import csvout
from ..r10_12h_core.run import read, save, sha
from ..r7_target_screen.runner import TargetReader
from ..r19_model_only.runtime import weights, image
from ..r24_c_context.run import hard_metrics

ID='R28_COMMON_STATE_DIAGNOSTIC'


def access_guard(c):
    root=Path(c['output_root']).resolve();checkpoint=Path(c['checkpoint_path']).resolve()
    current={'image':None}
    libraries=[Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve()]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root/'scorer') or p.is_relative_to(root/'scores'):
            raise PermissionError('diagnostic online label/score barrier')
        if p.suffix.lower() in ('.png','.jpg','.jpeg','.pt','.pth','.npy','.npz','.bits'):
            if p not in (checkpoint,current['image']) and not p.is_relative_to(root/'target') and not any(p.is_relative_to(a) for a in libraries):
                raise PermissionError('unregistered model or data access')
    sys.addaudithook(audit)
    try:(root/'scorer/denial_probe.json').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('label guard did not reject')
    return current


def online(c,job,guard):
    assignment=json.loads(os.environ['RUN_ASSIGNMENT'])
    base.gpu_policy(assignment);base.available_memory(assignment,6*1024**3)
    root=Path(c['output_root'])
    if subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(root)],text=True).strip() not in ('nfs','nfs4'):
        raise OSError('expected network storage absent')
    if shutil.disk_usage(root).free<64*1024**3:raise OSError('storage reserve')
    rows=read(root/'private'/f'ONLINE_o{job["order"]}.json')
    points={p['visit']:p for p in read(root/'private/PLAN.json')[str(job['order'])]}
    smoke=job.get('smoke',False)
    if smoke:rows=rows[:2];points={1:dict(visit=1,domain='engineering')}
    permit=access_guard(c)
    h=Host(weights(c),dict(id='GREEDY',family='GREEDY',host='C',params={},components={}),job['seed'],job['id'])
    base.attach(guard.meter,h)
    reader=TargetReader(c['target_root'],256*1024**2,'image')
    dest=root/'target'/job['id'];dest.mkdir(exist_ok=False);probes=0
    try:
        with (dest/'probes.jsonl').open('x') as out:
            for i,row in enumerate(rows,1):
                guard();x=image(reader,row,permit,guard)
                if i in points:
                    torch.save(h.snapshot(),dest/f'before_{i}.pt')
                    masks,d=h.probe(x,actions_for(job['seed'],job['order'],i))
                    np.packbits(masks).tofile(dest/f'masks_{i}.bits')
                    out.write(json.dumps(dict(points[i],seed=job['seed'],order=job['order'],**d))+'\n');out.flush();probes+=1
                else:h.step(x)
                guard.extra['arrivals']=i;guard.extra['probes']=probes
            h.check_frozen(True)
            assert h.visits==len(rows) and probes==(1 if smoke else 16)
        return dict(arrivals=len(rows),probes=probes,label_reads=0,trajectory='ANCHOR',smoke=smoke)
    finally:h.close()


def score(c):
    root=Path(c['output_root']);jobs=c['jobs']
    processes=[read(root/'processes'/f'diag_{j["id"]}.json') for j in jobs]
    if any(p['active'] or p['exit_code']!=0 for p in processes):raise ValueError('GPU retirement barrier')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer required')
    save(root/'LABEL_RELEASE.json',dict(at=time.time(),all_workers_retired=True,SEARCH_only=True))
    reader=TargetReader(c['target_root'],256*1024**2,'mask');results=[]
    split=read(root/'scorer/SPLIT.private.json')
    for j in jobs:
        rows=read(root/'scorer'/f'FULL_o{j["order"]}.json')
        dest=root/'target'/j['id']
        for line in (dest/'probes.jsonl').read_text().splitlines():
            d=json.loads(line);visit=d['visit']
            if split.get(rows[visit-1]['image_sha256'])!='SEARCH':raise ValueError('non-SEARCH probe')
            label=reader.read(rows[visit-1])
            masks=np.unpackbits(np.fromfile(dest/f'masks_{visit}.bits',dtype=np.uint8)).reshape(5,2,512,512).astype(bool)
            dice=np.array([[m['dice'] for m in hard_metrics(mask,label)] for mask in masks])*100
            candidate=dice[1:].mean(1);best=int(candidate.argmax());selected=d['selected']
            result=dict(d,anchor_dice=dice[0].mean(),oracle_gain_pp=candidate[best]-dice[0].mean(),
                actual_gain_pp=candidate[selected]-dice[0].mean(),selection_regret_pp=candidate[best]-candidate[selected],
                random_gain_pp=candidate.mean()-dice[0].mean(),oracle_candidate=best,
                channel_deltas=(dice[1:]-dice[0]).tolist())
            assert abs(result['actual_gain_pp']-(result['oracle_gain_pp']-result['selection_regret_pp']))<1e-9
            results.append(result)
    assert len(results)==96
    save(root/'private/SCORES.json',results)
    groups=defaultdict(list)
    for r in results:groups[r['seed'],r['order'],r['domain']].append(r)
    keys=['oracle_gain_pp','actual_gain_pp','selection_regret_pp','random_gain_pp']
    cells=[dict(seed=s,order=o,domain=d,n=len(g),**{k:float(np.mean([r[k] for r in g])) for k in keys}) for (s,o,d),g in sorted(groups.items())]
    csvout(root/'public/COMMON_STATE_CELLS.csv',cells)
    channels=[]
    for (s,o,d),g in sorted(groups.items()):
        for channel,name in enumerate(('OD','OC')):
            channels.append(dict(seed=s,order=o,domain=d,channel=name,
                selected_gain_pp=float(np.mean([r['channel_deltas'][r['selected']][channel] for r in g])),
                oracle_joint_choice_gain_pp=float(np.mean([r['channel_deltas'][r['oracle_candidate']][channel] for r in g]))))
    csvout(root/'public/DOMAIN_CHANNEL.csv',channels)
    order=[float(np.mean([r['oracle_gain_pp'] for r in cells if r['order']==o])) for o in (0,1)]
    trajectories=[float(np.mean([r['oracle_gain_pp'] for r in cells if r['seed']==s and r['order']==o])) for s in c['seeds'] for o in (0,1)]
    summary={k:float(np.mean([r[k] for r in cells])) for k in keys}
    summary.update(states=96,orders_oracle_gain_pp=order,positive_oracle_trajectories=sum(x>0 for x in trajectories),
        proceed_to_C=summary['oracle_gain_pp']>=.3 and min(order)>0 and sum(x>0 for x in trajectories)>=5,
        selection='joint OD/OC oracle, actual first-argmax nesting selector',independent_confirmation=False)
    save(root/'public/DIAGNOSTIC_DECISION.json',summary)
    return summary


def worker():
    c=base.config();root=Path(c['output_root']);phase=os.environ['RUN_PHASE'];start=float(os.environ['RUN_STARTED'])
    guard=base.Guard(c,phase);failure=None;result=None;cost={};cpu=phase.startswith('score_')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    try:
        if cpu:result=score(c)
        else:
            with base.Meter(None,guard.observe) as meter:
                guard.meter=meter;result=online(c,json.loads(os.environ['RUN_JOB']),guard)
                cost=meter.cost.copy()
    except BaseException as e:
        failure=dict(type=type(e).__name__,reason=str(e));traceback.print_exc()
        if guard.meter:cost=guard.meter.cost.copy()
    cost.update(gpu_seconds=0 if cpu else time.time()-start,cpu_seconds=time.time()-start if cpu else 0)
    save(root/'attempts'/f'{phase}.0.json',dict(status='COMPLETE' if failure is None else 'FAILED',phase=phase,
         attempt=0,started=start,ended=time.time(),cost=cost,result=result,failure=failure,code_sha=c['code_sha']))
    return failure is None


def supervise():
    c=base.config();root=Path(c['output_root']);state=dict(status='PROFILING',jobs={j['id']:'NOT_RUN' for j in c['jobs']})
    def persist():save(root/'RUN_STATE.json',state);base.ledger(c,state)
    persist()
    try:
        for phase,jobs in [('profile',[dict(id=f'probe{i}',seed=17,order=0,smoke=True) for i in range(3)]),('diag',c['jobs'])]:
            state['status']='PROFILING' if phase=='profile' else 'RUNNING';persist()
            for offset in range(0,len(jobs),3):
                batch=jobs[offset:offset+3]
                if phase=='diag':
                    for j in batch:state['jobs'][j['id']]='RUNNING'
                    persist()
                with concurrent.futures.ThreadPoolExecutor(3) as pool:
                    futures={pool.submit(base.run_task,c,phase+'_'+j['id'],gpu,1800 if phase=='profile' else 86400,j):j for j,gpu in zip(batch,c['gpu_assignments'])}
                    failed=False
                    for f in concurrent.futures.as_completed(futures):
                        r=f.result();j=futures[f]
                        if phase=='diag':state['jobs'][j['id']]=r['status']
                        failed|=r['status']!='COMPLETE';persist()
                if failed:raise RuntimeError('profile/replay failed; no automatic scientific retry')
        state['status']='SCORING';persist()
        result=base.run_task(c,'score_all',None,3600,{})
        if result['status']!='COMPLETE':raise RuntimeError('scoring failed')
        state.update(status='COMPLETE',ended=time.time(),decision=result['result'],delivery='PENDING_GITHUB');persist()
    except BaseException as e:
        state.update(status='BLOCKED',reason=str(e));persist();traceback.print_exc()


def main():
    base.ID=ID
    mode=os.environ['RUN_MODE']
    if mode=='worker':sys.exit(0 if worker() else 1)
    elif mode=='supervise':supervise()
    elif mode=='watch':base.watch()
    else:raise ValueError('unknown mode')


if __name__=='__main__':main()
