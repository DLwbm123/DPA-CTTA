"""Reuse R28 frozen states and scheduler; no new full trajectory or reward fitting."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from collections import defaultdict
import numpy as np
import torch
from .method import Host,ARMS
from ..r28_common_state import run as previous
from ..r20_model_only_search import runtime as base
from ..r20_model_only_search.report import csvout
from ..r10_12h_core.run import read,save
from ..r7_target_screen.runner import TargetReader
from ..r19_model_only.runtime import weights,image
from ..r19_model_only.method import equal
from ..r24_c_context.run import hard_metrics


def online(c,job,guard):
    assignment=json.loads(os.environ['RUN_ASSIGNMENT']);base.gpu_policy(assignment);base.available_memory(assignment,6*1024**3)
    root=Path(c['output_root'])
    if subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(root)],text=True).strip() not in ('nfs','nfs4') or shutil.disk_usage(root).free<64*1024**3:
        raise OSError('network storage/reserve unavailable')
    source_job=c['jobs'][0] if job.get('smoke') else job
    points=read(root/'private/PLAN.json')[str(source_job['order'])]
    if job.get('smoke'):points=points[:1]
    rows=read(root/'private'/f'ONLINE_o{source_job["order"]}.json')
    source=Path(c['snapshot_input_root'])/source_job['id']
    permit=previous.access_guard(c)
    h=Host(weights(c),dict(id='GREEDY',family='GREEDY',host='C',params={},components={}),source_job['seed'],source_job['id'])
    base.attach(guard.meter,h);reader=TargetReader(c['target_root'],256*1024**2,'image')
    dest=root/'target'/job['id'];dest.mkdir(exist_ok=False)
    try:
        with (dest/'probes.jsonl').open('x') as out:
            for point in points:
                guard();visit=point['visit']
                # These are trusted, internally generated R28 snapshots, never user pickle files.
                state=torch.load(source/f'before_{visit}.pt',map_location='cpu',weights_only=False)
                h.restore(state);x=image(reader,rows[visit-1],permit,guard)
                masks=h.probe_strength(x)
                expected=np.unpackbits(np.fromfile(source/f'masks_{visit}.bits',dtype=np.uint8)).reshape(5,2,512,512)[0].astype(bool)
                if not np.array_equal(masks[0],expected):raise ValueError('FULL failed exact R28 mask replay')
                if not equal(state,h.snapshot()):raise ValueError('diagnostic changed input snapshot')
                np.packbits(masks).tofile(dest/f'masks_{visit}.bits')
                out.write(json.dumps(dict(point,seed=source_job['seed'],order=source_job['order'],FULL_replay_exact=True))+'\n');out.flush()
                guard.extra['probes']+=1
            h.check_frozen(True)
        return dict(probes=len(points),FULL_replay_exact=True,label_reads=0,source='frozen R28 ANCHOR states',smoke=bool(job.get('smoke')))
    finally:h.close()


def score(c):
    root=Path(c['output_root']);jobs=c['jobs']
    processes=[read(root/'processes'/f'diag_{j["id"]}.json') for j in jobs]
    if any(p['active'] or p['exit_code']!=0 for p in processes):raise ValueError('worker retirement barrier')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer required')
    save(root/'LABEL_RELEASE.json',dict(at=time.time(),all_workers_retired=True,SEARCH_only=True))
    split=read(root/'scorer/SPLIT.private.json');reader=TargetReader(c['target_root'],256*1024**2,'mask');results=[]
    for j in jobs:
        rows=read(root/'scorer'/f'FULL_o{j["order"]}.json');dest=root/'target'/j['id']
        for line in (dest/'probes.jsonl').read_text().splitlines():
            d=json.loads(line);visit=d['visit']
            if split.get(rows[visit-1]['image_sha256'])!='SEARCH':raise ValueError('non-SEARCH state')
            label=reader.read(rows[visit-1]);masks=np.unpackbits(np.fromfile(dest/f'masks_{visit}.bits',dtype=np.uint8)).reshape(3,2,512,512).astype(bool)
            dice=np.array([[m['dice'] for m in hard_metrics(mask,label)] for mask in masks])*100
            delta=dice.mean(1)-dice[0].mean();oracle=int(delta.argmax())
            results.append(dict(d,zero_gain_pp=float(delta[1]),half_gain_pp=float(delta[2]),oracle_gain_pp=float(delta[oracle]),
                oracle_arm=ARMS[oracle],channel_deltas=(dice-dice[0]).tolist()))
    if len(results)!=96:raise ValueError('96-state coverage')
    save(root/'private/SCORES.json',results);groups=defaultdict(list)
    for r in results:groups[r['seed'],r['order'],r['domain']].append(r)
    keys=('zero_gain_pp','half_gain_pp','oracle_gain_pp')
    cells=[dict(seed=s,order=o,domain=d,n=len(g),**{k:float(np.mean([r[k] for r in g])) for k in keys}) for (s,o,d),g in sorted(groups.items())]
    csvout(root/'public/COMMON_STATE_CELLS.csv',cells)
    channels=[dict(seed=s,order=o,domain=d,arm=arm,channel=name,delta_pp=float(np.mean([r['channel_deltas'][i][k] for r in g]))) for (s,o,d),g in sorted(groups.items()) for i,arm in enumerate(ARMS[1:],1) for k,name in enumerate(('OD','OC'))]
    csvout(root/'public/DOMAIN_CHANNEL.csv',channels)
    summary={k:float(np.mean([r[k] for r in cells])) for k in keys}
    order=[float(np.mean([r['oracle_gain_pp'] for r in cells if r['order']==o])) for o in (0,1)]
    trajectory=[float(np.mean([r['oracle_gain_pp'] for r in cells if r['seed']==s and r['order']==o])) for s in c['seeds'] for o in (0,1)]
    summary.update(states=96,order_oracle_gain_pp=order,positive_oracle_trajectories=sum(v>0 for v in trajectory),
        headroom_gate=summary['oracle_gain_pp']>=.3 and min(order)>0 and sum(v>0 for v in trajectory)>=5,
        oracle_tie_order=list(ARMS),includes_full_in_oracle=True,independent_confirmation=False)
    save(root/'public/DIAGNOSTIC_DECISION.json',summary)
    return summary


def main():
    previous.ID='R29_UPDATE_STRENGTH_DIAGNOSTIC';previous.online=online;previous.score=score
    previous.main()


if __name__=='__main__':main()
