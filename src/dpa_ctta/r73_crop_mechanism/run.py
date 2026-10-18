"""Fixed late-anchor diagnostic; labels are read only after online retirement."""
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .diagnostic import Host, configuration, pair, self_check, ARMS, BRANCHES
from ..r51_same_state.run import read, save, load_anchors, run_child


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights,image
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    root=Path(c['output_root']);start=time.time();host=None
    status=dict(status='QUALIFYING',formal_pairs=0,started=start)
    save(root/'RUN_STATE.json',status)
    def guard():
        if time.time()-start>c['gpu_cap_seconds']-30: raise TimeoutError('registered GPU budget')
    guard.extra={'image_accesses':0}
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpu'])
        permit,opened,rejected=filesystem_guard(c)
        state=weights(c);anchors=load_anchors(c)
        rows=read(root/'private/images.json');qualification=read(root/'private/qualification.json')
        if len(rows)!=48 or len(anchors)!=2 or len(qualification)!=2:raise ValueError('fixed cohort coverage')
        reader=TargetReader(c['target_root'],256*1024**2,'image');checks=[]
        for order,anchor in enumerate(anchors):
            host=Host(state,configuration('NATIVE'),c['seed'],anchor['context']['sha256'])
            x=image(reader,qualification[order],permit,guard);tick=time.time()
            masks,targets,details=pair(host,anchor,x,qualify=True);torch.cuda.synchronize()
            old=np.unpackbits(np.frombuffer((root/'private'/f'qualification_o{order}.bits').read_bytes(),dtype=np.uint8)).reshape(2,512,512)
            if not np.array_equal(masks[1],old):raise ValueError('historical native qualification differs')
            checks.append(dict(order=order,seconds=time.time()-tick,repeated_full_state_exact=True,
                               compute_only_full_state_exact=True,native_weak_targets_exact=True,
                               native_rng_exact=True,historical_native_mask_exact=True))
            host.close();host=None;guard()
        estimate=max(q['seconds'] for q in checks)*96*1.5
        if time.time()-start+estimate>c['gpu_cap_seconds']-30:raise TimeoutError('formal diagnostic not admitted')
        save(root/'QUALIFICATION.json',dict(status='PASS',checks=checks,estimated_formal_seconds=estimate))
        status['status']='RUNNING_FIXED_PAIRS';save(root/'RUN_STATE.json',status)
        with (root/'private/pairs.jsonl').open('x') as trace:
            for order,anchor in enumerate(anchors):
                host=Host(state,configuration('NATIVE'),c['seed'],anchor['context']['sha256'])
                for i,row in enumerate(rows):
                    guard();x=image(reader,row,permit,guard);tick=time.time()
                    masks,targets,details=pair(host,anchor,x);torch.cuda.synchronize();index=order*48+i
                    with (root/'predictions'/f'{index:03d}.bits').open('xb') as f:f.write(np.packbits(masks).tobytes())
                    with (root/'predictions'/f'{index:03d}.npy').open('xb') as f:np.save(f,targets,allow_pickle=False)
                    trace.write(json.dumps(dict(pair=index,seconds=time.time()-tick,diagnostics=details),allow_nan=False)+'\n');trace.flush()
                    status['formal_pairs']+=1
                    if status['formal_pairs']%8==0:save(root/'RUN_STATE.json',status)
                host.close();host=None
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input changed')
        save(root/'private/ACCESS_AUDIT.json',dict(label_reads=0,image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),rejected=rejected))
        status['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e));traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        status.update(ended=time.time(),worker_seconds=time.time()-start);save(root/'RUN_STATE.json',status)


def target_statistics(q,y):
    from scipy.ndimage import maximum_filter,minimum_filter
    edge=maximum_filter(y,size=(1,7,7),mode='constant',cval=0)!=minimum_filter(y,size=(1,7,7),mode='constant',cval=0)
    regions=(np.ones_like(y),y,edge)
    if any(not mask[ch].any() for mask in regions for ch in (0,1)):
        raise ValueError('empty registered scoring region; no imputation')
    brier=np.empty((3,2,3));corrected=np.empty_like(brier);spoiled=np.empty_like(brier)
    for b in range(3):
        error=(q[b].astype(np.float64)-y)**2
        for ch in (0,1):
            for r,mask in enumerate(regions):brier[b,ch,r]=error[ch][mask[ch]].mean()
    for k,(a,b) in enumerate(((2,0),(2,1),(1,0))):
        old=(q[b]>=.5)==y;new=(q[a]>=.5)==y
        for ch in (0,1):
            for r,mask in enumerate(regions):
                corrected[k,ch,r]=(~old[ch]&new[ch])[mask[ch]].mean()
                spoiled[k,ch,r]=(old[ch]&~new[ch])[mask[ch]].mean()
    return brier,corrected,spoiled


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import interval,csvout
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer required')
    root=Path(c['output_root']);status=read(root/'RUN_STATE.json');retired=read(root/'processes/worker.json')
    if status['status']!='GPU_COMPLETE' or status['formal_pairs']!=96 or retired['active'] or retired['exit_code']!=0:
        raise ValueError('complete successful worker retirement required')
    panel=read(root/'scorer/panel.json');traces=[json.loads(s) for s in (root/'private/pairs.jsonl').read_text().splitlines()]
    if len(panel)!=48 or [t['pair'] for t in traces]!=list(range(96)):raise ValueError('paired output coverage')
    domains=np.array([r['domain'] for r in panel]);names=sorted(set(domains))
    if len(names)!=4 or any(sum(domains==d)!=12 for d in names):raise ValueError('fixed balanced panel')
    reader=TargetReader(c['target_root'],256*1024**2,'mask')
    dice=np.empty((48,2,4,2));tdice=np.empty((48,2,3,2));brier=np.empty((48,2,3,2,3))
    corrected=np.empty((48,2,3,2,3));spoiled=np.empty_like(corrected)
    contrasts=(('CROP-NATIVE',2,0),('CROP-RESAMPLE',2,1),('RESAMPLE-NATIVE',1,0))
    region_names=('all','foreground','boundary_r3')
    for i,row in enumerate(panel):
        label=reader.read(row);y=label.numpy()[0].astype(bool)
        for order in (0,1):
            index=order*48+i;raw=(root/'predictions'/f'{index:03d}.bits').read_bytes()
            if len(raw)!=4*65536:raise ValueError('mask coverage')
            masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(4,2,512,512).astype(bool)
            q=np.load(root/'predictions'/f'{index:03d}.npy',allow_pickle=False)
            if q.shape!=(3,2,512,512) or q.dtype!=np.float32 or not np.isfinite(q).all() or q.min()<0 or q.max()>1:
                raise ValueError('soft target coverage or range')
            for b,m in enumerate(masks):dice[i,order,b]=[100*v['dice'] for v in hard_metrics(m,label)]
            for b in range(3):
                tdice[i,order,b]=[100*v['dice'] for v in hard_metrics(q[b]>=.5,label)]
            brier[i,order],corrected[i,order],spoiled[i,order]=target_statistics(q,y)
    reader.after_check()
    np.savez(root/'scorer/metrics.private.npz',update_dice=dice,target_dice=tdice,brier=brier,corrected=corrected,spoiled=spoiled)
    target_summaries=[];target_cells=[];update_summaries=[];update_cells=[]
    def summary(delta):return dict(mean=float(delta.mean()),ci95=interval(delta,domains),anchor_mean=delta.mean((0,2)).tolist())
    for k,(name,a,b) in enumerate(contrasts):
        for r,region in enumerate(region_names):
            delta=brier[:,:,b,:,r]-brier[:,:,a,:,r]
            s=summary(delta);s.update(contrast=name,region=region,corrected_fraction=float(corrected[:,:,k,:,r].mean()),spoiled_fraction=float(spoiled[:,:,k,:,r].mean()))
            target_summaries.append(s)
            for order in (0,1):
                for domain in names:
                    for ch,channel in enumerate(('OD','OC')):
                        ix=domains==domain
                        target_cells.append(dict(contrast=name,region=region,order=order,domain=domain,channel=channel,
                            brier_reduction=float(delta[ix,order,ch].mean()),corrected_fraction=float(corrected[ix,order,k,ch,r].mean()),spoiled_fraction=float(spoiled[ix,order,k,ch,r].mean())))
        for endpoint,delta in (('update_Dice_pp',dice[:,:,a+1]-dice[:,:,b+1]),('target_Dice_pp',tdice[:,:,a]-tdice[:,:,b])):
            s=summary(delta);s.update(contrast=name,endpoint=endpoint);update_summaries.append(s)
            for order in (0,1):
                for domain in names:
                    for ch,channel in enumerate(('OD','OC')):
                        update_cells.append(dict(contrast=name,endpoint=endpoint,order=order,domain=domain,channel=channel,delta_pp=float(delta[domains==domain,order,ch].mean())))
    for b,arm in enumerate(ARMS):
        s=summary(dice[:,:,b+1]-dice[:,:,0]);s.update(contrast=arm+'-NO_UPDATE',endpoint='update_Dice_pp');update_summaries.append(s)
    def positive(s):return s['mean']>0 and s['ci95'][0]>0 and min(s['anchor_mean'])>0
    target_gate=all(positive(s) for s in target_summaries if s['contrast'] in ('CROP-NATIVE','CROP-RESAMPLE'))
    primary=[s for s in update_summaries if s['endpoint']=='update_Dice_pp' and s['contrast'] in ('CROP-NATIVE','CROP-RESAMPLE')]
    update_gate=all(positive(s) and s['mean']>=.05 for s in primary) and min(x['delta_pp'] for x in update_cells if x['endpoint']=='update_Dice_pp' and x['contrast']=='CROP-NATIVE')>=-.5
    out=root/'public';result=dict(status='COMPLETE',target_quality_supported=bool(target_gate),one_step_supported=bool(update_gate),joint_mechanism_supported=bool(target_gate and update_gate),does_not_reverse_R72=True,independent_validation=False,target_brier_reduction=target_summaries,dice_contrasts=update_summaries)
    save(out/'RESULTS.json',result);csvout(out/'TARGET_CELLS.csv',target_cells);csvout(out/'DICE_CELLS.csv',update_cells)
    csvout(out/'UPDATE_DIAGNOSTICS.csv',[dict(arm=arm,mean_update_norm=float(np.mean([t['diagnostics'][arm]['update_norm'] for t in traces])),mean_gradient_norm=float(np.mean([t['diagnostics'][arm]['gradient_norm'] for t in traces])),mean_cosine_to_native=float(np.mean([t['diagnostics'][arm]['cosine_to_native'] for t in traces if t['diagnostics'][arm]['cosine_to_native'] is not None]))) for arm in ARMS])
    save(out/'QUALIFICATION.json',read(root/'QUALIFICATION.json'))
    save(out/'COMPLETION_AUDIT.json',dict(pairs=96,contents=48,late_anchors=2,scored_predictions=384,scored_soft_targets=288,online_label_reads=0,all_workers_retired=True))
    status.update(status='COMPLETE',scored_predictions=384,scored_soft_targets=288,ended=time.time());save(root/'RUN_STATE.json',status)


def main():
    mode=os.environ['DIAG_MODE']
    if mode=='self_check':self_check();return 0
    c=read(os.environ['DIAG_CONFIG']);root=Path(c['output_root'])
    if mode=='worker':return worker(c)
    if mode=='score':score(c);return 0
    if mode!='controller':raise ValueError('unknown mode')
    with (root/'started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU check missing')
    attempts=[]
    try:
        attempts.append(run_child(c,'worker',c['gpu_cap_seconds'],True))
        if attempts[-1]['exit_code']==0:attempts.append(run_child(c,'score',c['cpu_cap_seconds'],False))
    finally:
        ledger=dict(attempts=attempts,gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),cpu_worker_seconds=c['prior_cpu_seconds']+sum(a['cpu_worker_seconds'] for a in attempts),preparation_cpu_seconds=c['prior_cpu_seconds'],all_workers_retired=True)
        save(root/'RESOURCE_LEDGER.json',ledger);save(root/'public/RESOURCE_LEDGER.json',ledger)
    return 0 if len(attempts)==2 and all(a['exit_code']==0 for a in attempts) else 1


if __name__=='__main__':raise SystemExit(main())
