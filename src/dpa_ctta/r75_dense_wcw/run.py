"""A bounded screen and conditional full-stream confirmation, with CPU-only labels."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
from .method import ARMS, Readouts, self_check
from ..r51_same_state.run import read, save

WIDTH = 65536


def jobs(stage):
    seeds = [20260907] if stage == 'screen' else [20260907, 17011, 29009]
    return [dict(id=f'{stage}_s{s}_o{o}',seed=s,order=o) for s in seeds for o in (0,1)]


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights, image
    from ..r19_model_only.method import equal
    from ..r37_cw_orientation.method import Host, candidates
    from ..r7_target_screen.runner import TargetReader, file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    stage=os.environ['STAGE']; slot=int(os.environ['SLOT']); root=Path(c['output_root'])
    start=time.time(); ident=f'{stage}_worker{slot}'; dest=root/'private'/ident
    dest.mkdir(); status=dict(status='QUALIFYING',arrivals=0,completed_jobs=[],started=start)
    def status_save():save(dest/'RUN_STATE.json',status)
    status_save(); host=None
    def guard():
        if time.time()>min(c['deadline']-300,float(os.environ['CHILD_DEADLINE']))-15:
            raise TimeoutError('bounded experiment deadline')
    guard.extra={'image_accesses':0}
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][slot])
        permit,opened,rejected=filesystem_guard(c);state=weights(c)
        reader=TargetReader(c['target_root'],256*1024**2,'image')
        assigned=jobs(stage) if stage=='screen' else jobs(stage)[slot::2]
        # Instrumented native output/state must equal the retained host, before any scored trajectory.
        row=read(root/'private'/f'{stage}_o{assigned[0]["order"]}.json')[0]
        pixels=image(reader,row,permit,guard)
        host=Host(state,candidates()[3],20260907,'qualification')
        initial=host.snapshot(); native,_=host.step(pixels); reference=host.snapshot()
        host.restore(initial); captured={}
        handle=host.native.model.seg_head.register_forward_hook(
            lambda m,a,out:captured.update(features=a[0].detach(),logits=out.detach()))
        repeated,_=host.step(pixels)
        if not torch.equal(native,repeated) or not equal(host.snapshot(),reference):
            raise ValueError('capture hook changed native output/state')
        if not torch.equal(repeated,captured['logits'].cpu()):raise ValueError('head/output interface mismatch')
        observer=Readouts(host.native.model.seg_head)
        for _ in range(25):outputs,details=observer.emit(captured['features'],captured['logits'])
        if not equal(host.snapshot(),reference):raise ValueError('readout changed native state/RNG')
        if not torch.equal(outputs['NATIVE'].cpu(),native):raise ValueError('native readout parity')
        save(dest/'QUALIFICATION.json',dict(status='PASS',native_output_exact=True,native_state_exact=True,
             readout_state_rng_exact=True,head_channels=host.native.model.seg_head.in_channels,
             qualification_repeated_measurements=25,qualification_not_formal=True,online_label_reads=0))
        handle.remove();host.check_frozen(True);host.close();host=None;captured.clear();del observer,outputs
        status['status']='RUNNING';status_save()
        for job in assigned:
            guard();rows=read(root/'private'/f'{stage}_o{job["order"]}.json')
            if len(rows)!=(192 if stage=='screen' else 1951):raise ValueError('fixed cohort length')
            jobroot=root/'jobs'/job['id'];jobroot.mkdir()
            host=Host(state,candidates()[3],job['seed'],job['id']);observer=Readouts(host.native.model.seg_head)
            captured={}
            handle=host.native.model.seg_head.register_forward_hook(
                lambda m,a,out:captured.update(features=a[0].detach(),logits=out.detach()))
            durations=[]
            with (jobroot/'predictions.bits').open('xb') as bits,(jobroot/'diagnostics.jsonl').open('x') as trace:
                for i,row in enumerate(rows):
                    guard();pixels=image(reader,row,permit,guard);tick=time.time()
                    native,detail=host.step(pixels)
                    if not torch.equal(native,captured['logits'].cpu()):raise ValueError('head capture is not final native prediction')
                    outputs,mechanism=observer.emit(captured['features'],captured['logits'])
                    torch.cuda.synchronize();duration=time.time()-tick;durations.append(duration)
                    masks=torch.stack([outputs[a][0]>=0 for a in ARMS]).cpu().numpy()
                    bits.write(np.packbits(masks).tobytes())
                    trace.write(json.dumps(dict(visit=i+1,seconds=duration,readouts=mechanism,
                         foreground_fraction=masks.mean(axis=(2,3)).tolist(),
                         native_update_norm=detail['diagnostics']['BN_update_norm']),allow_nan=False)+'\n')
                    status['arrivals'] += 1
                    if (i+1)%16==0:
                        bits.flush();trace.flush();status.update(job=job['id'],visit=i+1,
                            latest_seconds=duration);status_save()
                if bits.tell()!=len(rows)*len(ARMS)*WIDTH:raise ValueError('prediction byte coverage')
            handle.remove();host.check_frozen(True)
            # Retain state sufficient for a later authorized analysis, without a float prediction bank.
            torch.save(dict(host=host.snapshot(),moments=observer.moments.__dict__),jobroot/'terminal.private.pt')
            host.close();host=None;captured.clear();del observer,outputs
            save(jobroot/'COMPLETE.json',dict(**job,arrivals=len(rows),readouts=len(ARMS),
                 seconds=sum(durations),p95_step_seconds=float(np.quantile(durations[24:],.95)),online_label_reads=0))
            status['completed_jobs'].append(job['id']);status_save()
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input changed')
        save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,
             image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),rejections=rejected))
        status['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',error_type=type(e).__name__,reason=str(e))
        traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        status.update(ended=time.time(),gpu_process_seconds=time.time()-start);status_save()


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import interval, csvout
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only scorer')
    root=Path(c['output_root']);stage=os.environ['STAGE']; start=time.time();planned=jobs(stage)
    slots=[0] if stage=='screen' else [0,1]
    for slot in slots:
        process=read(root/'processes'/f'{stage}_worker{slot}.json')
        state=read(root/'private'/f'{stage}_worker{slot}'/'RUN_STATE.json')
        if process['active'] or process['exit_code']!=0 or state['status']!='GPU_COMPLETE':
            raise ValueError('all stage GPU workers must retire successfully before labels')
    panel=read(root/'scorer'/f'{stage}_o0.json');ids=[r['image_sha256'] for r in panel]
    index={v:i for i,v in enumerate(ids)}
    if len(index)!=len(ids):raise ValueError('unique paired contents required')
    domains=np.array([r['domain'] for r in panel]);names=sorted(set(domains))
    anonymous={d:f'D{i}' for i,d in enumerate(names)}
    reader=TargetReader(c['target_root'],256*1024**2,'mask')
    labels={r['image_sha256']:reader.read(r) for r in panel}
    values=np.empty((len(panel),len(planned),len(ARMS),2));diagnostics=[]
    for j,job in enumerate(planned):
        rows=read(root/'scorer'/f'{stage}_o{job["order"]}.json');folder=root/'jobs'/job['id']
        receipt=read(folder/'COMPLETE.json')
        if receipt['arrivals']!=len(panel):raise ValueError('incomplete trajectory')
        traces=[json.loads(v) for v in (folder/'diagnostics.jsonl').read_text().splitlines()]
        if [v['visit'] for v in traces]!=list(range(1,len(panel)+1)):raise ValueError('trace coverage')
        with (folder/'predictions.bits').open('rb') as f:
            for row in rows:
                raw=f.read(len(ARMS)*WIDTH)
                if len(raw)!=len(ARMS)*WIDTH:raise ValueError('truncated predictions')
                masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(len(ARMS),2,512,512).astype(bool)
                for a in range(len(ARMS)):
                    values[index[row['image_sha256']],j,a]=[100*m['dice'] for m in hard_metrics(masks[a],labels[row['image_sha256']])]
            if f.read(1):raise ValueError('extra predictions')
        for a,arm in enumerate(ARMS[1:],1):
            for ch,channel in enumerate(('OD','OC')):
                active=[t['readouts'][arm][ch] for t in traces if t['readouts'][arm][ch]['active']]
                diagnostics.append(dict(seed=job['seed'],order=job['order'],arm=arm,channel=channel,
                    active_arrivals=len(active),mean_gamma=float(np.mean([x['gamma'] for x in active])) if active else None,
                    mean_operator_change=float(np.mean([x['operator_change'] for x in active])) if active else None,
                    mean_logit_shift=float(np.mean([x['sampled_corrected_mean_logit']-x['sampled_native_mean_logit'] for x in active])) if active else None,
                    native_fg_fraction=float(np.mean([t['foreground_fraction'][0][ch] for t in traces])),
                    corrected_fg_fraction=float(np.mean([t['foreground_fraction'][a][ch] for t in traces]))))
    reader.after_check();np.save(root/'scorer'/f'{stage}.private.npy',values)
    split=read(root/'scorer/SPLIT.private.json') if stage!='screen' else None
    roles={'SEARCH':np.ones(len(panel),dtype=bool)} if stage=='screen' else {
        role:np.array([split[v]==role for v in ids]) for role in ('SEARCH','SEALED_REVIEW')}
    summaries=[];cells=[];scores=[]
    for role,select in roles.items():
        for a,arm in enumerate(ARMS):
            scores.append(dict(role=role,arm=arm,macro_Dice_percent=float(np.mean([
                values[select&(domains==d),:,a].mean() for d in names])),imageweighted_Dice_percent=float(values[select,:,a].mean())))
        comparisons=[(a,0) for a in range(1,len(ARMS))]+[(2,1),(3,2)]
        for a,b in comparisons:
            delta=values[:,:,a]-values[:,:,b];name=ARMS[a]+'-'+ARMS[b]
            for d in names:
                for o in (0,1):
                    jobselect=np.array([x['order']==o for x in planned])
                    for ch,channel in enumerate(('OD','OC')):
                        cells.append(dict(role=role,contrast=name,domain=anonymous[d],order=o,channel=channel,
                            n_contents=int((select&(domains==d)).sum()),
                            delta_pp=float(delta[select&(domains==d)][:,jobselect,ch].mean())))
            trajectory=np.mean([delta[select&(domains==d)].mean((0,2)) for d in names],axis=0)
            summaries.append(dict(role=role,contrast=name,macro_delta_pp=float(trajectory.mean()),
                ci95_pp=interval(delta[select],domains[select]),
                order_delta_pp=[float(trajectory[[j for j,x in enumerate(planned) if x['order']==o]].mean()) for o in (0,1)],
                positive_trajectories=int((trajectory>0).sum()),trajectories=len(planned),
                imageweighted_delta_pp=float(delta[select].mean()),
                worst_cell_pp=min(r['delta_pp'] for r in cells if r['role']==role and r['contrast']==name)))
    def passes(s):
        return (s['macro_delta_pp']>=.2 and s['ci95_pp'][0]>0 and min(s['order_delta_pp'])>0
                and s['imageweighted_delta_pp']>=0 and s['worst_cell_pp']>=-2
                and s['positive_trajectories']>=len(planned)-1)
    gates={arm:passes(next(x for x in summaries if x['role']=='SEARCH' and x['contrast']==arm+'-NATIVE'))
           for arm in ('WITHIN_STREAM','WITHIN_IMAGE')}
    selected=next((a for a in ('WITHIN_STREAM','WITHIN_IMAGE') if gates[a]),None)
    result=dict(status='COMPLETE',stage=stage,primary='WITHIN_STREAM',selected_for_followup=selected,
        allocation_gates=gates,contrasts=summaries,contents=len(panel),native_trajectories=len(planned),
        readout_predictions=len(panel)*len(planned)*len(ARMS),independent_validation=False,
        shared_native_state=True,readout_feedback_to_learning=False,cpu_seconds=time.time()-start)
    out=root/'public';save(out/f'{stage.upper()}_RESULTS.json',result)
    csvout(out/f'{stage.upper()}_ARM_SCORES.csv',scores);csvout(out/f'{stage.upper()}_ALL_CELLS.csv',cells)
    csvout(out/f'{stage.upper()}_NEGATIVE_CELLS.csv',[x for x in cells if x['delta_pp']<0])
    csvout(out/f'{stage.upper()}_MECHANISM.csv',diagnostics)
    save(out/f'{stage.upper()}_COMPLETION.json',dict(gpu_workers_retired=True,online_label_reads=0,
        native_trajectories=len(planned),scored_predictions=result['readout_predictions']))
    return 0


def child(c,mode,stage,slot,cap):
    root=Path(c['output_root']);name=f'{stage}_{mode}{slot}';start=time.time()
    if mode=='worker':
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                     '--format=csv,noheader,nounits'],text=True)
        inventory={int(i):(u.strip(),int(m)) for i,u,m in (line.split(',') for line in raw.splitlines())}
        gpu=c['gpus'][slot];uuid,free=inventory[gpu['physical_id']]
        if gpu['physical_id'] not in (0,1) or uuid!=gpu['uuid'] or free<6144:
            raise RuntimeError('GPU memory/UUID admission failed; no other GPU fallback')
        save(root/'private'/f'{name}_ADMISSION.json',dict(gpu=gpu['physical_id'],uuid=uuid,free_MiB=free,at=start))
    env=dict(os.environ,MODE=mode,STAGE=stage,SLOT=str(slot),CHILD_DEADLINE=str(min(start+cap,c['deadline']-240)),
             CUDA_VISIBLE_DEVICES=str(c['gpus'][slot]['physical_id']) if mode=='worker' else '')
    log=(root/'logs'/f'{name}.log').open('x')
    process=subprocess.Popen([sys.executable,os.environ['ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=process.pid,active=True,started=start,mode=mode,stage=stage,slot=slot)
    save(root/'processes'/f'{name}.json',record)
    return dict(process=process,record=record,name=name,log=log,cap=cap)


def finish_child(c,item):
    process=item['process'];record=item['record'];root=Path(c['output_root'])
    elapsed=time.time()-record['started']
    if process.poll() is None and (elapsed>item['cap'] or time.time()>c['deadline']-240):
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    if process.poll() is None:return None
    item['log'].close();record.update(active=False,ended=time.time(),exit_code=process.returncode)
    save(root/'processes'/f'{item["name"]}.json',record)
    receipt=dict(name=item['name'],status='COMPLETE' if process.returncode==0 else 'FAILED',exit_code=process.returncode,
        seconds=elapsed,gpu_process_seconds=elapsed if record['mode']=='worker' else 0,
        cpu_process_seconds=elapsed if record['mode']!='worker' else 0)
    save(root/'attempts'/f'{item["name"]}.json',receipt)
    return receipt


def controller(c):
    root=Path(c['output_root']);attempts=[];status=dict(status='SCREEN',started=time.time(),source_commit=c['source_commit'])
    with (root/'START_ONCE.json').open('x') as f:json.dump(status,f)
    def publish():
        save(root/'RUN_STATE.json',status)
        save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,
             gpu_process_seconds=sum(x['gpu_process_seconds'] for x in attempts),
             cpu_process_seconds=c['preparation_cpu_seconds']+sum(x['cpu_process_seconds'] for x in attempts),
             gpu_cap_seconds=c['gpu_cap_seconds'],all_workers_retired=all(not read(p)['active'] for p in (root/'processes').glob('*.json'))))
    def run_group(mode,stage,slots,cap):
        active=[child(c,mode,stage,s,cap) for s in slots]
        while active:
            for item in list(active):
                receipt=finish_child(c,item)
                if receipt is not None:attempts.append(receipt);active.remove(item);publish()
            if active:time.sleep(5)
        return all(x['exit_code']==0 for x in attempts if x['name'].startswith(stage+'_'+mode))
    try:
        if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU admission missing')
        publish()
        if not run_group('worker','screen',[0],1800):raise RuntimeError('screen worker failed')
        if not run_group('score','screen',[0],600):raise RuntimeError('screen scorer failed')
        result=read(root/'public/SCREEN_RESULTS.json')
        if result['selected_for_followup'] is None:
            status.update(status='SCREEN_COMPLETE_NO_SIGNAL',scientific_followup=False);publish();return 0
        remaining=c['gpu_cap_seconds']-sum(x['gpu_process_seconds'] for x in attempts)
        receipts=[read(root/'jobs'/j['id']/'COMPLETE.json') for j in jobs('screen')]
        projection=max(x['p95_step_seconds'] for x in receipts)*1951*6*1.25+180
        admitted=(projection<remaining and time.time()+projection/2+900<c['deadline'])
        save(root/'FULL_ADMISSION.json',dict(admitted=admitted,projected_gpu_seconds=projection,
             remaining_gpu_seconds=remaining,selected_from_screen=result['selected_for_followup']))
        if not admitted:
            status.update(status='SCREEN_COMPLETE_FULL_BUDGET_NOT_ADMITTED');publish();return 0
        status['status']='FULL_STREAM';publish()
        if not run_group('worker','full',[0,1],min(remaining/2, c['deadline']-time.time()-600)):
            raise RuntimeError('full worker failed')
        if not run_group('score','full',[0],900):raise RuntimeError('full scorer failed')
        status.update(status='COMPLETE',scientific_followup=False);publish();return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:
        status['ended']=time.time();publish()


def main():
    mode=os.environ['MODE']
    if mode=='self_check':
        self_check()
        from ..b1_host import official
        from ..integrations.ctta_suite import load_reference_interfaces
        from ..r19_model_only.runtime import weights, image
        from ..r37_cw_orientation.method import Host, candidates
        from ..r7_target_screen.runner import TargetReader, file_identity
        from ..r9_current_first.assets import gpu_policy
        from ..r36_incremental_modules.run import filesystem_guard
        official();load_reference_interfaces()
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU check initialized CUDA')
        print('FULL_ENTRY_IMPORT_CHECK_PASS',flush=True)
        return 0
    c=read(os.environ['CONFIG'])
    if mode=='worker':return worker(c)
    if mode=='score':return score(c)
    if mode=='controller':return controller(c)
    raise ValueError('unknown mode')


if __name__=='__main__':raise SystemExit(main())
