"""1152 new updates, two reused frozen endpoints, bounded detached execution."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from .method import Host,ARMS,SEEDS,CELLS,style,self_check,reference_criterion,NEXT
from ..r51_same_state.run import read,save
from .process import child,finish_child

WIDTH=65536
QUALIFICATION_UPDATES=8


def jobs(slot):
    return [dict(id=f'c{cell}_s{s}_o{slot}_{a}',cell=cell,seed=s,order=slot,arm=a)
            for s in SEEDS for cell in CELLS for a in ARMS]


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights,image
    from ..r19_model_only.method import equal
    from ..r79_mechanism_control.method import Host as Previous
    from ..r79_mechanism_control.method import supplied_style
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    stage=os.environ['STAGE'];slot=int(os.environ['SLOT']);root=Path(c['output_root']);started=time.time()
    dest=root/'private'/f'{stage}_worker{slot}';dest.mkdir();host=None;reference=None
    state=dict(status='QUALIFYING' if stage=='qualify' else 'RUNNING',started=started,arrivals=0,
               physical_updates=0,completed_physical_updates=0,physical_forwards=0,completed_jobs=[])
    def persist():save(dest/'RUN_STATE.json',state)
    def charge(kind):state[{'attempt':'physical_updates','complete':'completed_physical_updates','forward':'physical_forwards'}[kind]]+=1
    def meter(h):h.handles.append(h.native.model.register_forward_hook(lambda *args:charge('forward')))
    def guard():
        if time.time()>float(os.environ['CHILD_DEADLINE'])-20:raise TimeoutError('frozen process deadline')
    def storage_guard():
        used=sum(p.stat().st_size for p in (root/'jobs').glob('*/*') if p.is_file())
        if used>c['output_cap_bytes']-64*1024**2:raise RuntimeError('frozen output cap')
    guard.extra={'image_accesses':0};persist()
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][slot])
        torch.cuda.set_per_process_memory_fraction(1/3,0)
        permit,opened,rejected=filesystem_guard(c);checkpoint=weights(c)
        reader=TargetReader(c['target_root'],256*1024**2,'image')
        canonical=read(root/'private/screen_o0.json');content={r['image_sha256']:i for i,r in enumerate(canonical)}
        if len(content)!=192:raise ValueError('frozen unique cohort')
        if stage=='qualify':
            xs=[image(reader,r,permit,guard) for r in canonical[:2]];checks={};steady={};cold={}
            for arm in ARMS:
                for collapse in (True,False):
                    host=Host(checkpoint,arm,SEEDS[0],'qualification',charge=charge);meter(host)
                    reference=Previous(checkpoint,arm,SEEDS[0],'qualification',charge=charge);meter(reference)
                    for i,x in enumerate(xs):
                        raw=style(x,SEEDS[0],i);alternative=raw if collapse else style(x,NEXT[SEEDS[0]],i)
                        saved=host.snapshot();host.before(x)
                        if not equal(saved,host.snapshot()):raise ValueError('PRE changed complete state')
                        tick=time.time();forwards=state['physical_forwards'];z,detail=host.intervention_step(x,raw,alternative)
                        torch.cuda.synchronize();duration=time.time()-tick
                        if not collapse:(cold if i==0 else steady)[arm]=duration
                        if collapse and not detail['diagnostics']['second_style_equals_first_input']:raise ValueError('native post-style normalization mismatch')
                        if state['physical_forwards']-forwards!=9:raise ValueError('nine step forwards')
                        if not collapse:reference._criterion=reference_criterion(reference,alternative)
                        rz,rd=reference.keyed_step(x,raw)
                        if not torch.equal(z,rz) or not equal(host.snapshot()['base'],reference.snapshot()['base']):
                            raise ValueError('independent output/gradient/model/Adam/RNG parity')
                        from ..r20_model_only_search.method import weights_for
                        expected_q=torch.stack(reference.weak).sigmoid().mean(0)
                        expected_w=weights_for(torch.stack(reference.weak).sigmoid(),.1,2.)
                        if any(not torch.equal(a,b) for a,b in zip(host.target_cache,(expected_q,expected_w))):raise ValueError('current q/W provenance')
                        counts=detail['native'][0]['counts']
                        if counts['backwards']!=1 or counts['base_adam']!=1 or host.native.steps!=i+1:raise ValueError('Adam/count continuity')
                    host.check_frozen(True);host.close();host=None;reference.close();reference=None;persist()
                checks[arm]=dict(status='PASS',collapsed_second_style_original_core_state_output_exact=True,
                    independent_two_style_output_gradient_Adam_corestate_exact=True,current_q_W_exact=True,
                    PRE_full_state_exact=True,one_backward_one_Adam=True,continuous_Adam=True,trainable_parameters=19136)
            if state['physical_updates']!=QUALIFICATION_UPDATES or state['completed_physical_updates']!=QUALIFICATION_UPDATES:
                raise ValueError('qualification physical accounting')
            save(root/'public/QUALIFICATION.json',dict(status='PASS',arms=checks,qualification_physical_updates=QUALIFICATION_UPDATES,
                steady_step_seconds=steady,cold_step_seconds=cold,physical_forwards=state['physical_forwards'],
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,
                online_label_reads=0,source_image_reads=0))
        elif stage=='formal':
            if read(root/'public/QUALIFICATION.json')['status']!='PASS' or not read(root/'FORMAL_ADMISSION.json')['admitted']:
                raise ValueError('qualification/resource barrier')
            for job in jobs(slot):
                guard();folder=root/'jobs'/job['id'];folder.mkdir();durations=[]
                host=Host(checkpoint,job['arm'],job['seed'],job['id'],qw=job['cell'],charge=charge);meter(host)
                rows=read(root/'private'/f'screen_o{job["order"]}.json')
                if len(rows)!=192:raise ValueError('frozen arrival count')
                physical_before=state['physical_updates']
                with (folder/'predictions.bits').open('xb') as bits,(folder/'diagnostics.jsonl').open('x') as trace:
                    for visit,row in enumerate(rows):
                        guard();x=image(reader,row,permit,guard);tick=time.time();raw=style(x,job['seed'],content[row['image_sha256']])
                        physical_before_step=state['physical_forwards'];pre=host.before(x);post,detail=host.intervention_step(x,raw,style(x,NEXT[job['seed']],content[row['image_sha256']]))
                        if state['physical_forwards']-physical_before_step!=10:raise ValueError('formal physical forward count')
                        torch.cuda.synchronize();duration=time.time()-tick;durations.append(duration)
                        masks=np.stack([(z[0]>=0).numpy() for z in (pre,post)]);bits.write(np.packbits(masks).tobytes())
                        trace.write(json.dumps(dict(visit=visit+1,seconds=duration,cell=job['cell'],K=detail['diagnostics']),allow_nan=False)+'\n')
                        state['arrivals']+=1
                        if (visit+1)%16==0:
                            bits.flush();trace.flush();storage_guard();state.update(job=job['id'],visit=visit+1,latest_seconds=duration);persist()
                    if bits.tell()!=192*2*WIDTH:raise ValueError('prediction byte coverage')
                if state['physical_updates']-physical_before!=192:raise ValueError('job physical accounting')
                host.check_frozen(True);torch.save(host.snapshot()['base'],folder/'terminal.private.pt');host.close();host=None
                save(folder/'COMPLETE.json',dict(**job,arrivals=192,physical_updates=192,seconds=sum(durations),online_label_reads=0))
                state['completed_jobs'].append(job['id']);persist();storage_guard()
        else:raise ValueError('unregistered stage')
        if state['physical_updates']!=state['completed_physical_updates']:raise ValueError('attempt/complete mismatch')
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input changed')
        save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,image_accesses=guard.extra['image_accesses'],
            opened_paths=sorted(opened),rejections=rejected))
        state['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        state.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        if reference is not None:reference.close()
        state.update(ended=time.time(),gpu_process_seconds=time.time()-started);persist()


def controller(c):
    root=Path(c['output_root']);attempts=[];status=dict(status='QUALIFICATION',started=time.time(),source_commit=c['source_commit'])
    with (root/'START_ONCE.json').open('x') as f:json.dump(status,f)
    def publish():
        save(root/'RUN_STATE.json',status);save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,
            gpu_process_seconds=sum(x['gpu_process_seconds'] for x in attempts),
            cpu_process_seconds=c['preparation_cpu_seconds']+sum(x['cpu_process_seconds'] for x in attempts),
            gpu_cap_seconds=None,all_workers_retired=all(not read(p)['active'] for p in (root/'processes').glob('*.json'))))
    def remaining():return None
    def group(mode,stage,slots,cap):
        active=[];failed=False
        try:
            for slot in slots:
                if mode=='worker':
                    import subprocess
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
                    inv={int(i):(u.strip(),int(m)) for i,u,m in (line.split(',') for line in raw.splitlines())}
                    gpu=c['gpus'][slot];u,m=inv[gpu['physical_id']]
                    if u!=gpu['uuid'] or m<10240:raise ValueError('GPU admission failed')
                active.append(child(c,mode,stage,slot,cap))
        finally:
            while active:
                for item in list(active):
                    receipt=finish_child(c,item)
                    if receipt is not None:attempts.append(receipt);failed|=receipt['exit_code']!=0;active.remove(item);publish()
                if active:time.sleep(5)
        return not failed
    try:
        if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU admission missing')
        publish()
        if not group('worker','qualify',[0],None):raise RuntimeError('qualification failed; retain evidence for autonomous diagnosis')
        q=read(root/'public/QUALIFICATION.json')
        projection=sum(q['steady_step_seconds'].values())*1152*1.15+180
        admitted=q['status']=='PASS'
        admission=dict(admitted=admitted,projected_GPU_process_seconds=projection,remaining_GPU_process_seconds=remaining(),
            gpu_cap_seconds=None,new_formal_physical_updates=1152,reused_endpoint_updates=2304,
            automatic_extension=False,automatic_retry=False,time_GPU_caps_removed_by_user=True)
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        if not admitted:raise RuntimeError('frozen resource admission failed')
        c['formal_start_epoch']=time.time();c['deadline']=None;save(root/'private/config.json',c)
        admission.update(formal_start_epoch=c['formal_start_epoch'],deadline=c['deadline'])
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        frozen=read(root/'public/FROZEN.json');frozen.update(formal_start_epoch=c['formal_start_epoch'],deadline=c['deadline']);save(root/'public/FROZEN.json',frozen)
        status['status']='FORMAL';publish()
        if not group('worker','formal',[0,1],None):raise RuntimeError('formal failed; retain evidence for autonomous diagnosis')
        status['status']='SCORING';publish()
        if not group('score','final',[0],None):raise RuntimeError('CPU scoring failed; no retry')
        status.update(status='COMPLETE',automatic_followup=False);return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:status['ended']=time.time();publish()


def main():
    mode=os.environ['MODE']
    if mode=='self_check':
        self_check()
        from ..b1_host import official
        from ..integrations.ctta_suite import load_reference_interfaces
        from .score import self_check as score_check
        official();load_reference_interfaces();score_check()
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU admission initialized CUDA')
        print('FULL_CPU_ENTRY_SELF_CHECK_PASS',flush=True);return 0
    c=read(os.environ['CONFIG'])
    if mode=='score':
        from .score import score
        return score(c)
    return {'worker':worker,'controller':controller}[mode](c)

if __name__=='__main__':raise SystemExit(main())
