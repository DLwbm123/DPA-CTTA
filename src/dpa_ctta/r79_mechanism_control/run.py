"""8640 physical updates, no diagnostic feedback, labels after all GPU retirement."""
import json,os,time,traceback,copy
from pathlib import Path
import numpy as np
from .method import Host,ARMS,SEEDS,NEXT,style,self_check
from ..r51_same_state.run import read,save
from ..r75_dense_wcw.run import child,finish_child

WIDTH=65536


def jobs(stage,slot):
    if stage=='main':return [dict(id=f'k_s{s}_o{slot}_{a}',seed=s,order=slot,arm=a) for s in SEEDS for a in ARMS]
    return [dict(id=f'r_s{s}_{a}_part{slot}',seed=s,arm=a,part=slot) for s in SEEDS for a in ARMS]


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights,image
    from ..r19_model_only.method import equal
    from ..r78_aligned_barycenter.method import Host as Previous
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    stage=os.environ['STAGE'];slot=int(os.environ['SLOT']);root=Path(c['output_root']);started=time.time()
    dest=root/'private'/f'{stage}_worker{slot}';dest.mkdir();host=None
    state=dict(status='QUALIFYING' if stage=='qualify' else 'RUNNING',started=started,arrivals=0,
        physical_updates=0,completed_physical_updates=0,completed_jobs=[])
    def persist():save(dest/'RUN_STATE.json',state)
    def charge(kind):
        if kind=='attempt':state['physical_updates']+=1
        else:state['completed_physical_updates']+=1
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
            x=image(reader,canonical[0],permit,guard);y=image(reader,canonical[1],permit,guard)
            checks={};cycle_times=[];cold_cycle_times=[];reset_times=[]
            for arm in ARMS:
                guard();host=Host(checkpoint,arm,SEEDS[0],'qualification',charge=charge)
                initial=host.snapshot();raw=style(x,SEEDS[0],0);alternative=style(x,NEXT[SEEDS[0]],0)
                if arm=='NATIVE':
                    charge('attempt');z,_=host.step(x);charge('complete');ours=host.snapshot()['base']
                    reference=Previous(checkpoint,arm,SEEDS[0],'qualification')
                    charge('attempt');rz,_=reference.step(x);charge('complete')
                    if not torch.equal(z,rz) or not equal(ours,reference.snapshot()):raise ValueError('original NATIVE parity')
                    reference.close();del reference;host.restore(initial)
                # Exercise U restoration and then the exact committed trajectory.
                before=host.snapshot();host.before(x)
                if not equal(host.snapshot(),before):raise ValueError('pre observer changed state')
                tick=time.time();outputs,_=host.cycle(x,raw,alternative);torch.cuda.synchronize();cold_cycle_times.append(time.time()-tick)
                after=host.snapshot();host.restore(initial);alone,_=host.keyed_step(x,raw)
                if not torch.equal(outputs[2],alone) or not equal(host.snapshot(),after):raise ValueError('U affected committed state')
                # Next arrival with and without diagnostic branches must agree too.
                current=style(y,SEEDS[0],1);alternate=style(y,NEXT[SEEDS[0]],1)
                tick=time.time();second,_=host.cycle(y,current,alternate);torch.cuda.synchronize();cycle_times.append(time.time()-tick)
                second_state=host.snapshot();host.restore(after);plain,_=host.keyed_step(y,current)
                if not torch.equal(second[2],plain) or not equal(host.snapshot(),second_state):raise ValueError('next-arrival interference')
                # Reset controls are independent of previously processed images.
                host.restore(initial);tick=time.time();host.before(x);rx,_=host.keyed_step(x,raw);torch.cuda.synchronize();reset_times.append(time.time()-tick)
                rs=host.snapshot();host.restore(initial);host.keyed_step(y,current);host.restore(initial);repeat,_=host.keyed_step(x,raw)
                if not torch.equal(rx,repeat) or not equal(host.snapshot(),rs):raise ValueError('initial state control contaminated')
                if any(v['observed_loss_scale'] not in (.5,1.) for v in host.diag['gradient_sites'].values()):raise ValueError('gradient site tagging')
                host.check_frozen(True)
                checks[arm]=dict(status='PASS',pre_observer_exact=True,U_discard_exact=True,next_arrival_exact=True,
                    reset_order_independent=True,q_W_exact=True,trainable_parameters=19136,one_backward_one_Adam=True,
                    gradient_sites=list(host.diag['gradient_sites']))
                host.close();host=None;persist()
            if state['physical_updates']!=29 or state['completed_physical_updates']!=29:raise ValueError('qualification accounting')
            # Use steady qualified cycles, without construction/import time, for the frozen admission estimate.
            save(root/'public/QUALIFICATION.json',dict(status='PASS',arms=checks,qualification_physical_updates=29,
                max_cycle_seconds=max(cycle_times),cold_cycle_seconds=cold_cycle_times,max_reset_seconds=max(reset_times),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,
                online_label_reads=0,source_image_reads=0))
        elif stage in ('main','reset'):
            if read(root/'public/QUALIFICATION.json')['status']!='PASS':raise ValueError('qualification barrier')
            if not read(root/'FORMAL_ADMISSION.json')['admitted']:raise ValueError('resource barrier')
            for job in jobs(stage,slot):
                guard();folder=root/'jobs'/job['id'];folder.mkdir()
                host=Host(checkpoint,job['arm'],job['seed'],job['id'],charge=charge);initial=host.snapshot()
                rows=read(root/'private'/f'screen_o{job["order"]}.json') if stage=='main' else canonical[slot::2]
                if len(rows)!=(192 if stage=='main' else 96):raise ValueError('job cohort coverage')
                physical_before=state['physical_updates'];durations=[]
                with (folder/'predictions.bits').open('xb') as bits,(folder/'diagnostics.jsonl').open('x') as trace:
                    for visit,row in enumerate(rows):
                        guard();x=image(reader,row,permit,guard);i=content[row['image_sha256']];tick=time.time()
                        raw=style(x,job['seed'],i)
                        if stage=='main':
                            alternative=style(x,NEXT[job['seed']],i);outputs,detail=host.cycle(x,raw,alternative)
                        else:
                            host.restore(initial);pre=host.before(x);post,record=host.keyed_step(x,raw)
                            outputs=(pre,post);detail=dict(R=record['diagnostics'])
                        torch.cuda.synchronize();duration=time.time()-tick;durations.append(duration)
                        masks=np.stack([(z[0]>=0).numpy() for z in outputs]);bits.write(np.packbits(masks).tobytes())
                        trace.write(json.dumps(dict(visit=visit+1,seconds=duration,**detail),allow_nan=False)+'\n')
                        state['arrivals']+=1
                        if (visit+1)%16==0:
                            bits.flush();trace.flush();storage_guard();state.update(job=job['id'],visit=visit+1,latest_seconds=duration);persist()
                    if bits.tell()!=len(rows)*(3 if stage=='main' else 2)*WIDTH:raise ValueError('prediction coverage')
                expected=len(rows)*(2 if stage=='main' else 1)
                if state['physical_updates']-physical_before!=expected:raise ValueError('job physical update accounting')
                host.check_frozen(True)
                if stage=='main':torch.save(host.snapshot()['base'],folder/'terminal.private.pt')
                host.close();host=None
                save(folder/'COMPLETE.json',dict(**job,arrivals=len(rows),physical_updates=expected,seconds=sum(durations),online_label_reads=0))
                state['completed_jobs'].append(job['id']);persist();storage_guard()
        else:raise ValueError('unregistered stage')
        if state['physical_updates']!=state['completed_physical_updates']:raise ValueError('incomplete attempted update')
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input changed')
        save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,image_accesses=guard.extra['image_accesses'],
            opened_paths=sorted(opened),rejections=rejected))
        state['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        state.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        state.update(ended=time.time(),gpu_process_seconds=time.time()-started);persist()


def controller(c):
    root=Path(c['output_root']);attempts=[];status=dict(status='QUALIFICATION',started=time.time(),source_commit=c['source_commit'])
    with (root/'START_ONCE.json').open('x') as f:json.dump(status,f)
    def publish():
        save(root/'RUN_STATE.json',status);save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,
            gpu_process_seconds=sum(x['gpu_process_seconds'] for x in attempts),
            cpu_process_seconds=c['preparation_cpu_seconds']+sum(x['cpu_process_seconds'] for x in attempts),
            gpu_cap_seconds=18000,all_workers_retired=all(not read(p)['active'] for p in (root/'processes').glob('*.json'))))
    def remaining():return 18000-sum(x['gpu_process_seconds'] for x in attempts)
    def group(mode,stage,slots,cap):
        active=[];failed=False
        try:
            for slot in slots:
                if mode=='worker':
                    import subprocess
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
                    inventory={int(i):(u.strip(),int(m)) for i,u,m in (line.split(',') for line in raw.splitlines())}
                    gpu=c['gpus'][slot];u,m=inventory[gpu['physical_id']]
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
        if not group('worker','qualify',[0],600):raise RuntimeError('qualification failed; no automatic retry')
        q=read(root/'public/QUALIFICATION.json')
        projection=(q['max_cycle_seconds']*3456+q['max_reset_seconds']*1728)*1.15+180
        admitted=q['status']=='PASS' and projection<remaining()-30 and projection/2+900<10800
        admission=dict(admitted=admitted,projected_GPU_process_seconds=projection,remaining_GPU_process_seconds=remaining(),
            gpu_cap_seconds=18000,main_physical_updates=6912,reset_physical_updates=1728,total_physical_updates=8640,
            automatic_extension=False,automatic_retry=False)
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        if not admitted:raise RuntimeError('frozen resource admission failed')
        c['formal_start_epoch']=time.time();c['deadline']=c['formal_start_epoch']+10800;save(root/'private/config.json',c)
        admission.update(formal_start_epoch=c['formal_start_epoch'],deadline=c['deadline'])
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        frozen=read(root/'public/FROZEN.json');frozen.update(formal_start_epoch=c['formal_start_epoch'],deadline=c['deadline']);save(root/'public/FROZEN.json',frozen)
        status['status']='MAIN';publish()
        # Main cycles account for 80% of initial allocation; reset gets the actual remainder.
        if not group('worker','main',[0,1],(remaining()-30)*.4):raise RuntimeError('main failed; no automatic retry')
        status['status']='RESET_DIAGNOSTICS';publish()
        if not group('worker','reset',[0,1],(remaining()-30)/2):raise RuntimeError('reset failed; no automatic retry')
        status['status']='SCORING';publish()
        if not group('score','final',[0],900):raise RuntimeError('CPU scoring failed; no retry')
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
        from ..r19_model_only.runtime import weights,image
        from ..r36_incremental_modules.run import filesystem_guard
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
