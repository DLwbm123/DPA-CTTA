"""10368 new updates, a reused frozen 11 anchor, bounded detached execution."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from .method import Host,ARMS,SEEDS,CELLS,style,self_check
from ..r51_same_state.run import read,save
from ..r75_dense_wcw.run import child,finish_child

WIDTH=65536
QUALIFICATION_UPDATES=30


def jobs(slot):
    return [dict(id=f'c{cell}_s{s}_o{slot}_{a}',cell=cell,seed=s,order=slot,arm=a)
            for s in SEEDS for cell in CELLS for a in ARMS]


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights,image
    from ..r19_model_only.method import equal
    from ..r79_mechanism_control.method import Host as Previous
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    stage=os.environ['STAGE'];slot=int(os.environ['SLOT']);root=Path(c['output_root']);started=time.time()
    dest=root/'private'/f'{stage}_worker{slot}';dest.mkdir();host=None;reference=None
    state=dict(status='QUALIFYING' if stage=='qualify' else 'RUNNING',started=started,arrivals=0,
               physical_updates=0,completed_physical_updates=0,completed_jobs=[])
    def persist():save(dest/'RUN_STATE.json',state)
    def charge(kind):state['physical_updates' if kind=='attempt' else 'completed_physical_updates']+=1
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
                guard();host=Host(checkpoint,arm,SEEDS[0],'qualification',cell='11',charge=charge)
                reference=Previous(checkpoint,arm,SEEDS[0],'qualification',charge=charge)
                reference_first=None;fixed_first=None
                for i,x in enumerate(xs):
                    raw=style(x,SEEDS[0],i);saved=host.snapshot();host.before(x)
                    if not equal(saved,host.snapshot()):raise ValueError('PRE changed complete state')
                    z,_=host.intervention_step(x,raw);rz,_=reference.keyed_step(x,raw)
                    if not torch.equal(z,rz) or not equal(host.snapshot(),reference.snapshot()):raise ValueError('R79 11 original output/state parity')
                    if i==0:reference_first=host.snapshot();firstq=tuple(v.clone() for v in host.target_cache)
                host.check_frozen(True);host.close();host=None;reference.close();reference=None
                for cell in CELLS:
                    guard();host=Host(checkpoint,arm,SEEDS[0],'qualification',cell=cell,charge=charge)
                    initial=host.snapshot()
                    for i,x in enumerate(xs):
                        raw=style(x,SEEDS[0],i);saved=host.snapshot()
                        if cell[1]=='0':
                            oracle=host.explicit_initial_targets(x)
                            if not equal(saved,host.snapshot()):raise ValueError('initial reference observer changed state')
                        tick=time.time();raw=style(x,SEEDS[0],i);pre=host.before(x)
                        if not equal(saved,host.snapshot()):raise ValueError('PRE branch contamination')
                        z,detail=host.intervention_step(x,raw);torch.cuda.synchronize();duration=time.time()-tick
                        (cold if i==0 else steady)[arm+'/'+cell]=duration
                        if cell[1]=='0' and any(not torch.equal(a,b) for a,b in zip(oracle,host.target_cache)):
                            raise ValueError('initial q/W donor mismatch')
                        if i==0:
                            if cell=='00':fixed_first=host.snapshot()
                            if cell=='10' and not equal(fixed_first,host.snapshot()):raise ValueError('empty Adam initial control differs')
                            if cell=='01' and not equal(reference_first,host.snapshot()):raise ValueError('self q initial control differs')
                            if any(not torch.equal(a,b) for a,b in zip(firstq,host.target_cache)):raise ValueError('S0 targets unequal across cells')
                        if cell[0]=='0':
                            for p in host.native.params:
                                s=host.native.base.state[p]
                                if int(s['step'])!=1:raise ValueError('reset Adam not age one')
                                if not torch.allclose(s['exp_avg'],p.grad*.1,atol=1e-9,rtol=1e-5):raise ValueError('Adam first moment not fresh')
                                if not torch.allclose(s['exp_avg_sq'],p.grad.square()*.001,atol=1e-12,rtol=1e-5):raise ValueError('Adam second moment not fresh')
                        elif int(host.native.steps)!=i+1:raise ValueError('carry Adam age drift')
                        counts=detail['native'][0]['counts']
                        if counts['backwards']!=1 or counts['base_adam']!=1:raise ValueError('one physical step contract')
                    host.check_frozen(True);host.close();host=None;persist()
                checks[arm]=dict(status='PASS',R79_11_state_output_exact=True,PRE_state_exact=True,
                    initial_donor_q_W_exact=True,initial_donor_state_restored=True,Adam_reset_age_moments=True,
                    empty_state_first_update_exact=True,one_backward_one_Adam=True,trainable_parameters=19136)
            if state['physical_updates']!=QUALIFICATION_UPDATES or state['completed_physical_updates']!=QUALIFICATION_UPDATES:
                raise ValueError('qualification physical accounting')
            save(root/'public/QUALIFICATION.json',dict(status='PASS',arms=checks,qualification_physical_updates=QUALIFICATION_UPDATES,
                steady_step_seconds=steady,cold_step_seconds=cold,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,online_label_reads=0,source_image_reads=0))
        elif stage=='formal':
            if read(root/'public/QUALIFICATION.json')['status']!='PASS' or not read(root/'FORMAL_ADMISSION.json')['admitted']:
                raise ValueError('qualification/resource barrier')
            for job in jobs(slot):
                guard();folder=root/'jobs'/job['id'];folder.mkdir();durations=[]
                host=Host(checkpoint,job['arm'],job['seed'],job['id'],cell=job['cell'],charge=charge)
                rows=read(root/'private'/f'screen_o{job["order"]}.json')
                if len(rows)!=192:raise ValueError('frozen arrival count')
                physical_before=state['physical_updates']
                with (folder/'predictions.bits').open('xb') as bits,(folder/'diagnostics.jsonl').open('x') as trace:
                    for visit,row in enumerate(rows):
                        guard();x=image(reader,row,permit,guard);tick=time.time();raw=style(x,job['seed'],content[row['image_sha256']])
                        pre=host.before(x);post,detail=host.intervention_step(x,raw)
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
            gpu_cap_seconds=18000,all_workers_retired=all(not read(p)['active'] for p in (root/'processes').glob('*.json'))))
    def remaining():return 18000-sum(x['gpu_process_seconds'] for x in attempts)
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
        if not group('worker','qualify',[0],600):raise RuntimeError('qualification failed; no automatic retry')
        q=read(root/'public/QUALIFICATION.json')
        projection=sum(q['steady_step_seconds'].values())*1152*1.15+180
        admitted=q['status']=='PASS' and projection<remaining()-30 and projection/2+900<10800
        admission=dict(admitted=admitted,projected_GPU_process_seconds=projection,remaining_GPU_process_seconds=remaining(),
            gpu_cap_seconds=18000,new_formal_physical_updates=10368,reused_11_updates=3456,
            automatic_extension=False,automatic_retry=False)
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        if not admitted:raise RuntimeError('frozen resource admission failed')
        c['formal_start_epoch']=time.time();c['deadline']=c['formal_start_epoch']+10800;save(root/'private/config.json',c)
        admission.update(formal_start_epoch=c['formal_start_epoch'],deadline=c['deadline'])
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        frozen=read(root/'public/FROZEN.json');frozen.update(formal_start_epoch=c['formal_start_epoch'],deadline=c['deadline']);save(root/'public/FROZEN.json',frozen)
        status['status']='FORMAL';publish()
        if not group('worker','formal',[0,1],(remaining()-30)/2):raise RuntimeError('formal failed; no automatic retry')
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
