"""One qualification barrier, eighteen continuous trajectories, CPU-only labels."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from .method import ARMS,Host,self_check
from ..r75_dense_wcw.run import child,finish_child
from ..r51_same_state.run import read,save
from ..r51_same_state.score import interval,csvout

SEEDS=(20260907,17011,29009)
WIDTH=65536
CELLS=tuple(a+'/'+r for a in ARMS for r in ('O','A'))


def jobs():
    return [dict(id=f's{s}_o{o}_{a}',seed=s,order=o,arm=a) for s in SEEDS for o in (0,1) for a in ARMS]


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights,image
    from ..r19_model_only.method import equal
    from ..r37_cw_orientation.method import Host as Reference,candidates
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    slot=int(os.environ['SLOT']);stage=os.environ['STAGE'];root=Path(c['output_root']);start=time.time()
    dest=root/'private'/f'{stage}_worker{slot}';dest.mkdir();host=None
    status=dict(status='QUALIFYING' if stage=='qualify' else 'RUNNING',arrivals=0,qualification_updates=0,completed_jobs=[],started=start)
    def persist():save(dest/'RUN_STATE.json',status)
    def guard():
        if time.time()>float(os.environ['CHILD_DEADLINE'])-20:raise TimeoutError('fixed GPU budget exhausted')
    def disk_guard():
        used=sum(p.stat().st_size for p in (root/'jobs').glob('*/*') if p.is_file())
        if used>c['output_cap_bytes']-32*1024**2:raise RuntimeError('output storage cap reached')
    guard.extra={'image_accesses':0};persist()
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][slot])
        torch.cuda.set_per_process_memory_fraction(1/3,0)
        permit,opened,rejected=filesystem_guard(c);state=weights(c)
        reader=TargetReader(c['target_root'],256*1024**2,'image')
        if stage=='qualify':
            rows=read(root/'private/screen_o0.json');pixels=image(reader,rows[0],permit,guard)
            checks={};timings=[]
            for arm in ARMS:
                guard();host=Host(state,arm,SEEDS[0],'qualification');initial=host.snapshot()
                tick=time.time();z,detail=host.step(pixels);status['qualification_updates']+=1
                after=host.snapshot();diag=json.loads(json.dumps(host.diag))
                outputs=host.readouts(pixels,z);torch.cuda.synchronize();timings.append(time.time()-tick)
                if not equal(host.snapshot(),after) or host.diag!=diag:raise ValueError('observer changed state/diagnostics')
                host.restore(initial);repeat,_=host.step(pixels);status['qualification_updates']+=1
                if not torch.equal(z,repeat) or not equal(host.snapshot(),after):raise ValueError('full-state replay failed')
                if arm=='NATIVE':
                    reference=Reference(state,candidates()[3],SEEDS[0],'qualification')
                    rz,_=reference.step(pixels);status['qualification_updates']+=1
                    if not torch.equal(z,rz) or not equal(reference.snapshot(),after):raise ValueError('original baseline parity')
                    # The next arrival must also be identical after the extra observation.
                    other=image(reader,rows[1],permit,guard)
                    nz,_=host.step(other);rr,_=reference.step(other);status['qualification_updates']+=2
                    if not torch.equal(nz,rr) or not equal(host.snapshot(),reference.snapshot()):raise ValueError('observer changed next arrival')
                    reference.close();del reference
                else:
                    if min(host.diag[k] for k in ('branch0_gradient_norm','branch1_gradient_norm'))<=0:raise ValueError('detached branch')
                    tick=time.time();host.step(pixels);status['qualification_updates']+=1;torch.cuda.synchronize()
                    timings.append(time.time()-tick)
                if not torch.equal(outputs[0],z.sigmoid()):raise ValueError('native readout parity')
                host.check_frozen(True)
                checks[arm]=dict(full_state_replay_exact=True,observer_state_exact=True,
                    trainable_parameters=sum(p.numel() for p in host.native.params),trainable_names=host.native.names,
                    forwards=8+int(arm!='NATIVE'),observer_forwards=1,backwards=1,adam=1,
                    branches_connected=arm!='NATIVE',original_baseline_exact=arm=='NATIVE')
                host.close();host=None;persist()
            save(root/'public/QUALIFICATION.json',dict(status='PASS',arms=checks,
                qualification_updates=status['qualification_updates'],max_measured_step_seconds=max(timings),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,online_label_reads=0))
        elif stage=='formal':
            if read(root/'public/QUALIFICATION.json')['status']!='PASS' or not read(root/'FORMAL_ADMISSION.json')['admitted']:
                raise ValueError('qualification/resource barrier absent')
            assigned=[j for j in jobs() if j['order']==slot]
            for job in assigned:
                guard();rows=read(root/'private'/f'screen_o{job["order"]}.json')
                if len(rows)!=192:raise ValueError('frozen cohort size')
                folder=root/'jobs'/job['id'];folder.mkdir();host=Host(state,job['arm'],job['seed'],job['id']);durations=[]
                with (folder/'predictions.bits').open('xb') as bits,(folder/'diagnostics.jsonl').open('x') as trace:
                    for i,row in enumerate(rows):
                        guard();pixels=image(reader,row,permit,guard);tick=time.time();z,detail=host.step(pixels)
                        outputs=host.readouts(pixels,z);torch.cuda.synchronize();duration=time.time()-tick;durations.append(duration)
                        masks=(outputs[:,0]>=.5).numpy();bits.write(np.packbits(masks).tobytes())
                        trace.write(json.dumps(dict(visit=i+1,seconds=duration,foreground_fraction=masks.mean((2,3)).tolist(),
                            **detail['diagnostics']),allow_nan=False)+'\n');status['arrivals']+=1
                        if (i+1)%16==0:
                            bits.flush();trace.flush();disk_guard();status.update(job=job['id'],visit=i+1,latest_seconds=duration);persist()
                    if bits.tell()!=192*2*WIDTH:raise ValueError('prediction coverage')
                host.check_frozen(True);torch.save(host.snapshot(),folder/'terminal.private.pt');host.close();host=None
                save(folder/'COMPLETE.json',dict(**job,arrivals=192,readouts=2,seconds=sum(durations),online_label_reads=0))
                status['completed_jobs'].append(job['id']);persist();disk_guard()
        else:raise ValueError('unregistered stage')
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input changed')
        save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,
            image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),rejections=rejected))
        status['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        status.update(ended=time.time(),gpu_process_seconds=time.time()-start);persist()


def contrasts(v):
    out={CELLS[a]+'-'+CELLS[b]:v[:,:,a]-v[:,:,b] for a,b in ((2,0),(4,0),(4,2),(1,0),(3,2),(5,4),(5,0))}
    out['BARY_READOUT_INTERACTION']=(v[:,:,5]-v[:,:,4])-(v[:,:,1]-v[:,:,0])
    return out


def summarize(values,domains):
    names=sorted(set(domains));cells=[];summaries=[]
    for name,delta in contrasts(values).items():
        for d in names:
            for seed in SEEDS:
                for order in (0,1):
                    j=SEEDS.index(seed)*2+order
                    for ch,channel in enumerate(('OD','OC')):
                        cells.append(dict(contrast=name,domain='D'+str(names.index(d)),seed=seed,order=order,
                            channel=channel,n_contents=int((domains==d).sum()),delta_pp=float(delta[domains==d,j,ch].mean())))
        oc=np.mean([delta[domains==d,:,1].mean(0) for d in names],0)
        domainmean=[delta[domains==d].mean((0,1)) for d in names]
        seedmean=[delta[domains==d,o::2].mean((0,1)) for d in names for o in (0,1)]
        summaries.append(dict(contrast=name,macro_delta_pp=float(np.mean(domainmean)),OC_delta_pp=float(oc.mean()),
            OD_delta_pp=float(np.mean(domainmean,0)[0]),ci95_pp=interval(delta,domains),
            OC_ci95_pp=interval(delta[:,:,1:2],domains),OC_order_delta_pp=[float(oc[o::2].mean()) for o in (0,1)],
            OC_positive_trajectories=int((oc>0).sum()),imageweighted_delta_pp=float(delta.mean()),
            worst_domain_channel_pp=float(np.min(domainmean)),worst_seedmean_cell_pp=float(np.min(seedmean)),
            worst_individual_cell_pp=min(r['delta_pp'] for r in cells if r['contrast']==name)))
    by={s['contrast']:s for s in summaries};b=by['BARY/O-NATIVE/O'];v=by['BARY/O-VERTEX/O']
    learning=(b['OC_delta_pp']>=.2 and b['OC_ci95_pp'][0]>0 and b['macro_delta_pp']>=0 and
        min(b['OC_order_delta_pp'])>0 and b['OC_positive_trajectories']>=5 and b['imageweighted_delta_pp']>=0)
    harm=b['worst_domain_channel_pp']>=-.5 and b['worst_seedmean_cell_pp']>=-2
    specific=v['OC_ci95_pp'][0]>0 and v['macro_delta_pp']>=0
    return summaries,cells,dict(learning_signal=learning,domain_harm_gate=harm,barycenter_specificity=specific,
        all_pass=learning and harm and specific,automatic_followup=False)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only label scorer')
    root=Path(c['output_root']);start=time.time()
    for name in ('qualify_worker0','formal_worker0','formal_worker1'):
        p=read(root/'processes'/f'{name}.json');s=read(root/'private'/name/'RUN_STATE.json')
        if p['active'] or p['exit_code']!=0 or s['status']!='GPU_COMPLETE':raise ValueError('GPU retirement barrier')
    panel=read(root/'scorer/screen_o0.json');ids=[r['image_sha256'] for r in panel];index={v:i for i,v in enumerate(ids)}
    if len(index)!=192:raise ValueError('unique development contents')
    domains=np.array([r['domain'] for r in panel]);names=sorted(set(domains))
    reader=TargetReader(c['target_root'],256*1024**2,'mask');labels={r['image_sha256']:reader.read(r) for r in panel}
    values=np.empty((192,6,6,2));errors=np.empty((192,6,6,2,3));mechanism=[];temporal=[]
    for job in jobs():
        folder=root/'jobs'/job['id'];receipt=read(folder/'COMPLETE.json')
        if receipt['arrivals']!=192:raise ValueError('incomplete trajectory')
        a=ARMS.index(job['arm'])*2;j=SEEDS.index(job['seed'])*2+job['order'];rows=read(root/'scorer'/f'screen_o{job["order"]}.json')
        prior=(Path(c['reference_native_root'])/'jobs'/f's{job["seed"]}_o{job["order"]}_NATIVE_BN'/'predictions.bits').open('rb') if a==0 else None
        with (folder/'predictions.bits').open('rb') as f:
            for visit,row in enumerate(rows):
                raw=f.read(2*WIDTH)
                if len(raw)!=2*WIDTH:raise ValueError('truncated masks')
                if prior is not None and raw[:WIDTH]!=prior.read(WIDTH):raise ValueError('historical native prediction parity')
                masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,2,512,512).astype(bool)
                label=labels[row['image_sha256']];i=index[row['image_sha256']];truth=label.numpy()[0].astype(bool)
                for r in (0,1):
                    values[i,j,a+r]=[100*m['dice'] for m in hard_metrics(masks[r],label)]
                    temporal.append(dict(cell=CELLS[a+r],seed=job['seed'],order=job['order'],half=visit//96,
                        domain='D'+str(names.index(row['domain'])),OD=float(values[i,j,a+r,0]),OC=float(values[i,j,a+r,1])))
                    for ch in (0,1):errors[i,j,a+r,ch]=[(masks[r,ch]&~truth[ch]).mean(),(~masks[r,ch]&truth[ch]).mean(),masks[r,ch].mean()]
            if f.read(1):raise ValueError('extra masks')
        if prior is not None:
            if prior.read(1):raise ValueError('extra reference masks')
            prior.close()
        traces=[json.loads(s) for s in (folder/'diagnostics.jsonl').read_text().splitlines()]
        if [t['visit'] for t in traces]!=list(range(1,193)):raise ValueError('trace coverage')
        mechanism.append(dict(arm=job['arm'],seed=job['seed'],order=job['order'],seconds=receipt['seconds'],
            mean_BN_gradient=float(np.mean([t['BN_gradient_norm'] for t in traces])),
            mean_BN_update=float(np.mean([t['BN_update_norm'] for t in traces])),
            branch0_gradient=float(np.mean([t.get('branch0_gradient_norm',0) for t in traces])),
            branch1_gradient=float(np.mean([t.get('branch1_gradient_norm',0) for t in traces]))))
    reader.after_check();np.save(root/'scorer/values.private.npy',values)
    summaries,cells,gates=summarize(values,domains);scores=[];error_rows=[];time_rows=[]
    for a,cell in enumerate(CELLS):
        scores.append(dict(cell=cell,macro_Dice_percent=float(np.mean([values[domains==d,:,a].mean() for d in names])),
            OD=float(np.mean([values[domains==d,:,a,0].mean() for d in names])),
            OC=float(np.mean([values[domains==d,:,a,1].mean() for d in names]))))
        for d in names:
            for ch,channel in enumerate(('OD','OC')):
                error_rows.append(dict(cell=cell,domain='D'+str(names.index(d)),channel=channel,
                    Dice_percent=float(values[domains==d,:,a,ch].mean()),
                    false_positive_pixel_fraction=float(errors[domains==d,:,a,ch,0].mean()),
                    false_negative_pixel_fraction=float(errors[domains==d,:,a,ch,1].mean()),foreground_fraction=float(errors[domains==d,:,a,ch,2].mean())))
        for seed in SEEDS:
            for order in (0,1):
                for half in (0,1):
                    for d in range(4):
                        subset=[t for t in temporal if t['cell']==cell and t['seed']==seed and t['order']==order and t['half']==half and t['domain']=='D'+str(d)]
                        time_rows.append(dict(cell=cell,seed=seed,order=order,half=half,domain='D'+str(d),n=len(subset),
                            OD=float(np.mean([t['OD'] for t in subset])) if subset else None,OC=float(np.mean([t['OC'] for t in subset])) if subset else None))
    out=root/'public';save(out/'RESULTS.json',dict(status='COMPLETE',contents=192,independent_trajectories=18,
        formal_updates=3456,predictions=6912,contrasts=summaries,gates=gates,independent_validation=False,
        historical_native_masks_exact=True,online_label_reads=0,cpu_seconds=time.time()-start))
    for name,rows in (('ARM_SCORES',scores),('ALL_CELLS',cells),('NEGATIVE_CELLS',[r for r in cells if r['delta_pp']<0]),
        ('DOMAIN_CHANNEL_ERRORS',error_rows),('MECHANISM',mechanism),('TEMPORAL_AGGREGATES',time_rows)):
        csvout(out/(name+'.csv'),rows)
    save(out/'COMPLETION.json',dict(all_gpu_workers_retired=True,formal_updates=3456,trajectories=18,predictions=6912,
        online_label_reads=0,source_image_reads=0,historical_native_masks_exact=True,automatic_followup=False))
    return 0


def controller(c):
    root=Path(c['output_root']);attempts=[];status=dict(status='QUALIFICATION',started=time.time(),source_commit=c['source_commit'])
    with (root/'START_ONCE.json').open('x') as f:json.dump(status,f)
    def publish():
        save(root/'RUN_STATE.json',status);save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,
            gpu_process_seconds=sum(x['gpu_process_seconds'] for x in attempts),
            cpu_process_seconds=c['preparation_cpu_seconds']+sum(x['cpu_process_seconds'] for x in attempts),
            gpu_cap_seconds=c['gpu_cap_seconds'],all_workers_retired=all(not read(p)['active'] for p in (root/'processes').glob('*.json'))))
    def group(mode,stage,slots,cap):
        active=[];failed=False
        try:
            for slot in slots:
                if mode=='worker':
                    import subprocess
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
                    inventory={int(i):(u.strip(),int(m)) for i,u,m in (line.split(',') for line in raw.splitlines())}
                    gpu=c['gpus'][slot];uuid,free=inventory[gpu['physical_id']]
                    if uuid!=gpu['uuid'] or free<10240:raise RuntimeError('10 GiB free-memory admission failed')
                active.append(child(c,mode,stage,slot,cap))
        finally:
            while active:
                for item in list(active):
                    receipt=finish_child(c,item)
                    if receipt is not None:attempts.append(receipt);failed|=receipt['exit_code']!=0;active.remove(item);publish()
                if active:time.sleep(5)
        return not failed
    try:
        if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU check absent')
        publish()
        if not group('worker','qualify',[0],300):raise RuntimeError('qualification failed; stop without retry')
        q=read(root/'public/QUALIFICATION.json');remaining=c['gpu_cap_seconds']-sum(x['gpu_process_seconds'] for x in attempts)
        projection=q['max_measured_step_seconds']*3456*1.3+120
        admitted=q['status']=='PASS' and projection<remaining and time.time()+projection/2+900<c['deadline']
        admission=dict(admitted=admitted,projected_gpu_seconds=projection,remaining_gpu_seconds=remaining,
            gpu_cap_seconds=c['gpu_cap_seconds'],output_cap_bytes=c['output_cap_bytes'],per_worker_allocator_cap_GiB=8,
            deadline=c['deadline'],automatic_retry=False)
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        if not admitted:raise RuntimeError('frozen resource admission failed; no automatic extension')
        status['status']='FORMAL';publish()
        # Reserve polling/termination accounting overhead inside the aggregate cap.
        if not group('worker','formal',[0,1],(remaining-30)/2):raise RuntimeError('formal matrix failed; stop without retry')
        status['status']='SCORING';publish()
        if not group('score','final',[0],600):raise RuntimeError('CPU scoring failed; no retry')
        status.update(status='COMPLETE',scientific_followup=False);return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:status['ended']=time.time();publish()


def main():
    if os.environ['MODE']=='self_check':
        self_check()
        from ..b1_host import official
        from ..integrations.ctta_suite import load_reference_interfaces
        from ..r19_model_only.runtime import weights,image
        from ..r36_incremental_modules.run import filesystem_guard
        from ..r24_c_context.run import hard_metrics
        official();load_reference_interfaces()
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU check initialized CUDA')
        # Synthetic scores exercise all contrast dimensions and fixed harm gates.
        values=np.zeros((8,6,6,2));domains=np.repeat(np.arange(4),2)
        values[:,:,4,1]=.4;values[:,:,4,0]=.1;values[:,:,2,1]=.1
        _,cells,gates=summarize(values,domains)
        assert gates['all_pass'] and len(cells)==8*4*3*2*2
        values[:2,:,4,0]=-3
        assert not summarize(values,domains)[2]['domain_harm_gate']
        print('FULL_ENTRY_AND_SCORER_SELF_CHECK_PASS',flush=True);return 0
    c=read(os.environ['CONFIG']);return {'worker':worker,'score':score,'controller':controller}[os.environ['MODE']](c)

if __name__=='__main__':raise SystemExit(main())
