"""Fixed 2x2 development experiment; labels only after both GPU workers retire."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from .method import ARMS,Host,self_check
from ..r75_dense_wcw.run import child,finish_child
from ..r51_same_state.run import read,save

SEEDS=(20260907,17011,29009)
WIDTH=65536


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
    slot=int(os.environ['SLOT']);root=Path(c['output_root']);start=time.time();dest=root/'private'/f'fixed_worker{slot}'
    dest.mkdir();status=dict(status='QUALIFYING',arrivals=0,completed_jobs=[],started=start);host=None
    def persist():save(dest/'RUN_STATE.json',status)
    def guard():
        if time.time()>float(os.environ['CHILD_DEADLINE'])-20:raise TimeoutError('fixed GPU budget exhausted')
    guard.extra={'image_accesses':0};persist()
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][slot])
        permit,opened,rejected=filesystem_guard(c);state=weights(c)
        reader=TargetReader(c['target_root'],256*1024**2,'image')
        row=read(root/'private/screen_o0.json')[0];pixels=image(reader,row,permit,guard)
        qualification={}
        for arm in ARMS:
            guard();host=Host(state,arm,SEEDS[0],'qualification')
            initial=host.snapshot();z,detail=host.step(pixels);after=host.snapshot()
            host.restore(initial);repeat,_=host.step(pixels)
            if not torch.equal(z,repeat) or not equal(host.snapshot(),after):raise ValueError('full state replay failed: '+arm)
            if arm=='NATIVE_BN':
                reference=Reference(state,candidates()[3],SEEDS[0],'qualification')
                rz,_=reference.step(pixels);rs=reference.snapshot()
                if not torch.equal(z,rz) or not all(equal(after[k],v) for k,v in rs.items()):
                    raise ValueError('retained continuous baseline parity failed')
                reference.close();del reference
            if host.teacher is not None and any(p.grad is not None for p in host.teacher.parameters()):
                raise ValueError('teacher received gradients')
            host.check_frozen(True)
            qualification[arm]=dict(full_state_replay_exact=True,trainable_parameters=sum(p.numel() for p in host.native.params),
                trainable_names=host.native.names,student_forwards=8,teacher_forwards=6 if host.dsr else 0,
                backwards=1,adam=1,teacher_stop_gradient=True)
            host.close();host=None
        save(dest/'QUALIFICATION.json',dict(status='PASS',arms=qualification,retained_baseline_exact=True,online_label_reads=0))
        status['status']='RUNNING';persist()
        # Split by seed/order pair, so each worker exercises all four cells.
        assigned=[j for j in jobs() if (SEEDS.index(j['seed'])*2+j['order'])%2==slot]
        for job in assigned:
            guard();rows=read(root/'private'/f'screen_o{job["order"]}.json')
            if len(rows)!=192:raise ValueError('fixed cohort coverage')
            folder=root/'jobs'/job['id'];folder.mkdir();host=Host(state,job['arm'],job['seed'],job['id']);durations=[]
            with (folder/'predictions.bits').open('xb') as bits,(folder/'diagnostics.jsonl').open('x') as trace:
                for i,row in enumerate(rows):
                    guard();pixels=image(reader,row,permit,guard);tick=time.time();z,detail=host.step(pixels)
                    torch.cuda.synchronize();duration=time.time()-tick;durations.append(duration)
                    masks=(z[0]>=0).numpy();bits.write(np.packbits(masks).tobytes())
                    trace.write(json.dumps(dict(visit=i+1,seconds=duration,foreground_fraction=masks.mean((1,2)).tolist(),
                                               **detail['diagnostics']),allow_nan=False)+'\n')
                    status['arrivals']+=1
                    if (i+1)%16==0:
                        bits.flush();trace.flush();status.update(job=job['id'],visit=i+1,latest_seconds=duration);persist()
                if bits.tell()!=len(rows)*WIDTH:raise ValueError('prediction bytes')
            host.check_frozen(True)
            # Terminal checkpoints are private, never inputs to later trajectories.
            torch.save(host.snapshot(),folder/'terminal.private.pt')
            host.close();host=None
            save(folder/'COMPLETE.json',dict(**job,arrivals=len(rows),seconds=sum(durations),online_label_reads=0))
            status['completed_jobs'].append(job['id']);persist()
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input changed')
        save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,image_accesses=guard.extra['image_accesses'],
                                         opened_paths=sorted(opened),rejections=rejected))
        status['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        status.update(ended=time.time(),gpu_process_seconds=time.time()-start);persist()


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import interval,csvout
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only labels')
    root=Path(c['output_root']);start=time.time()
    for slot in (0,1):
        process=read(root/'processes'/f'fixed_worker{slot}.json');status=read(root/'private'/f'fixed_worker{slot}'/'RUN_STATE.json')
        if process['active'] or process['exit_code']!=0 or status['status']!='GPU_COMPLETE':raise ValueError('GPU retirement barrier')
    panel=read(root/'scorer/screen_o0.json');ids=[r['image_sha256'] for r in panel];index={v:i for i,v in enumerate(ids)}
    if len(index)!=192:raise ValueError('paired contents not unique')
    domains=np.array([r['domain'] for r in panel]);names=sorted(set(domains));anonymous={d:f'D{i}' for i,d in enumerate(names)}
    reader=TargetReader(c['target_root'],256*1024**2,'mask');labels={r['image_sha256']:reader.read(r) for r in panel}
    values=np.empty((192,6,4,2));errors=np.empty((192,6,4,2,3));mechanism=[];prior_parity=True
    for job in jobs():
        folder=root/'jobs'/job['id'];receipt=read(folder/'COMPLETE.json')
        if receipt['arrivals']!=192:raise ValueError('incomplete trajectory')
        a=ARMS.index(job['arm']);j=SEEDS.index(job['seed'])*2+job['order'];rows=read(root/'scorer'/f'screen_o{job["order"]}.json')
        prior=None
        if a==0 and job['seed']==SEEDS[0]:
            prior=(Path(c['reference_native_root'])/'jobs'/f'screen_s{job["seed"]}_o{job["order"]}'/'predictions.bits').open('rb')
        with (folder/'predictions.bits').open('rb') as f:
            for row in rows:
                raw=f.read(WIDTH)
                if len(raw)!=WIDTH:raise ValueError('truncated masks')
                if prior is not None:
                    old=prior.read(4*WIDTH)
                    if raw!=old[:WIDTH]:raise ValueError('historical native mask parity failed')
                masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool)
                label=labels[row['image_sha256']];i=index[row['image_sha256']]
                values[i,j,a]=[100*m['dice'] for m in hard_metrics(masks,label)]
                truth=label.numpy()[0].astype(bool)
                for ch in range(2):errors[i,j,a,ch]=[(masks[ch]&~truth[ch]).mean(),(~masks[ch]&truth[ch]).mean(),masks[ch].mean()]
            if f.read(1):raise ValueError('extra masks')
        if prior is not None:
            if prior.read(1):raise ValueError('extra reference masks')
            prior.close()
        traces=[json.loads(v) for v in (folder/'diagnostics.jsonl').read_text().splitlines()]
        if [x['visit'] for x in traces]!=list(range(1,193)):raise ValueError('trace coverage')
        mechanism.append(dict(arm=job['arm'],seed=job['seed'],order=job['order'],hard_activations=sum(t.get('hard_active',False) for t in traces),
            mean_discrepancy=float(np.mean([t.get('discrepancy',0) for t in traces])),
            mean_BN_gradient=float(np.mean([t['BN_only_gradient_norm'] for t in traces])),
            mean_decoder_gradient=float(np.mean([t['decoder_gradient_norm'] for t in traces])),
            seconds=receipt['seconds'],final_threshold=traces[-1].get('positive_threshold')))
    reader.after_check();np.save(root/'scorer/values.private.npy',values)
    contrasts={f'{ARMS[a]}-{ARMS[b]}':values[:,:,a]-values[:,:,b] for a,b in ((2,0),(3,1),(1,0),(3,2),(3,0))}
    contrasts['INTERACTION']=(values[:,:,3]-values[:,:,1])-(values[:,:,2]-values[:,:,0])
    summaries=[];cells=[];scores=[];error_rows=[]
    for name,delta in contrasts.items():
        for d in names:
            for seed in SEEDS:
                for order in (0,1):
                    j=SEEDS.index(seed)*2+order
                    for ch,channel in enumerate(('OD','OC')):
                        cells.append(dict(contrast=name,domain=anonymous[d],seed=seed,order=order,channel=channel,
                                          n_contents=int((domains==d).sum()),delta_pp=float(delta[domains==d,j,ch].mean())))
        trajectory=np.mean([delta[domains==d].mean((0,2)) for d in names],axis=0)
        seedmean_cells=[np.mean([r['delta_pp'] for r in cells if r['contrast']==name and r['domain']==anonymous[d] and r['order']==o and r['channel']==ch]) for d in names for o in (0,1) for ch in ('OD','OC')]
        summaries.append(dict(contrast=name,macro_delta_pp=float(trajectory.mean()),ci95_pp=interval(delta,domains),
            OC_delta_pp=float(np.mean([delta[domains==d,:,1].mean() for d in names])),
            OD_delta_pp=float(np.mean([delta[domains==d,:,0].mean() for d in names])),
            order_delta_pp=[float(trajectory[o::2].mean()) for o in (0,1)],positive_trajectories=int((trajectory>0).sum()),
            imageweighted_delta_pp=float(delta.mean()),worst_seedmean_cell_pp=float(min(seedmean_cells)),
            worst_individual_cell_pp=min(r['delta_pp'] for r in cells if r['contrast']==name)))
    for a,arm in enumerate(ARMS):
        scores.append(dict(arm=arm,macro_Dice_percent=float(np.mean([values[domains==d,:,a].mean() for d in names]))))
        for d in names:
            for ch,channel in enumerate(('OD','OC')):
                error_rows.append(dict(arm=arm,domain=anonymous[d],channel=channel,Dice_percent=float(values[domains==d,:,a,ch].mean()),
                    false_positive_pixel_fraction=float(errors[domains==d,:,a,ch,0].mean()),
                    false_negative_pixel_fraction=float(errors[domains==d,:,a,ch,1].mean()),
                    foreground_fraction=float(errors[domains==d,:,a,ch,2].mean())))
    signals={}
    for s in summaries:
        signals[s['contrast']]=(s['macro_delta_pp']>=.2 and s['ci95_pp'][0]>0 and s['OC_delta_pp']>=0 and
            min(s['order_delta_pp'])>0 and s['positive_trajectories']>=5 and s['imageweighted_delta_pp']>=0 and s['worst_seedmean_cell_pp']>=-2)
    out=root/'public';save(out/'RESULTS.json',dict(status='COMPLETE',contents=192,independent_trajectories=24,formal_updates=4608,
        contrasts=summaries,development_signal_gates=signals,automatic_followup=False,independent_validation=False,
        historical_native_masks_exact=prior_parity,cpu_seconds=time.time()-start))
    csvout(out/'ARM_SCORES.csv',scores);csvout(out/'ALL_CELLS.csv',cells)
    csvout(out/'NEGATIVE_CELLS.csv',[r for r in cells if r['delta_pp']<0]);csvout(out/'MECHANISM.csv',mechanism)
    csvout(out/'DOMAIN_CHANNEL_ERRORS.csv',error_rows)
    save(out/'COMPLETION.json',dict(all_gpu_workers_retired=True,online_label_reads=0,source_image_reads=0,
        formal_updates=4608,independent_trajectories=24,historical_native_masks_exact=True))
    return 0


def controller(c):
    root=Path(c['output_root']);attempts=[];status=dict(status='FIXED_MATRIX',started=time.time(),source_commit=c['source_commit'])
    with (root/'START_ONCE.json').open('x') as f:json.dump(status,f)
    def publish():
        save(root/'RUN_STATE.json',status);save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,
            gpu_process_seconds=sum(x['gpu_process_seconds'] for x in attempts),
            cpu_process_seconds=c['preparation_cpu_seconds']+sum(x['cpu_process_seconds'] for x in attempts),
            gpu_cap_seconds=c['gpu_cap_seconds'],all_workers_retired=all(not read(p)['active'] for p in (root/'processes').glob('*.json'))))
    def group(mode,slots,cap):
        active=[];failed=False
        try:
            for slot in slots:active.append(child(c,mode,'fixed',slot,cap))
        finally:
            # If the second launch fails, finish accounting for the first within its existing cap.
            while active:
                for item in list(active):
                    receipt=finish_child(c,item)
                    if receipt is not None:attempts.append(receipt);failed|=receipt['exit_code']!=0;active.remove(item);publish()
                if active:time.sleep(5)
        return not failed
    try:
        if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU admission absent')
        publish()
        if not group('worker',[0,1],c['gpu_cap_seconds']/2):raise RuntimeError('fixed matrix failed; no automatic retry')
        status['status']='SCORING';publish()
        if not group('score',[0],600):raise RuntimeError('independent scoring failed')
        status.update(status='COMPLETE',scientific_followup=False);return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:status['ended']=time.time();publish()


def main():
    mode=os.environ['MODE']
    if mode=='self_check':
        self_check()
        from ..b1_host import official
        from ..integrations.ctta_suite import load_reference_interfaces
        official();load_reference_interfaces()
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU check initialized CUDA')
        print('FULL_ENTRY_IMPORT_CHECK_PASS',flush=True);return 0
    c=read(os.environ['CONFIG'])
    return {'worker':worker,'score':score,'controller':controller}[mode](c)

if __name__=='__main__':raise SystemExit(main())
