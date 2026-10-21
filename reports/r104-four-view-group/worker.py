"""Qualification and twelve label-free trajectories; score only in a later process."""
import gzip,json,os,runpy,time
from pathlib import Path
import numpy as np
SEEDS=(20260907,17011,29009)
def read(p):return json.loads(Path(p).read_text())
def run(c,write,mode):
    import torch
    from method import Four,Plain,close_enough,self_check
    from dpa_ctta.r85_migration.method import Host
    from dpa_ctta.r79_mechanism_control.method import style
    from dpa_ctta.r19_model_only.method import equal
    from dpa_ctta.r19_model_only.runtime import weights,image
    from dpa_ctta.r7_target_screen.runner import TargetReader
    from dpa_ctta.r9_current_first.assets import gpu_policy
    root=Path(c['output_root']);panels=[read(root/'private'/f'screen_o{o}.json') for o in (0,1)]
    canonical={r['image_sha256']:i for i,r in enumerate(panels[0])};assert len(canonical)==192
    assert set(canonical)=={r['image_sha256'] for r in panels[1]}
    prior=read(root/'private/qualify_state.json') if mode=='formal' else dict(physical_updates=0,physical_forwards=0,training_backwards=0)
    state=dict(status='RUNNING',mode=mode,started=time.time(),physical_updates=0,completed_updates=0,physical_forwards=0,training_backwards=0,completed_jobs=[],online_label_reads=0,online_score_reads=0)
    def persist():write('private/'+mode+'_state.json',state)
    def guard():
        if time.time()>c['deadline']-120:raise TimeoutError('absolute deadline')
    guard.extra={'image_accesses':0}
    def charge(k):
        guard()
        if k=='attempt':
            assert prior['physical_updates']+state['physical_updates']<c['max_updates'];state['physical_updates']+=1
        elif k=='complete':state['completed_updates']+=1
        else:raise ValueError('unexpected diagnostic backward')
    def meter(h):
        def forward(*_):
            guard();assert prior['physical_forwards']+state['physical_forwards']<c['max_forwards'];state['physical_forwards']+=1
        def backward(g):
            guard();assert prior['training_backwards']+state['training_backwards']<c['max_backwards'];state['training_backwards']+=1;return g
        h.handles.append(h.native.model.register_forward_pre_hook(forward));h.handles.append(h.native.params[0].register_hook(backward))
    def bits(z):return np.packbits((z>=0).numpy()).tobytes()
    def vector(h):return torch.cat([p.detach().flatten().cpu() for p in h.native.params])
    torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][0])
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    helper=runpy.run_path('/tmp/n91/m1_helpers.py');cfg=dict(c);cfg['reference_predictions']={**{'C2/'+k:v for k,v in c['c2_predictions'].items()},**{'VV/'+k:v for k,v in c['vv_predictions'].items()}}
    if mode=='formal':cfg['reference_predictions']={}
    permit,opened,rejected=helper['access_guard'](cfg)
    checkpoint=weights(c);reader=TargetReader(c['target_root'],256*1024**2,'image');hosts=[];timings=[];packed_sizes=[]
    persist()
    try:
        if mode=='qualify':
            self_check();checks=[]
            for seed in SEEDS:
                for order in (0,1):
                    for full in (False,True):
                        a=Host(checkpoint,'VV',seed,'duplicate',charge=charge) if full else Plain(checkpoint,'C2',seed,'duplicate',charge=charge)
                        b=Four(checkpoint,'DUPV' if full else 'DUPC',seed,'duplicate',charge=charge)
                        hosts=[a,b]
                        for h in hosts:meter(h)
                        refpath=c['vv_predictions' if full else 'c2_predictions'][f'{seed}_{order}']
                        with open(refpath,'rb') as ref:
                            for visit,row in enumerate(panels[order][:2]):
                                x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];raw=style(x,seed,idx)
                                before=[vector(h) for h in hosts];tick=time.time();za,da=a.keyed_step(x,raw);zb,db=b.keyed_step(x,raw);torch.cuda.synchronize();timings.append(time.time()-tick)
                                assert close_enough(za,zb) and close_enough(da['diagnostics']['joint_loss'],db['diagnostics']['joint_loss'])
                                assert close_enough(vector(a)-before[0],vector(b)-before[1]),'duplicate update mismatch'
                                aa,bb=a.snapshot()['base'],b.snapshot()['base']
                                for key in ('parameters','buffers','gradients','adam','steps','modes','native_rng'):
                                    assert close_enough(aa[key],bb[key]),'duplicate state mismatch '+key
                                if full:assert len(ref.read(65536))==65536
                                assert ref.read(65536)==bits(za[0]),'old reference prefix mismatch'
                        for h in hosts:h.check_frozen(True);h.close()
                        hosts=[];checks.append(dict(check='duplicate_four_vs_two',full_W_LSO=full,seed=seed,order=order,prefix=2,passed=True));persist()
            for cell in ('C4','V4'):
                h=Four(checkpoint,cell,SEEDS[0],cell,charge=charge);hosts=[h];meter(h)
                for row in panels[0][:2]:
                    x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];z,detail=h.keyed_step(x,style(x,SEEDS[0],idx));assert torch.isfinite(z).all();packed_sizes.append(len(gzip.compress(bits(z[0]),compresslevel=1)))
                h.check_frozen(True);h.close();hosts=[];persist();checks.append(dict(check='finite_two_arrivals',cell=cell,passed=True))
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==52
            assert state['physical_forwards']==524
            projection=max(timings)*2304*1.5+3600
            projected_bytes=max(packed_sizes)*2304*4+64*1024**2
            current_bytes=sum(max(p.lstat().st_size,p.lstat().st_blocks*512) for p in Path(c['storage_root']).rglob('*') if not p.is_symlink())
            result=dict(status='PASS',admitted=projection<c['deadline']-time.time()-180 and current_bytes+projected_bytes<c['remote_normal_cap_bytes'],projected_additional_bytes=projected_bytes,storage_at_admission=current_bytes,checks=checks,projected_seconds=projection,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,short_prefix_not_full_trajectory_replay=True)
            write('public/QUALIFICATION.json',result)
        else:
            assert read(root/'public/QUALIFICATION.json')['admitted']
            assert read(root/'public/QUALIFY_WRAPPER.json')['exit_code']==0
            for s in SEEDS:
                for o in (0,1):
                    for cell in ('C4','V4'):
                        job=f'{cell}_s{s}_o{o}';folder=root/'jobs'/job;folder.mkdir();h=Four(checkpoint,cell,s,job,charge=charge);hosts=[h];meter(h)
                        start=time.time();f0=state['physical_forwards'];u0=state['physical_updates']
                        with gzip.GzipFile(filename=str(folder/'K.bits.gz'),mode='xb',compresslevel=1,mtime=0) as out,(folder/'trace.jsonl').open('x') as trace:
                            for visit,row in enumerate(panels[o]):
                                guard();x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];tick=time.time();z,detail=h.keyed_step(x,style(x,s,idx))
                                out.write(bits(z[0]));d=detail['diagnostics']
                                trace.write(json.dumps(dict(arrival=visit+1,loss=d['joint_loss'],update_norm=d['BN_update_norm'],gradient_norm=d['BN_gradient_norm'],seconds=time.time()-tick))+'\n')
                                state.update(job=job,arrival=visit+1)
                                if (visit+1)%8==0:out.flush();trace.flush();persist()
                            assert out.tell()==192*65536
                        assert h.native.steps==h.visits==192 and state['physical_updates']-u0==192
                        assert state['physical_forwards']-f0==192*11
                        h.check_frozen(True);h.close();hosts=[]
                        write('jobs/'+job+'/COMPLETE.json',dict(status='COMPLETE',cell=cell,seed=s,order=o,arrivals=192,forwards=state['physical_forwards']-f0,seconds=time.time()-start))
                        state['completed_jobs'].append(job);persist()
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==2304 and state['physical_forwards']==25344
            result=dict(status='SEALED',jobs=12,arrivals=2304,online_label_reads=0,online_score_reads=0)
            write('public/RAW_SEALED.json',result)
        # Original reader's one end check, with explicit allowed observed-image set.
        permit['reverify_images']={Path(p).resolve() for p in reader.used};assert permit['reverify_images']<=permit['opened_images']
        try:reader.after_check()
        finally:permit['reverify_images'].clear()
        state.update(status='COMPLETE',ended=time.time(),image_accesses=guard.extra['image_accesses']);persist()
        write('public/'+mode.upper()+'_ACCESS.json',dict(online_labels=0,online_scores=0,source_images=0,unique_images=len(permit['opened_images']),denial_probes=len(rejected)))
        return result
    except BaseException as exc:
        state.update(status='FAILED',ended=time.time(),error=type(exc).__name__,reason=str(exc));persist();raise
    finally:
        for h in hosts:h.close()
