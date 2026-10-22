"""Qualification and twelve label-free trajectories; score only in a later process."""
import json,os,runpy,time,gzip
from pathlib import Path
import numpy as np
SEEDS=(20260907,17011,29009)
def read(p):return json.loads(Path(p).read_text())
def run(c,write,mode):
    import torch
    from method import Plain,DuplicateFull,Previous,plain_oracle,close_enough,self_check,reset_episode
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
    helper=runpy.run_path('/tmp/n91/m1_helpers.py');cfg=dict(c)
    if mode=='formal':cfg['reference_predictions']={}
    permit,opened,rejected=helper['access_guard'](cfg)
    checkpoint=weights(c);reader=TargetReader(c['target_root'],256*1024**2,'image');hosts=[];timings=[]
    persist()
    try:
        if mode=='qualify':
            self_check();checks=[]
            def make(cell,seed,identity):
                return Plain(checkpoint,'C2',seed,identity,charge=charge) if cell=='C2' else Host(checkpoint,'VV',seed,identity,charge=charge)
            for s0 in SEEDS:
                for o in (0,1):
                    for cell in ('C2','VV'):
                        h=make(cell,s0,'reuse');hosts=[h];meter(h)
                        path=c['reference_predictions'][f'{cell}_{s0}_{o}']
                        with open(path,'rb') as ref:
                            for row in panels[o][:2]:
                                x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];pre=h.before(x)
                                tick=time.time();z,_=h.keyed_step(x,style(x,s0,idx));torch.cuda.synchronize();timings.append(time.time()-tick)
                                expected=bits(z[0]) if cell=='C2' else bits(torch.stack((pre[0],z[0])))
                                assert ref.read(len(expected))==expected,'reuse endpoint mismatch'
                        h.check_frozen(True);h.close();hosts=[];checks.append(dict(check='reuse',cell=cell,seed=s0,order=o,prefix=2,exact=True));persist()
            for cell in ('C2','VV'):
                a=make(cell,SEEDS[0],'episode');b=make(cell,SEEDS[0],'episode');hosts=[a,b]
                for h in hosts:meter(h)
                initial=b.snapshot();row=panels[0][0];x=image(reader,row,permit,guard);raw=style(x,SEEDS[0],0)
                za,_=a.keyed_step(x,raw);qa=a.target_cache[0].clone()
                zb,_=b.keyed_step(x,raw)
                assert torch.equal(za,zb) and close_enough(a.snapshot()['base'],b.snapshot()['base'])
                row=panels[0][1];y=image(reader,row,permit,guard);b.keyed_step(y,style(y,SEEDS[0],1))
                reset_episode(b,initial);zr,_=b.keyed_step(x,raw)
                assert torch.equal(za,zr) and torch.equal(qa,b.target_cache[0]),'predecessor dependence after reset'
                assert np.array_equal(raw,style(x,SEEDS[0],0))
                for h in hosts:h.check_frozen(True);h.close()
                hosts=[];checks.append(dict(check='episode',cell=cell,first_equal=True,predecessor_independent=True,own_q_exact=True,full_reset=True));persist()
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==32 and state['physical_forwards']==312
            projection=max(timings)*2304*1.5+3600
            result=dict(status='PASS',admitted=projection<c['deadline']-time.time()-180,checks=checks,projected_seconds=projection,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,qualified_step_seconds=timings,short_prefix_not_full_trajectory_replay=True)
            write('public/QUALIFICATION.json',result)
        else:
            assert read(root/'public/QUALIFICATION.json')['admitted']
            assert read(root/'public/QUALIFY_WRAPPER.json')['exit_code']==0
            for s in SEEDS:
                for o in (0,1):
                    for cell in ('E_C2','E_VV'):
                        job=f'{cell}_s{s}_o{o}';folder=root/'jobs'/job;folder.mkdir();h=Plain(checkpoint,'C2',s,job,charge=charge) if cell=='E_C2' else Host(checkpoint,'VV',s,job,charge=charge);hosts=[h];meter(h);initial=h.snapshot()
                        start=time.time();f0=state['physical_forwards'];u0=state['physical_updates']
                        with gzip.GzipFile(filename=str(folder/'K.bits.gz'),mode='xb',compresslevel=1,mtime=0) as out,(folder/'trace.jsonl').open('x') as trace:
                            for visit,row in enumerate(panels[o]):
                                guard();x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];tick=time.time();reset_episode(h,initial);z,detail=h.keyed_step(x,style(x,s,idx))
                                out.write(bits(z[0]));d=detail['diagnostics']
                                trace.write(json.dumps(dict(arrival=visit+1,loss=d['joint_loss'],update_norm=d['BN_update_norm'],gradient_norm=d['BN_gradient_norm'],seconds=time.time()-tick))+'\n')
                                state.update(job=job,arrival=visit+1)
                                if (visit+1)%8==0:out.flush();trace.flush();persist()
                            assert out.tell()==192*65536
                        assert h.native.steps==h.visits==1 and state['physical_updates']-u0==192
                        assert state['physical_forwards']-f0==192*9
                        h.check_frozen(True);h.close();hosts=[]
                        write('jobs/'+job+'/COMPLETE.json',dict(status='COMPLETE',cell=cell,seed=s,order=o,arrivals=192,forwards=state['physical_forwards']-f0,seconds=time.time()-start))
                        state['completed_jobs'].append(job);persist()
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==2304 and state['physical_forwards']==20736
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
