"""Qualification and twelve label-free trajectories; score only in a later process."""
import json,os,runpy,time,gzip
from io_adapter import PackedWriter,TraceWriter,usage,emergency,read_json
import lzma
from pathlib import Path
import numpy as np
SEEDS=(20260907,17011,29009)
def read(p):return read_json(p)
def run(c,write,mode):
    import torch
    from method import Factorial,oracle,self_check
    from plain import Plain,close_enough
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
    assert mode=='formal'
    for k,v in dict(physical_updates=2249,physical_forwards=20241,training_backwards=2249).items():prior[k]+=v
    state=dict(status='RUNNING',mode=mode,started=time.time(),physical_updates=0,completed_updates=0,physical_forwards=0,training_backwards=0,completed_jobs=[],online_label_reads=0,online_score_reads=0)
    def persist():write('private/final_'+mode+'_state.json',state)
    def guard():
        if time.time()>c['deadline']-120:raise TimeoutError('absolute deadline')
        if usage(root)>=c['remote_normal_cap_bytes']:raise RuntimeError('unexpected allocation change')
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
            for s0 in SEEDS:
                for o in (0,1):
                    for cell in ('C2','VV'):
                        h=Factorial(checkpoint,cell,s0,'reuse',charge=charge);hosts=[h];meter(h)
                        paired=(s0==SEEDS[0] and o==0)
                        old=(Plain(checkpoint,'C2',s0,'reuse',charge=charge) if cell=='C2' else Host(checkpoint,'VV',s0,'reuse',charge=charge)) if paired else None
                        if old is not None:hosts.append(old);meter(old)
                        path=c['reference_predictions'][f'{cell}_{s0}_{o}']
                        with open(path,'rb') as ref:
                            for row in panels[o][:2]:
                                x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];pre=h.before(x);raw=style(x,s0,idx)
                                tick=time.time();z,_=h.keyed_step(x,raw);torch.cuda.synchronize();timings.append(time.time()-tick)
                                expected=bits(z[0]) if cell=='C2' else bits(torch.stack((pre[0],z[0])))
                                assert ref.read(len(expected))==expected,'reuse endpoint mismatch'
                                if old is not None:
                                    zz,_=old.keyed_step(x,raw)
                                    assert torch.equal(z,zz) and close_enough(h.snapshot()['base'],old.snapshot()['base']),'endpoint state/output'
                                    assert torch.equal(h.target_cache[0],old.target_cache[0]),'own q endpoint'
                        for hh in hosts:hh.check_frozen(True);hh.close()
                        hosts=[];checks.append(dict(check='reuse',cell=cell,seed=s0,order=o,prefix=2,exact_masks=True,paired_state=paired));persist()
            for cell in ('W_ONLY','LSO_ONLY'):
                a=Factorial(checkpoint,cell,SEEDS[0],'formula',charge=charge);b=Host(checkpoint,'VV',SEEDS[0],'formula',charge=charge)
                b._criterion=oracle(b,cell);hosts=[a,b]
                for h in hosts:meter(h)
                for row in panels[0][:2]:
                    x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];raw=style(x,SEEDS[0],idx)
                    tick=time.time();za,da=a.keyed_step(x,raw);torch.cuda.synchronize();timings.append(time.time()-tick);zb,db=b.keyed_step(x,raw)
                    assert torch.equal(za,zb) and close_enough(a.snapshot()['base'],b.snapshot()['base']),'ablation oracle state/output'
                    assert torch.equal(a.target_cache[0],b.target_cache[0]) and torch.equal(a.target_cache[1],b.target_cache[1])
                    assert da['diagnostics']['joint_loss']==db['diagnostics']['joint_loss']
                for h in hosts:h.check_frozen(True);h.close()
                hosts=[];checks.append(dict(check='direct_oracle',cell=cell,arrivals=2,loss_exact=True,output_exact=True,state_close=True,own_q_W_exact=True));persist()
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==36 and state['physical_forwards']==348
            projection=max(timings)*2304*1.5+3600
            size=c['storage_projection']['estimated_total_bytes']
            result=dict(status='PASS',admitted=projection<c['deadline']-time.time()-180 and size<c['remote_normal_cap_bytes'],checks=checks,projected_seconds=projection,projected_allocated_bytes=size,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,qualified_step_seconds=timings,short_prefix_not_full_trajectory_replay=True)
            write('public/QUALIFICATION.json',result)
        else:
            assert read(root/'public/QUALIFICATION.json')['admitted']
            assert read(root/'public/QUALIFY_WRAPPER.json')['exit_code']==0
            for s in SEEDS:
                for o in (0,1):
                    for cell in ('W_ONLY','LSO_ONLY'):
                        job=f'{cell}_s{s}_o{o}';folder=root/'jobs'/job
                        if (s,o,cell)==(29009,0,'LSO_ONLY'):folder=folder.with_name(job+'_recovery')
                        if (folder/'COMPLETE.json').exists():
                            assert read(folder/'COMPLETE.json')['arrivals']==192
                            state['completed_jobs'].append(job);continue
                        assert (s,o,cell)==(29009,1,'LSO_ONLY')
                        prefix=b'';old_trace=[]
                        if o==1:
                            prefix=lzma.LZMADecompressor().decompress((folder/'K.bits.xz').read_bytes())
                            with gzip.open(folder/'trace.jsonl.gz','rt') as f:old_trace=[json.loads(v) for v in f]
                            assert len(old_trace)==72 and 0<len(prefix)<72*65536
                            folder=root/'jobs'/(job+'_final')
                        h=Factorial(checkpoint,cell,s,job,charge=charge);hosts=[h];meter(h)
                        start=time.time();f0=state['physical_forwards'];u0=state['physical_updates']
                        with PackedWriter(folder/'K.bits.xz',c) as out,TraceWriter(folder/'trace.jsonl.gz',c) as trace:
                            for visit,row in enumerate(panels[o]):
                                guard();x=image(reader,row,permit,guard);idx=canonical[row['image_sha256']];tick=time.time();z,detail=h.keyed_step(x,style(x,s,idx))
                                encoded=bits(z[0]);d=detail['diagnostics'];state.update(job=job,arrival=visit+1)
                                if visit<len(old_trace):
                                    for key,dk in (('loss','joint_loss'),('update_norm','BN_update_norm'),('gradient_norm','BN_gradient_norm'),('BCE','BCE'),('LSO','LSO'),('W_mean','weight_mean'),('W_std','weight_std')):
                                        assert old_trace[visit][key]==d[dk],'replay diagnostic drift'
                                expected=prefix[visit*65536:(visit+1)*65536]
                                assert encoded[:len(expected)]==expected,'replay prediction prefix drift'
                                out.write(encoded)
                                trace.write(json.dumps(dict(arrival=visit+1,loss=d['joint_loss'],update_norm=d['BN_update_norm'],gradient_norm=d['BN_gradient_norm'],BCE=d['BCE'],LSO=d['LSO'],W_mean=d['weight_mean'],W_std=d['weight_std'],seconds=time.time()-tick))+'\n')
                                state.update(job=job,arrival=visit+1)
                                if (visit+1)%8==0:out.flush();trace.flush();persist()
                            assert out.tell()==192*65536
                        assert h.native.steps==h.visits==192 and state['physical_updates']-u0==192
                        assert state['physical_forwards']-f0==192*9
                        h.check_frozen(True);h.close();hosts=[]
                        write(str(folder.relative_to(root))+'/COMPLETE.json',dict(status='COMPLETE',cell=cell,seed=s,order=o,arrivals=192,forwards=state['physical_forwards']-f0,seconds=time.time()-start))
                        state['completed_jobs'].append(job);persist()
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==192 and state['physical_forwards']==1728
            assert len(state['completed_jobs'])==12
            result=dict(status='SEALED',jobs=12,arrivals=2304,online_label_reads=0,online_score_reads=0,reused_jobs=11,new_jobs=1,replayed_trace_rows=72,replayed_prefix_bytes=len(prefix))
            write('public/RAW_SEALED.json',result)
        # Original reader's one end check, with explicit allowed observed-image set.
        permit['reverify_images']={Path(p).resolve() for p in reader.used};assert permit['reverify_images']<=permit['opened_images']
        try:reader.after_check()
        finally:permit['reverify_images'].clear()
        state.update(status='COMPLETE',ended=time.time(),image_accesses=guard.extra['image_accesses']);persist()
        write('public/FINAL_'+mode.upper()+'_ACCESS.json',dict(online_labels=0,online_scores=0,source_images=0,unique_images=len(permit['opened_images']),denial_probes=len(rejected)))
        return result
    except BaseException as exc:
        state.update(status='FAILED',ended=time.time(),error=type(exc).__name__,reason=str(exc)[:512]);emergency(c,worker_state=state)
        try:persist()
        except Exception as persist_error:emergency(c,persist_error=str(persist_error)[:512])
        raise
    finally:
        for h in hosts:h.close()
