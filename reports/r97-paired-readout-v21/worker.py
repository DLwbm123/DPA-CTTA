"""Frozen VV replay with pure readout probes; no score or label IO."""
import gzip,json,os,runpy,time
from pathlib import Path
import numpy as np
from signal_v21 import compute,EPS

def read(p):return json.loads(Path(p).read_text())
def run(c,write,mode):
    import torch
    from dpa_ctta.r85_loss_aggregation.method import Host
    from dpa_ctta.r79_mechanism_control.method import style
    from dpa_ctta.r19_model_only.method import equal
    from dpa_ctta.r19_model_only.runtime import weights,image
    from dpa_ctta.r7_target_screen.runner import TargetReader
    from dpa_ctta.r9_current_first.assets import gpu_policy
    from dpa_ctta.b1_host import official
    from dpa_ctta.hosts.vptta import model_input_from_pixels
    root=Path(c['output_root']);old=read(c['prior_config']);r91=Path(old['output_root'])
    assert read(root/'public/ACTION_AUDIT.json')['admitted']
    if mode=='formal':assert read(root/'public/QUALIFICATION.json')['admitted']
    panels=[read(r91/'private'/f'screen_o{o}.json') for o in (0,1)]
    canonical={row['image_sha256']:i for i,row in enumerate(panels[0])};assert len(canonical)==192
    cfg=dict(old);cfg.update(c);cfg['reference_predictions']=dict(old['reference_predictions'],q0=str(r91/'jobs/Q0.bits.gz'))
    prior=read(root/'private/qualify_state.json') if mode=='formal' else dict(physical_updates=0,physical_forwards=0,training_backwards=0)
    state=dict(status='RUNNING',mode=mode,started=time.time(),physical_updates=0,completed_updates=0,physical_forwards=0,training_backwards=0,online_label_reads=0,online_score_reads=0)
    def persist():write('private/'+mode+'_state.json',state)
    def guard():
        if time.time()>c['deadline']-120:raise TimeoutError('shared deadline')
    guard.extra={'image_accesses':0}
    def charge(k):
        guard()
        if k=='attempt':
            assert prior['physical_updates']+state['physical_updates']<c['max_updates'];state['physical_updates']+=1
        elif k=='complete':state['completed_updates']+=1
        else:raise ValueError('unregistered backward')
    def meter(h):
        def forward(*_):
            guard();assert prior['physical_forwards']+state['physical_forwards']<c['max_forwards'];state['physical_forwards']+=1
        def backward(g):
            assert prior['training_backwards']+state['training_backwards']<c['max_backwards'];state['training_backwards']+=1;return g
        h.handles.append(h.native.model.register_forward_pre_hook(forward));h.handles.append(h.native.params[0].register_hook(backward))
    def probe(h,initial,x):
        saved=h.snapshot()
        try:
            h.restore(initial);h.collect=False;z=model_input_from_pixels(x,'fundus').to(h.device)
            api=official().Rotate_and_Flip();out=[]
            with torch.no_grad():
                out.append(h.native.model(z)[0].detach().cpu())
                for factor in range(5):out.append(api.inverse(h.native.model(api(z,factor))[0],factor).detach().cpu())
            logits=torch.stack(out);probs=logits.sigmoid()
        finally:h.restore(saved)
        assert equal(saved,h.snapshot()),'static probe altered full state'
        return probs[:,0],probs.mean(0)[0],out[0][0]
    def step(h,x,raw):
        p=h.before(x);k,_=h.keyed_step(x,raw)
        return p[0],k[0]
    def observe(h,p,k,static):
        v0,q,first=static;vt=torch.stack(h.weak).sigmoid()[:,0]
        assert vt.shape==v0.shape==(6,2,512,512)
        assert torch.equal(vt[0],p.sigmoid()),'incoming identity mismatch'
        # Original six weak logits are already inverse aligned in the base adapter.
        return compute(first.sigmoid().numpy(),p.sigmoid().numpy(),k.sigmoid().numpy(),q.numpy(),v0.numpy(),vt.numpy())
    def bits(z):return np.packbits(z.numpy()).tobytes()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][0])
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    helper=runpy.run_path(str(r91/'m1_helpers.py'));permit,opened,rejected=helper['access_guard'](cfg)
    checkpoint=weights(cfg);reader=TargetReader(cfg['target_root'],256*1024**2,'image');hosts=[];timings=[]
    persist()
    try:
        if mode=='qualify':
            a=Host(checkpoint,'VV',20260907,'qualification',charge=charge);b=Host(checkpoint,'VV',20260907,'qualification',charge=charge);hosts=[a,b]
            for h in hosts:meter(h)
            initial=a.snapshot();checks=[]
            for visit,row in enumerate(panels[0][:3]):
                guard();tick=time.time();x=image(reader,row,permit,guard);raw=style(x,20260907,visit)
                static=probe(a,initial,x)
                if visit==0:
                    repeat=probe(a,initial,x);assert all(torch.equal(u,v) for u,v in zip(static,repeat)), 'repeat static not exact'
                pa,ka=step(a,x,raw);sa=a.snapshot();obs=observe(a,pa,ka,static)
                assert equal(sa,a.snapshot()),'signal observer mutated host'
                pb,kb=step(b,x,raw)
                assert torch.equal(pa,pb) and torch.equal(ka,kb) and equal(a.snapshot(),b.snapshot()),'probed baseline continuation differs'
                torch.cuda.synchronize();timings.append(time.time()-tick);checks.append(dict(arrival=visit+1,PRE_K_full_state_exact=True,signal_state_exact=True));persist()
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==6 and state['physical_forwards']==84
            projection=max(timings)*384*1.5+600
            result=dict(status='PASS',admitted=projection<c['deadline']-time.time(),checks=checks,static_repeat_exact=True,epsilon=EPS,
                        projected_formal_and_score_seconds=projection,remaining_seconds=c['deadline']-time.time(),seconds_per_qualified_pair=timings,
                        note='projection conservatively uses double-host qualification time for each single-host formal arrival')
            write('public/QUALIFICATION.json',result)
        else:
            # Existing candidates are read-only endpoint references, never score labels.
            with gzip.open(cfg['reference_predictions']['q0'],'rb') as f:qrefs=[f.read(65536) for _ in range(192)];assert not f.read(1)
            assert all(len(x)==65536 for x in qrefs)
            for order in (0,1):
                h=Host(checkpoint,'VV',20260907,'formal_'+str(order),charge=charge);hosts=[h];meter(h);initial=h.snapshot();rows=[]
                with open(cfg['reference_predictions'][f'20260907_{order}'],'rb') as ref:
                    for visit,row in enumerate(panels[order]):
                        guard();tick=time.time();index=canonical[row['image_sha256']];x=image(reader,row,permit,guard)
                        static=probe(h,initial,x);p,k=step(h,x,style(x,20260907,index))
                        assert ref.read(131072)==bits(torch.stack((p,k))>=0),'P/K existing endpoint mismatch'
                        assert qrefs[index]==bits(static[1]>=.5),'Q existing endpoint mismatch'
                        obs=observe(h,p,k,static);obs.update(index=index,visit=visit,order=order);rows.append(obs)
                        state.update(order=order,arrival=visit+1);torch.cuda.synchronize();timings.append(time.time()-tick)
                        if (visit+1)%8==0:persist()
                    assert not ref.read(1)
                assert h.native.steps==h.visits==192;h.check_frozen(True);h.close();hosts=[]
                write('private/signals_'+str(order)+'.json',dict(status='SEALED',rows=rows,online_score_reads=0,online_label_reads=0,sealed_at=time.time()))
                persist()
            assert state['physical_updates']==state['completed_updates']==state['training_backwards']==384 and state['physical_forwards']==6144
            result=dict(status='SEALED',orders=2,arrivals=384,PRE_K_Q0_exact=True,online_label_reads=0,online_score_reads=0,
                        signal_only_no_new_masks=True,max_transaction_seconds=max(timings),mean_transaction_seconds=float(np.mean(timings)))
            write('public/SEALED.json',result)
        for h in hosts:h.check_frozen(True);h.close()
        hosts=[]
        permit['reverify_images']={Path(p).resolve() for p in reader.used};assert permit['reverify_images']<=permit['opened_images']
        try:reader.after_check()
        finally:permit['reverify_images'].clear()
        state.update(status='COMPLETE',ended=time.time(),image_accesses=guard.extra['image_accesses']);persist()
        write('public/'+mode.upper()+'_ACCESS.json',dict(online_labels=0,online_scores=0,source_images=0,unique_image_paths=len(permit['opened_images']),intentional_denial_probes=len(rejected)))
        return result
    except BaseException as exc:
        state.update(status='FAILED',ended=time.time(),error=type(exc).__name__,reason=str(exc));persist();raise
    finally:
        for h in hosts:h.close()
