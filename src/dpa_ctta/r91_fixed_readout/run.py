"""Original VV writes, read-only existing q export, and matched static six-view control."""
import copy,gzip,json,os,runpy,signal,sys,time
from pathlib import Path
import numpy as np
from dpa_ctta.r85_migration.process import child,finish_child,alive,identity,terminate
SEEDS=(20260907,17011,29009)

def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def size(p):return sum(max(x.stat().st_size,x.stat().st_blocks*512) for x in p.rglob('*') if not x.is_symlink())+max(p.stat().st_size,p.stat().st_blocks*512)
def usage(c):
    a=size(Path(c['output_root']));b=size(Path(c['campaign_root']))
    if a>=c['output_cap_bytes']-c['recovery_reserve_bytes'] or b>=c['campaign_cap_bytes']-134217728:raise RuntimeError('allocated storage recovery reserve')
    return a

def export(h):
    import torch
    from dpa_ctta.r19_model_only.method import equal
    before=h.snapshot();q=h.target_cache[0].clone()
    assert torch.equal(q,torch.stack(h.weak).sigmoid().mean(0))
    assert q.shape==(1,2,512,512) and torch.isfinite(q).all()
    assert equal(before,h.snapshot()),'q export mutated full state'
    return q

def static(h,initial,x):
    import torch
    from dpa_ctta.b1_host import official
    from dpa_ctta.hosts.vptta import model_input_from_pixels
    from dpa_ctta.r19_model_only.method import equal
    saved=h.snapshot()
    try:
        h.restore(initial);h.collect=False;z=model_input_from_pixels(x,'fundus').to(h.device)
        api=official().Rotate_and_Flip();out=[]
        with torch.no_grad():
            first=h.native.model(z)[0];out.append(first.detach().cpu())
            for factor in range(5):out.append(api.inverse(h.native.model(api(z,factor))[0],factor).detach().cpu())
        q=__import__('torch').stack(out).sigmoid().mean(0)
    finally:h.restore(saved)
    assert equal(saved,h.snapshot()),'Q0 mutated full state'
    return q,out[0]

def cpu_check(c):
    import torch
    from dpa_ctta.b1_host import official
    t=time.time();a=torch.arange(2*9*9).reshape(1,2,9,9).float();f=official().Rotate_and_Flip()
    assert all(torch.equal(f.inverse(f(a,k),k),a) for k in range(5))
    q=torch.stack([a]*6).sigmoid().mean(0)
    assert torch.allclose(q,a.sigmoid(),atol=1e-7,rtol=1e-7)
    assert torch.equal(q>=.5,a>=0)
    z=torch.tensor([-.2,.2]).reshape(1,2,1,1);assert not bool((z.sigmoid()>=.5)[0,0]) and bool((z.sigmoid()>=.5)[0,1])
    save(Path(c['output_root'])/'public/CPU_CHECK.json',dict(status='PASS',inverse_all_five=True,identical_view_degeneracy=True,independent_heads=True,new_model_calls=0,new_GPU_updates=0,CPU_body_seconds=time.time()-t))

def worker(c):
    import torch
    from dpa_ctta.r85_loss_aggregation.method import Host
    from dpa_ctta.r79_mechanism_control.method import style
    from dpa_ctta.r19_model_only.method import equal
    from dpa_ctta.r19_model_only.runtime import weights,image
    from dpa_ctta.r7_target_screen.runner import TargetReader
    from dpa_ctta.r9_current_first.assets import gpu_policy
    root=Path(c['output_root']);stage=os.environ['STAGE'];slot=int(os.environ['SLOT'])
    dest=root/'private'/f'{stage}_{slot}';dest.mkdir()
    s=dict(status='RUNNING',started=time.time(),physical_updates=0,completed_physical_updates=0,physical_forwards=0,training_backwards=0,online_label_reads=0,source_image_reads=0)
    cap,fcap={'qualify':(6,78),'source':(0,1152),'formal':(576,5760)}[stage]
    hosts=[];times=[];qtimes=[]
    def persist():save(dest/'RUN_STATE.json',s)
    def guard():
        if time.time()>c['deadline']-180:raise TimeoutError('shared deadline')
    guard.extra={'image_accesses':0}
    def charge(k):
        guard();key={'attempt':'physical_updates','complete':'completed_physical_updates'}[k]
        if k=='attempt' and s[key]>=cap:raise RuntimeError('physical update cap')
        s[key]+=1
    def meter(h):
        def forward(*_):
            guard()
            if s['physical_forwards']>=fcap:raise RuntimeError('physical forward cap')
            s['physical_forwards']+=1
        def back(g):s['training_backwards']+=1;return g
        h.handles.append(h.native.model.register_forward_pre_hook(forward));h.handles.append(h.native.params[0].register_hook(back))
    def step(h,x,raw):
        tick=time.time();p=h.before(x);k,_=h.keyed_step(x,raw);torch.cuda.synchronize();times.append(time.time()-tick);persist();return p,k
    def q0(h,init,x):
        t=time.time();v=static(h,init,x);torch.cuda.synchronize();qtimes.append(time.time()-t);return v
    def packed(q):return np.packbits((q[0]>=.5).numpy()).tobytes()
    persist()
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][slot])
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        helper=runpy.run_path(str(root/'m1_helpers.py'));permit,opened,rejected=helper['access_guard'](c)
        checkpoint=weights(c);reader=TargetReader(c['target_root'],256*1024**2,'image')
        canonical=read(root/'private/screen_o0.json');content={r['image_sha256']:i for i,r in enumerate(canonical)};assert len(content)==192
        if stage=='qualify':
            a=Host(checkpoint,'VV',SEEDS[0],'qualification',charge=charge);b=Host(checkpoint,'VV',SEEDS[0],'qualification',charge=charge);hosts=[a,b]
            for h in hosts:assert sum(p.numel() for p in h.native.params)==19136;meter(h)
            initial=a.snapshot();x=image(reader,canonical[0],permit,guard)
            v0,_=q0(a,initial,x);v1,_=q0(a,initial,x);assert torch.equal(v0,v1)
            for i in range(3):
                x=image(reader,canonical[i],permit,guard);raw=style(x,SEEDS[0],i)
                pa,ka=step(a,x,raw);qa=export(a);pb,kb=step(b,x,raw)
                assert torch.equal(pa,pb) and torch.equal(ka,kb) and equal(a.snapshot(),b.snapshot())
                if i==0:
                    assert torch.equal(qa,v0)
                    vv,_=q0(a,initial,x);assert torch.equal(vv,v0)
            assert s['physical_updates']==6 and s['physical_forwards']==78
            save(root/'public/QUALIFICATION.json',dict(status='PASS',updates=6,forwards=78,training_backwards=6,read_export_full_snapshot_exact=True,export_and_original_PRE_K_full_state_exact=True,Q0_repeat_and_initial_q_exact=True,Q0_restore_full_state_exact=True,max_steady_transaction_seconds=max(times[2:]),max_static_seconds=max(qtimes),transaction_seconds=times,static_seconds=qtimes))
        elif stage=='source':
            h=Host(checkpoint,'VV',SEEDS[0],'static');hosts=[h];meter(h);initial=h.snapshot()
            with gzip.GzipFile(filename=str(root/'jobs/Q0.bits.gz'),mode='xb',compresslevel=1,mtime=0) as out,Path(c['reference_predictions']['static']).open('rb') as ref:
                for i,row in enumerate(canonical):
                    usage(c);x=image(reader,row,permit,guard);q,first=q0(h,initial,x)
                    frame=ref.read(131072);assert len(frame)==131072
                    assert np.packbits((first[0]>=0).numpy()).tobytes()==frame[65536:],'Q0 identity/P0 exact'
                    out.write(packed(q));s.update(arrival=i+1);persist()
                assert not ref.read(1)
            assert equal(h.snapshot(),initial)
            save(root/'jobs/Q0_COMPLETE.json',dict(arrivals=192,physical_forwards=1152,physical_updates=0,P0_identity_exact=True,no_cross_image_learning=True))
        else:
            assert read(root/'FORMAL_ADMISSION.json')['admitted']
            rows=read(root/'private'/f'screen_o{slot}.json')
            for seed in SEEDS:
                h=Host(checkpoint,'VV',seed,f's{seed}_o{slot}',charge=charge);hosts=[h];meter(h)
                folder=root/'jobs'/f's{seed}_o{slot}';folder.mkdir()
                with gzip.GzipFile(filename=str(folder/'Qt.bits.gz'),mode='xb',compresslevel=1,mtime=0) as out,Path(c['reference_predictions'][f'{seed}_{slot}']).open('rb') as ref:
                    for visit,row in enumerate(rows):
                        usage(c);x=image(reader,row,permit,guard);p,k=step(h,x,style(x,seed,content[row['image_sha256']]))
                        assert ref.read(131072)==np.packbits(np.stack([(z[0]>=0).numpy() for z in (p,k)])).tobytes(),'PRE/K exact reference'
                        # The existing incoming q is exported after the unchanged update.
                        out.write(packed(export(h)));s.update(seed=seed,arrival=visit+1);persist()
                    assert not ref.read(1)
                assert h.native.steps==h.visits==192;h.check_frozen(True);h.close();hosts=[]
                save(folder/'COMPLETE.json',dict(arrivals=192,updates=192,forwards=1920,PRE_K_reference_exact=True))
        for h in hosts:h.check_frozen(True);h.close()
        hosts=[]
        assert s['physical_updates']==s['completed_physical_updates']==s['training_backwards']==cap and s['physical_forwards']==fcap
        permit['reverify_images']={Path(p).resolve() for p in reader.used};assert permit['reverify_images']<=permit['opened_images']
        try:reader.after_check()
        finally:permit['reverify_images'].clear()
        save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),rejections=rejected))
        s.update(status='GPU_COMPLETE',ended=time.time());persist()
    except BaseException as exc:s.update(status='FAILED',error=type(exc).__name__,reason=str(exc),ended=time.time());persist();raise
    finally:
        for h in hosts:h.close()

def watchdog(c):
    root=Path(c['output_root']);save(root/'WATCHDOG.json',dict(pid=os.getpid(),start_ticks=identity(os.getpid()),deadline=c['deadline']))
    reason='absolute_deadline'
    while time.time()<c['deadline']-90:
        records=[read(p) for p in (root/'processes').glob('*.json')]
        if (root/'CONTROLLER_EXIT.json').exists() and all(not alive(r) for r in records):save(root/'WATCHDOG_EXIT.json',dict(reason='all_owned_children_retired',ended=time.time()));return
        try:usage(c)
        except RuntimeError:reason='storage_recovery_reserve';break
        time.sleep(5)
    records=[read(p) for p in (root/'processes').glob('*.json')]
    if (root/'LAUNCH.json').exists():records.append(read(root/'LAUNCH.json'))
    for r in records:
        try:terminate(r)
        except ProcessLookupError:pass
    time.sleep(5)
    for r in records:
        if alive(r):
            try:os.killpg(r['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    save(root/'HARD_STOP.json',dict(reason=reason,ended=time.time()))

def controller(c):
    root=Path(c['output_root']);attempts=[]
    def status(stage,**kw):
        save(root/'RUN_STATE.json',dict(stage=stage,shared_started=c['started'],deadline=c['deadline'],**kw))
        save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,GPU_process_seconds=sum(a['gpu_process_seconds'] for a in attempts),CPU_process_seconds=sum(a['cpu_process_seconds'] for a in attempts)))
    def group(mode,stage,slots):
        pending=list(slots);running=[];done=[]
        while pending or running:
            if time.time()>c['deadline']-180:raise TimeoutError('shared deadline admission')
            usage(c)
            if pending:
                try:running.append(child(c,mode,stage,pending[0]));pending.pop(0)
                except RuntimeError as e:
                    if 'GPU UUID/free-memory' not in str(e):raise
                    status('WAIT_RESOURCE',pending=pending)
            for item in running[:]:
                result=finish_child(c,item)
                if result is not None:attempts.append(result);done.append(result);running.remove(item);status(stage)
            if pending or running:time.sleep(5 if running else 30)
        return all(a['exit_code']==0 for a in done)
    try:
        status('QUALIFY')
        if not group('worker','qualify',[0]):raise RuntimeError('qualification NOT_EVALUABLE')
        q=read(root/'public/QUALIFICATION.json');projection=q['max_steady_transaction_seconds']*1152*1.5/2+q['max_static_seconds']*192*1.5+2400
        admitted=q['status']=='PASS' and projection<c['deadline']-time.time() and c['prior_updates']+1158<=5000 and c['prior_forwards']+12750<=50000
        save(root/'FORMAL_ADMISSION.json',dict(admitted=admitted,projected_parallel_static_score_seconds=projection,prior_updates=c['prior_updates'],prior_forwards=c['prior_forwards'],expected_updates=1158,expected_forwards=12750))
        if not admitted:raise RuntimeError('profile/budget admission failed')
        status('SOURCE')
        if not group('worker','source',[0]):raise RuntimeError('static control NOT_EVALUABLE')
        status('FORMAL')
        if not group('worker','formal',[0,1]):raise RuntimeError('formal NOT_EVALUABLE')
        status('SCORING')
        if not group('score','final',[0]):raise RuntimeError('score NOT_EVALUABLE')
        status('COMPLETE');save(root/'CONTROLLER_EXIT.json',dict(status='COMPLETE',ended=time.time()))
    except BaseException as e:status('FAILED',error=type(e).__name__,reason=str(e));save(root/'CONTROLLER_EXIT.json',dict(status='FAILED',ended=time.time()));raise

if __name__=='__main__':
    c=read(Path(os.environ['CONFIG']));mode=os.environ['MODE']
    {'selfcheck':cpu_check,'worker':worker,'controller':controller,'watchdog':watchdog}[mode](c)
