"""R85 reference replay/probes then R86 mixed continuous trajectories; labels terminal only."""
import copy,json,os,time,traceback
from pathlib import Path
import numpy as np
from .method import Host,Previous,CELLS,SEEDS,style,self_check,oracle
from .process import child,finish_child,alive
from ..r51_same_state.run import read,save
WIDTH=65536
QUALIFICATION_UPDATES=16


def jobs(stage,slot):
    return [dict(id=f'{stage}_{cell}_s{s}_o{slot}',cell=cell,seed=s,order=slot) for s in SEEDS for cell in (('VV',) if stage=='replay' else ('VB','BV'))]


def vector(host,gradient=False):
    import torch
    return torch.cat([(p.grad if gradient else p).detach().flatten().cpu() for p in host.native.params])


def cosine(a,b):return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-20))


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights,image
    from ..r19_model_only.method import equal
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    stage=os.environ['STAGE'];slot=int(os.environ['SLOT']);root=Path(c['output_root']);start=time.time();host=reference=None
    dest=root/'private'/f'{stage}_worker{slot}';dest.mkdir()
    state=dict(status='QUALIFYING' if stage=='qualify' else 'RUNNING',started=start,arrivals=0,physical_updates=0,completed_physical_updates=0,physical_forwards=0,diagnostic_backwards=0,training_backwards=0,completed_jobs=[])
    def persist():save(dest/'RUN_STATE.json',state)
    def charge(kind):state[{'attempt':'physical_updates','complete':'completed_physical_updates','forward':'physical_forwards','diagnostic_backward':'diagnostic_backwards'}[kind]]+=1
    def meter(h):h.handles.append(h.native.model.register_forward_hook(lambda *args:charge('forward')))
    def guard():
        if time.time()>c['deadline']-180:raise TimeoutError('absolute cooperative deadline')
    guard.extra={'image_accesses':0}
    def storage_guard():
        used=0
        for p in root.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink():used+=p.stat().st_size
            except FileNotFoundError:pass
        if used>c['output_cap_bytes']-128*1024**2:raise RuntimeError('full experiment write cap')
        state['observed_root_bytes']=used
    def step(h,x,raw):
        guard();result=h.keyed_step(x,raw);state['training_backwards']+=1;return result
    persist()
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][slot]);torch.cuda.set_per_process_memory_fraction(1/3,0)
        permit,opened,rejected=filesystem_guard(c);checkpoint=weights(c);reader=TargetReader(c['target_root'],256*1024**2,'image')
        canonical=read(root/'private/screen_o0.json');content={r['image_sha256']:i for i,r in enumerate(canonical)}
        if len(content)!=192:raise ValueError('frozen unique cohort')
        if stage=='qualify':
            checks={};timings={}
            xs=[image(reader,r,permit,guard) for r in canonical[:2]]
            for cell in CELLS:
                host=Host(checkpoint,cell,SEEDS[0],'qualification',charge=charge);meter(host)
                reference=Previous(checkpoint,'VERTEX' if cell=='VV' else 'BARY',SEEDS[0],'qualification',charge=charge);meter(reference)
                if cell in ('VB','BV'):reference._criterion=oracle(reference,cell)
                for i,x in enumerate(xs):
                    raw=style(x,SEEDS[0],i);saved=host.snapshot();host.before(x)
                    if not equal(saved,host.snapshot()):raise ValueError('PRE mutated full state')
                    host.probe=cell=='VV';f=state['physical_forwards'];tick=time.time();z,detail=step(host,x,raw);torch.cuda.synchronize();timings[cell]=time.time()-tick
                    if state['physical_forwards']-f!=9:raise ValueError('nine physical step forwards')
                    rz,_=step(reference,x,raw)
                    if not torch.equal(z,rz) or not equal(host.snapshot()['base'],reference.snapshot()['base']):raise ValueError('oracle output/gradient/BN/Adam/RNG core exact rejected')
                    pred=torch.stack(reference.weak).sigmoid()
                    from ..r20_model_only_search.method import weights_for
                    if any(not torch.equal(a,b) for a,b in zip(host.target_cache,(pred.mean(0),weights_for(pred,.1,2.)))):raise ValueError('q/W exact provenance')
                    if host.native.steps!=i+1 or detail['native'][0]['counts']['base_adam']!=1:raise ValueError('Adam continuity')
                host.check_frozen(True);reference.check_frozen(True);host.close();host=None;reference.close();reference=None;persist()
                checks[cell]=dict(status='PASS',output_gradient_corestate_exact=True,PRE_full_restore=True,current_q_wholeW_exact=True,BN_only_parameters=19136,continuous_Adam=True)
            from ..r30_state_history.method import FrozenHost
            source=FrozenHost(checkpoint,'S_SOURCE',SEEDS[0]);source.model.register_forward_hook(lambda *args:charge('forward'))
            for x in xs:
                a,_=source.step(x);b,_=source.step(x)
                if not torch.equal(a,b):raise ValueError('frozen source-only repeat parity')
            source.check_frozen();source.close();del source
            if state['physical_updates']!=16 or state['physical_forwards']!=156 or state['diagnostic_backwards']!=16:raise ValueError('qualification physical cost contract')
            save(root/'public/QUALIFICATION.json',dict(status='PASS',cells=checks,physical_updates=16,physical_forwards=156,diagnostic_backwards=16,steady_step_seconds=timings,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,online_label_reads=0,source_image_reads=0))
        elif stage=='source':
            from ..r30_state_history.method import FrozenHost
            source=FrozenHost(checkpoint,'S_SOURCE',SEEDS[0]);source.model.register_forward_hook(lambda *args:charge('forward'))
            folder=root/'jobs/source_only';folder.mkdir()
            with (folder/'predictions.bits').open('xb') as f:
                for row in canonical:
                    guard();x=image(reader,row,permit,guard);z,_=source.step(x);f.write(np.packbits((z[0]>=0).numpy()).tobytes());state['arrivals']+=1
                    if state['arrivals']%16==0:storage_guard();persist()
            source.check_frozen();source.close();del source
            save(folder/'COMPLETE.json',dict(arrivals=192,physical_forwards=192,physical_updates=0,BN_mode='eval_stored_source_running_statistics',augmentation=False,online_label_reads=0))
            state['completed_jobs'].append('source_only')
        elif stage in ('replay','candidate'):
            if not read(root/'FORMAL_ADMISSION.json')['admitted']:raise ValueError('qualification/admission barrier')
            probes=set(read(root/'private/probes.json')[str(slot)])
            expected_predictions=read(root/'private/reference_bits.json') if stage=='replay' else None
            for job in jobs(stage,slot):
                folder=root/'jobs'/job['id'];folder.mkdir();host=Host(checkpoint,job['cell'],job['seed'],job['id'],charge=charge);meter(host)
                rows=read(root/'private'/f'screen_o{slot}.json');before_updates=state['physical_updates'];nprobe=0
                with (folder/'predictions.bits').open('xb') as bits,(folder/'diagnostics.jsonl').open('x') as trace:
                    for visit,row in enumerate(rows):
                        guard();tick=time.time();x=image(reader,row,permit,guard);raw=style(x,job['seed'],content[row['image_sha256']]);pre=host.before(x);diagnostic={}
                        if stage=='replay' and visit in probes:
                            nprobe+=1;saved=host.snapshot();inbound=vector(host);vectors={};outputs={};targets=None;details={}
                            for cell in ('VB','BV','BB','VV'):
                                host.restore(saved);host.cell=cell;host.arm='VERTEX' if cell=='VV' else 'BARY';host.probe=cell=='VV'
                                z,detail=step(host,x,raw);g=vector(host,True);delta=vector(host)-inbound
                                vectors[cell]=(g,delta);outputs[cell]=z;details[cell]=detail['diagnostics']
                                if targets is not None and not equal(targets,host.target_cache):raise ValueError('probe q/W changed')
                                targets=host.target_cache
                                if cell!='VV':host.restore(saved)
                            residual=vectors['VV'][0]-vectors['VB'][0]-vectors['BV'][0]+vectors['BB'][0]
                            scale=max(float(v[0].abs().max()) for v in vectors.values())
                            if float(residual.abs().max())>2e-6+1e-4*scale:raise ValueError('same-state gradient factorial identity')
                            predicted={}
                            cg=host.component_gradients
                            for cell in CELLS:
                                pg=sum((cg['B_'+cell[0]+str(ch)]+.1*cg['S_'+cell[1]+str(ch)]) for ch in (0,1))
                                if not torch.allclose(pg,vectors[cell][0],atol=2e-6,rtol=1e-4):raise ValueError('channel/component gradient sum identity')
                                predicted[cell]=dict(gradient_norm=float(vectors[cell][0].norm()),Adam_delta_norm=float(vectors[cell][1].norm()),gradient_cosine_VV=cosine(vectors[cell][0],vectors['VV'][0]),Adam_delta_cosine_VV=cosine(vectors[cell][1],vectors['VV'][1]),**details[cell]['loss_parts'])
                            diagnostic=dict(cells=predicted,gradient_factorial_max_abs=float(residual.abs().max()),component_norms={k:float(v.norm()) for k,v in cg.items()},OD_OC_cosines={cell:cosine(cg['B_'+cell[0]+'0']+.1*cg['S_'+cell[1]+'0'],cg['B_'+cell[0]+'1']+.1*cg['S_'+cell[1]+'1']) for cell in CELLS})
                            torch.save(dict(gradients={k:g for k,(g,d) in vectors.items()},Adam_deltas={k:d for k,(g,d) in vectors.items()},component_gradients=cg),folder/f'probe_{visit}.private.pt')
                            with (folder/f'probe_{visit}.bits').open('xb') as f:f.write(np.packbits(np.stack([(outputs[k][0]>=0).numpy() for k in CELLS])).tobytes())
                            post=outputs['VV'];detail={'diagnostics':details['VV']};host.probe=False
                        else:post,detail=step(host,x,raw)
                        packed=np.packbits(np.stack([(z[0]>=0).numpy() for z in (pre,post)])).tobytes()
                        if stage=='replay':
                            expected=expected_predictions[f's{job["seed"]}_o{slot}'][visit]
                            import hashlib
                            if hashlib.sha256(packed).hexdigest()!=expected:raise ValueError('retained VV PRE/K replay exact parity rejected')
                        bits.write(packed);trace.write(json.dumps(dict(visit=visit+1,K=detail['diagnostics'],probe=diagnostic,seconds=time.time()-tick),allow_nan=False)+'\n')
                        state['arrivals']+=1
                        if (visit+1)%16==0:
                            bits.flush();trace.flush();storage_guard();state.update(job=job['id'],visit=visit+1);persist()
                    if bits.tell()!=192*2*WIDTH:raise ValueError('prediction coverage')
                expected=192+3*nprobe
                if state['physical_updates']-before_updates!=expected:raise ValueError('job physical accounting')
                host.check_frozen(True);torch.save(host.snapshot()['base'],folder/'terminal.private.pt');host.close();host=None
                save(folder/'COMPLETE.json',dict(**job,arrivals=192,probes=nprobe,physical_updates=expected,online_label_reads=0,reference_PRE_K_exact=stage=='replay'))
                state['completed_jobs'].append(job['id']);persist();storage_guard()
        else:raise ValueError('unregistered stage')
        if state['physical_updates']!=state['completed_physical_updates']:raise ValueError('attempt/complete coverage')
        for path,(_,ident) in reader.used.items():
            if file_identity(path)!=ident:raise ValueError('input changed')
        save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),rejections=rejected))
        state['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        state.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        if reference is not None:reference.close()
        state.update(ended=time.time(),gpu_body_seconds=time.time()-start);persist()


def controller(c):
    root=Path(c['output_root']);attempts=[];status=dict(status='QUALIFICATION',started=time.time(),source_commit=c['source_commit'])
    with (root/'START_ONCE.json').open('x') as f:json.dump(status,f)
    def publish():
        save(root/'RUN_STATE.json',status);save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,gpu_process_seconds=sum(x['gpu_process_seconds'] for x in attempts),cpu_process_seconds=c['preparation_cpu_seconds']+sum(x['cpu_process_seconds'] for x in attempts),deadline=c['deadline'],started=c['started'],all_workers_retired=all(not alive(read(p)) for p in (root/'processes').glob('*.json'))))
    def group(mode,stage,slots):
        active=[];failed=False
        try:
            for slot in slots:active.append(child(c,mode,stage,slot))
        finally:
            while active:
                for item in list(active):
                    receipt=finish_child(c,item)
                    if receipt is not None:attempts.append(receipt);failed|=receipt['exit_code']!=0;active.remove(item);publish()
                if active:time.sleep(5)
        return not failed
    try:
        if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU check required')
        publish()
        if not group('worker','qualify',[0]):raise RuntimeError('fresh qualification failed')
        q=read(root/'public/QUALIFICATION.json');projection=max(q['steady_step_seconds'].values())*3600/2*1.5+600
        admitted=time.time()+projection<c['deadline']-300
        admission=dict(admitted=admitted,qualification_updates=16,replay_updates=1296,candidate_updates=2304,diagnostic_backward_calls=384,projected_wall_seconds=projection,deadline=c['deadline'])
        save(root/'FORMAL_ADMISSION.json',admission);save(root/'public/FORMAL_ADMISSION.json',admission)
        if not admitted:raise RuntimeError('insufficient shared 24h clock')
        for stage in ('source','replay','candidate'):
            status['status']=stage.upper();publish()
            if not group('worker',stage,[0] if stage=='source' else [0,1]):raise RuntimeError(stage+' failed; preserve attempt')
        status['status']='SCORING';publish()
        if not group('score','final',[0]):raise RuntimeError('terminal CPU scoring failed')
        status['status']='COMPLETE';return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e),error_type=type(e).__name__);traceback.print_exc();return 1
    finally:
        status['ended']=time.time();publish();save(root/'CONTROLLER_EXIT.json',dict(status=status['status'],ended=time.time()))


def main():
    mode=os.environ['MODE']
    if mode=='self_check':
        self_check()
        from ..b1_host import official
        official()
        from .score import self_check as sc
        sc()
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU selfcheck initialized CUDA')
        return 0
    c=read(os.environ['CONFIG'])
    if mode=='watchdog':
        from .process import watchdog
        return watchdog(c)
    if mode=='score':
        from .score import score
        return score(c)
    return {'controller':controller,'worker':worker}[mode](c)

if __name__=='__main__':raise SystemExit(main())
