"""Thin future execution wiring to the unchanged verified IO and owned supervisor.

Default authorization is disabled. Stage I never invokes this module's launch.
"""
import gc,json,os,subprocess,sys,time,uuid
from pathlib import Path
from ..r1.plan import binding,bound,REF,GRATA
from ..r1.assets import checkpoint
from ..r1.evidence import write,output_bytes
from ..r1.run import claim,failure_scope
from .evaluation import current
from .plan import ROOT,SCIENCE_SHA,science,matrix,allocation,stream,stream_summary,registration_digest


def authorize(auth,reg,current_sha=None):
    if auth.get('enabled') is not True:raise PermissionError('execution disabled')
    sha=current_sha or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if len(sha)!=40 or auth.get('approved_code_sha')!=sha:raise PermissionError('authorized full code SHA mismatch')
    if auth.get('approved_science_sha256')!=SCIENCE_SHA or auth.get('approved_registration_digest')!=registration_digest(reg):raise PermissionError('authorized frozen input mismatch')
    if auth.get('approved_stream_digest')!=stream_summary(reg)['stream_digest'] or auth.get('approved_trajectory_count')!=70:raise PermissionError('explicit full R4T scope required')
    review=auth.get('review',{})
    if review.get('status') not in ('EXTERNAL_PASS','EXPLICIT_USER_WAIVER') or not isinstance(review.get('reference'),str) or not review['reference'].strip():raise PermissionError('external review or explicit user waiver required')
    ids=auth.get('allowed_physical_gpu_ids',[]);workers=auth.get('max_workers')
    if not ids or len(ids)>3 or len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids):raise PermissionError('explicit unique GPU list, max three')
    if type(workers) is not int or not 1<=workers<=len(ids):raise PermissionError('worker count')
    if not isinstance(auth.get('background_allowed'),bool):raise PermissionError('background policy required')
    return ids[:workers]


def caps():
    c=science()['resources']
    return dict(trajectory_seconds=c['trajectory_hours_cap']*3600,wall_seconds=c['wall_hours_cap']*3600,
        active_seconds=c['total_active_worker_hours_cap']*3600,bytes=c['new_private_output_bytes_cap'])


def counters(host):
    if not host.legacy:return host.total.copy()
    c=host.legacy.counts
    return dict(network_forwards=c['forwards'],loss_backward_calls=c['backwards'],jacobian_vjp_calls=0,adam_calls=c['base_adam'],actual_parameter_replacements=0)


from ..r3.execution import worker_environment,check_worker_environment,backend_policy


def smoke(state,device,out,identity,backend,checkpoint_io):
    import torch
    from ..source_pilot import seed_all
    from ..r1.host import Host as OldHost
    from ..b4_run import capture
    from ..host_diagnostic import close
    from ..m2_run import deterministic_smoke_pair
    from ..b3_runtime import process_audit
    from .host import Host,ARMS
    from ..r3.testing import ready
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels
    total=dict(network_forwards=0,loss_backward_calls=0,jacobian_vjp_calls=0,adam_calls=0,actual_parameter_replacements=0)
    saved={};evidence={}
    try:
        with deterministic_smoke_pair():
            paired_comparison_backend=backend_policy()
            for arm in ('OLD_C','OLD_RP',*ARMS):
                seed_all(20260907)
                h=OldHost('C' if arm=='OLD_C' else 'C_PCA_REGION',state,device) if arm.startswith('OLD_') else Host(arm,state,device)
                # GPU process visibility requires the existing host allocation.
                if arm=='OLD_C' and torch.device(device).type=='cuda':process_audit(out,'smoke')
                if sum(p.numel() for p in (getattr(h,'bn_params',None) or h.params))!=19136:raise ValueError('registered affine dimensions')
                for step in range(2):
                    if step==1:
                        memory=h.contexts.slots[0]['memory'] if getattr(h,'contexts',None) else h.memory
                        if memory is not None:ready(memory)
                    z,row=h.step(pixels('fundus',step))
                    if arm.startswith('OLD_'):
                        c=row['counts'];c=dict(network_forwards=c['forwards'],loss_backward_calls=c['backwards'],jacobian_vjp_calls=0,adam_calls=c['base_adam'],actual_parameter_replacements=0)
                        saved[arm,step]=(z.detach().cpu(),capture(h))
                    else:
                        c=row['r4t']['counts'];evidence[arm+':'+str(step)]=row['r4t']
                        if arm in ('C','RP'):
                            close(z.cpu(),saved['OLD_'+arm,step][0]);close(capture(h.legacy),saved['OLD_'+arm,step][1])
                        elif step==0 and arm in ('MT','MT_RP','FT','FT_RP'):
                            close(z.cpu(),saved['OLD_C',0][0]);close([p.detach().cpu() for p in h.params],saved['OLD_C',0][1]['affine']);close(h.rng,saved['OLD_C',0][1]['rng'],exact=True)
                    if not arm.startswith('OLD_'):
                        from .evaluation import auxiliary
                        payload=h.take_evaluation()
                        if arm=='C' and step==0:saved['C_initial_q']=payload['q'].clone()
                        if arm.startswith('K') and step==0:close(payload['q'],saved['C_initial_q'])
                        evidence[arm+':'+str(step)]['auxiliary']=auxiliary(payload,torch.zeros_like(payload['q']));payload.clear()
                    for key in total:total[key]+=c[key]
                h.finish(state);del h,z;gc.collect()
        if any(total[k]!=v for k,v in dict(network_forwards=260,loss_backward_calls=32,adam_calls=32).items()) or total['jacobian_vjp_calls']>0:
            raise ValueError('32-update mechanical budget')
        write(out/'smoke.completion.json',dict(binding=identity,status='MECHANICAL_SMOKE_COMPLETE',physical=total,evidence=evidence,backend=backend,paired_comparison_backend=paired_comparison_backend,checkpoint_io=checkpoint_io))
    except BaseException as exc:
        write(out/'smoke.failure.json',dict(binding=identity,status='INCOMPLETE',physical=total,reason=str(exc),scope=failure_scope(exc)));raise


def trajectory(job,state,reg,out,identity,backend,checkpoint_io):
    import torch
    from ..source_pilot import seed_all
    from ..b3_runtime import process_audit
    from .host import Host
    p=claim(out,job['job_id']);h=None;written=0;started=time.monotonic();limits=caps()
    try:
        seed_all(science()['seed']);h=Host(job['arm'],state,'cuda:0');torch.cuda.reset_peak_memory_stats();process_audit(p,'run')
        with (p/'records.jsonl').open('x') as log:
            for entry in stream(reg,job['order']):
                if time.monotonic()-started>limits['trajectory_seconds']:raise RuntimeError('trajectory cap')
                record=current(h,entry)
                row=dict(binding=identity,arm=job['arm'],order=job['order'],**{k:entry[k] for k in ('group_id','sample_id','domain','subset')},**record,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();written+=1
                if written%64==0 and output_bytes(out)>limits['bytes']:raise RuntimeError('output cap')
        h.finish(state)
        if written!=job['records'] or time.monotonic()-started>limits['trajectory_seconds']:raise ValueError('trajectory count/time')
        write(p/'completion.json',dict(binding=identity,status='TRAJECTORY_COMPLETE',records=written,physical=counters(h),backend=backend,checkpoint_io=checkpoint_io,seconds=time.monotonic()-started))
    except BaseException as exc:
        write(p/'failure.json',dict(binding=identity,status='INCOMPLETE',records=written,physical=None if h is None else counters(h),reason=str(exc),scope=failure_scope(exc)));raise
    finally:del h;gc.collect()


def context():
    packet=json.loads(Path(os.environ['RUN_PACKET']).read_text());reg=packet['assets']['registration']
    ids=authorize(packet['authorization'],reg);index=int(os.environ['RUN_WORKER'])
    expected=dict(code_sha=packet['authorization']['approved_code_sha'],science_sha256=SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest=stream_summary(reg)['stream_digest'])
    if any(packet['binding'].get(k)!=v for k,v in expected.items()):raise PermissionError('execution binding')
    if [d['index'] for d in packet['devices']]!=ids or not 0<=index<len(ids):raise PermissionError('worker devices')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=packet['devices'][index]['uuid']:raise PermissionError('worker UUID')
    jobs=matrix()['jobs']
    if packet['jobs']!=jobs or packet['schedule']!=allocation(jobs,len(ids)):raise PermissionError('complete frozen schedule')
    return packet,reg,index,Path(packet['out'])


def worker():
    # Authorize before loading weights, querying a device or decoding an asset.
    packet,reg,index,out=context()
    check_worker_environment(os.environ.get('RUN_MODE'),os.environ)
    identity=binding(packet,index)
    import torch
    from ..source_pilot_release import environment
    torch.set_num_threads(2)
    if os.environ['RUN_MODE']=='smoke':
        p=claim(out,'device'+str(index))
        try:
            state,io=checkpoint(reg);backend=environment()|backend_policy();smoke(state,'cuda:0',p,identity,backend,io)
        except BaseException as exc:
            if not (p/'smoke.failure.json').exists():write(p/'smoke.failure.json',dict(binding=identity,status='INCOMPLETE',reason=str(exc),scope=failure_scope(exc)))
            raise
    elif os.environ['RUN_MODE']=='formal':
        job=next(j for j in packet['jobs'] if j['job_id']==os.environ['RUN_JOB'])
        if next(a['worker'] for a in packet['schedule']['assignments'] if a['job_id']==job['job_id'])!=index:raise PermissionError('job rotation')
        p=out/('device'+str(index));proof=json.loads((p/'smoke.completion.json').read_text());bound(proof,identity)
        if proof['status']!='MECHANICAL_SMOKE_COMPLETE' or list(p.glob('*failure.json')):raise PermissionError('failed/missing smoke')
        for key,value in dict(network_forwards=260,loss_backward_calls=32,adam_calls=32).items():
            if proof['physical'][key]!=value:raise PermissionError('smoke counts')
        if not 0<=proof['physical']['jacobian_vjp_calls']<=0:raise PermissionError('smoke VJP count')
        state,io=checkpoint(reg);backend=environment()|backend_policy();trajectory(job,state,reg,out,binding(packet,index,job),backend,io)
    else:raise PermissionError('unknown worker mode')


def launch(auth,assets,output):
    reg=assets['registration'];ids=authorize(auth,reg)
    for order in range(5):stream(reg,order)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('clean reviewed checkout required')
    for key,sha in (('ctta_dependency_root',REF),('grata_root',GRATA)):
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=assets[key],text=True).strip()!=sha:raise ValueError('dependency commit')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader'],text=True)
    mapping={int(row[0]):dict(uuid=row[1].strip(),model=row[2].strip(),free_MiB=int(row[3].split()[0])) for row in [line.split(',') for line in rows.splitlines()]}
    devices=[dict(index=i,**mapping[i]) for i in ids]
    if any(d['free_MiB']<4096 for d in devices):raise RuntimeError('insufficient peak headroom; no waiting')
    out=Path(output).resolve();root=Path(assets['private_storage_root']).resolve()
    if out.is_relative_to(ROOT) or not out.is_relative_to(root):raise ValueError('new private output location')
    fs=os.statvfs(out.parent)
    if fs.f_bavail*fs.f_frsize<caps()['bytes']+1024**3:raise RuntimeError('storage capacity')
    import tempfile
    with tempfile.TemporaryFile(dir=out.parent) as probe:
        probe.write(b'probe');probe.flush();os.fsync(probe.fileno());probe.seek(0)
        if probe.read()!=b'probe':raise RuntimeError('storage probe')
    out.mkdir(mode=0o700);jobs=matrix()['jobs']
    identity=dict(run_id=uuid.uuid4().hex,code_sha=auth['approved_code_sha'],science_sha256=SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest=stream_summary(reg)['stream_digest'])
    packet=dict(binding=identity,authorization=auth,assets=assets,out=str(out),devices=devices,jobs=jobs,schedule=allocation(jobs,len(devices)))
    write(out/'packet.private.json',packet);write(out/'receipt.json',{k:packet[k] for k in ('binding','devices','jobs','schedule')})
    from ..b3_runtime import ENTRY
    from ..r1.supervise import supervise
    def start(spec,log):
        slot=spec['worker'];env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join([str(ROOT/'src'),assets.get('dependency_pythonpath','')]),
            DPA_CTTA_BASE_ROOT=assets['ctta_dependency_root'],DPA_GRATA_ROOT=assets['grata_root'],CUDA_VISIBLE_DEVICES=devices[slot]['uuid'],
            RUN_FILE=str(ROOT/'scripts/run_r4t.py'),RUN_MODE=spec['phase'],RUN_WORKER=str(slot),RUN_JOB=spec['key'],RUN_PACKET=str(out/'packet.private.json'))
        env=worker_environment(env,spec['phase'])
        return subprocess.Popen([sys.executable,'-c',ENTRY],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    supervise(out,packet,start,caps=caps())
    from .analyze import recompute
    return recompute(out,reg)
