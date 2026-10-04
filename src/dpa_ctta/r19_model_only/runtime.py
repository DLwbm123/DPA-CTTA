"""Small model-only runner on the existing neutral subprocess/lease machinery."""
import concurrent.futures,hashlib,io,json,os,signal,sys,threading,time,traceback,subprocess,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from ..r10_12h_core.run import read,save,sha
from ..r16_evidence_correction import run as base
from ..r9_current_first.storage import lease
from ..r9_current_first.physical import Meter
from ..r9_current_first.assets import gpu_policy,available_memory
from ..r8_ba.journal import TargetJournal,_digest
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader,image_records
from .method import Host,readonly,probability,snapshot,restore_state,equal
ID='R19_MODEL_ONLY_VALIDATION_MEMORY'
ARMS=('G','G_HALF','G_VAL','G_MEM');WIDTH=65536*4;SOFT_WIDTH=2*512*512*4

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or sha(c)!=os.environ['RUN_CONFIG_SHA']:raise ValueError('config identity')
    return c

def filesystem_guard(c):
    """No source assets, labels, future image reads, or implicit pretrained weights."""
    root=Path(c['output_root']).resolve();nas=root.parent;checkpoint=Path(c['checkpoint_path']).resolve()
    code=[Path(os.environ[k]).resolve() for k in ('DPA_CTTA_BASE_ROOT','DPA_GRATA_ROOT','PYTHONPATH')]
    runtime=Path(sys.prefix).resolve();current={'image':None};opens=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(nas):return
        permitted=(p==checkpoint or p==current['image'] or p.is_relative_to(root) or p.is_relative_to(runtime)
                   or any(p.is_relative_to(a) for a in code) or p.suffix in ('.py','.pyc') and 'site-packages' in p.parts)
        if p.is_relative_to(root/'scorer'):permitted=False
        if not permitted:raise PermissionError('model-only filesystem allowlist rejected external asset')
        opens.add(str(p))
    sys.addaudithook(audit);return current,opens

def weights(c):
    p=Path(c['checkpoint_path']);raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=c['checkpoint_sha256']:raise ValueError('checkpoint identity')
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)

def amounts(c):
    r=Path(c['output_root']);done={};total=0.;phase={}
    for p in (r/'attempts').glob('*.json'):
        a=read(p);key=(a['phase'],a['attempt']);done[key]=1;v=a['cost'].get('gpu_seconds',0);total+=v;phase[a['phase']]=phase.get(a['phase'],0)+v
    for p in (r/'processes').glob('*.json'):
        a=read(p)
        if a['active'] and a['phase']!='score' and (a['phase'],a['attempt']) not in done:
            v=time.time()-a['started'];total+=v;phase[a['phase']]=phase.get(a['phase'],0)+v
    return total,phase

class Guard:
    def __init__(self,c,phase):
        self.c=c;self.root=Path(c['output_root']);self.phase=phase;self.deadline=float(os.environ['RUN_DEADLINE']);self.last=0;self.meter=None;self.extra=Counter()
    def observe(self,cost=None):
        if time.time()>self.deadline-10:raise TimeoutError('task hard deadline')
        if time.monotonic()-self.last<10:return
        total,ph=amounts(self.c)
        if self.phase!='score':
            if total>self.c['origin']['gpu_cap_seconds']-30:raise TimeoutError('8 GPU-hour cap')
            if self.phase=='preflight' and ph.get('preflight',0)>1800-30:raise TimeoutError('profile cap')
            if self.phase.startswith('online_') and sum(v for k,v in ph.items() if k.startswith('online_'))>21600-30:raise TimeoutError('trajectory plus diagnostic stage cap')
        size=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
        if size>self.c['origin']['cache_peak_cap_bytes']:raise RuntimeError('private disk cap')
        save(self.root/'live'/(self.phase+'.json'),dict(at=time.time(),pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],cost=cost or {},extra=dict(self.extra),disk_bytes=size))
        self.last=time.monotonic()
    def __call__(self):self.observe(self.meter.cost if self.meter else None)

def image(reader,row,permit,guard):
    permit['image']=Path(row['image_path']).resolve()
    try:out=reader.read(image_records([row])[0]);guard.extra['image_accesses']+=1;return out
    finally:permit['image']=None

def profile(c,guard,permit):
    root=Path(c['output_root']);rows=read(root/'private/ONLINE_o0.json');state=weights(c);reader=TargetReader(c['target_root'],256*1024**2,'image');profiles={};checks={}
    # Only four sequential arrivals. Reuse each already-arrived tensor across mechanical hosts.
    hosts={arm:Host(state,arm,sha(dict(config=sha(c),arm=arm,phase='profile'))) for arm in ARMS}
    for h in hosts.values():guard.meter.attach(h.native.model)
    refs=Host(state,'G','mechanical_native',diagnostics=False);guard.meter.attach(refs.native.model)
    from ..integrations.ctta_suite import build_reference_model
    from ..hosts.vptta import model_input_from_pixels
    c0,_=build_reference_model('fundus');c0.load_state_dict(state);c0.cuda().eval().requires_grad_(False)
    for module in c0.modules():
        if isinstance(module,torch.nn.BatchNorm2d):module.track_running_stats=False;module.running_mean=None;module.running_var=None;module.num_batches_tracked=None
    guard.meter.attach(c0)
    for i,row in enumerate(rows[:4]):
        x=image(reader,row,permit,guard)
        if i==0:
            with torch.no_grad():
                z0=c0(model_input_from_pixels(x,'fundus').cuda())[0].float().cpu();zf=torch.flip(c0(model_input_from_pixels(torch.flip(x,(-1,)),'fundus').cuda())[0].float().cpu(),(-1,));zq=torch.logit((.75*z0.sigmoid()+.25*zf.sigmoid()).clamp(1e-6,1-1e-6));ds=torch.where((z0.sigmoid()>=.5)!=(zq.sigmoid()>=.5),zq,z0)
            checks['C0_native_initial_exact']=torch.equal(z0,readonly(refs.native,x));checks['DS_no_auxiliary_asset_finite']=bool(torch.isfinite(ds).all())
        zref,_=refs.native.step(x);zref=zref.detach().float().cpu()
        for arm,h in hosts.items():
            if i==0:
                before=snapshot(h.native);readonly(h.native,x);checks[arm+'_readonly_exact']=equal(before,snapshot(h.native))
            start=time.perf_counter();z,t=h.step(x);
            # Charge a measured worst-case history pair even if profile decisions leave memory empty.
            if arm=='G_MEM' and t['memory_empty']:
                readonly(h.native,x);readonly(h.native,x);guard.extra['mechanical_history_forwards']+=2
            torch.cuda.synchronize();profiles.setdefault(arm,[]).append(time.perf_counter()-start)
            if arm=='G':
                checks[f'G_native_logits_{i}']=torch.equal(z,zref)
                a=snapshot(h.native);b=snapshot(refs.native)
                checks[f'G_native_state_{i}']=all(equal(a[k],b[k]) for k in ('parameters','buffers','adam','grata','native_rng','gradients'))
        if i==0:checks['empty_memory_matches_VAL']=equal(hosts['G_VAL'].last_outputs,hosts['G_MEM'].last_outputs)
    # CPU/synthetic snapshot test already checks continuation; real check uses generated input, no extra target read.
    h=hosts['G_VAL'];s=h.snapshot();g=torch.Generator().manual_seed(20261004);x=torch.rand((1,3,512,512),generator=g)
    z,t=h.step(x);end=h.snapshot();h.restore(s);zz,tt=h.step(x);checks['real_model_synthetic_resume_exact']=torch.equal(z,zz) and equal(end,h.snapshot())
    for h in list(hosts.values())+[refs]:h.check_frozen(True);h.close()
    if not all(checks.values()):raise ValueError('mechanical qualification failed: '+str([k for k,v in checks.items() if not v]))
    p=dict(status='PASS',checks=checks,profile=profiles,arrivals=4,logical_method_visits=16,labels_read=0,synthetic_extra_visits=2)
    audit=read(root/'MODEL_ONLY_AUDIT.json');audit.update(model_loaded=True,trainable_parameters=refs.native.names,trainable_count=sum(x.numel() for x in refs.native.params),real_mechanical_checks=checks,real_target_arrivals=4,source_reads=0,target_label_reads=0);save(root/'MODEL_ONLY_AUDIT.json',audit)
    save(root/'PREFLIGHT.json',p);return p

def online(c,guard,permit,job):
    root=Path(c['output_root']);rows=read(root/'private'/f'ONLINE_o{job["order"]}.json')
    if rows_sha(rows)!=c['manifests'][job['order']]['sha256']:raise ValueError('manifest mutated')
    h=Host(weights(c),job['arm'],sha(dict(config=sha(c),job=job)));guard.meter.attach(h.native.model)
    dest=root/'target'/job['id'];journal=TargetJournal(dest,h,job['id'],rows_sha(rows),WIDTH);reader=TargetReader(c['target_root'],256*1024**2,'image');journal.create()
    with (dest/'probabilities.f32').open('xb') as probs,(dest/'diagnostic-probabilities.f32').open('xb') as diag:
        for row in rows:
            guard();x=image(reader,row,permit,guard);start=time.perf_counter();z,t=h.step(x)
            guard.extra['history_tensor_accesses']+=0 if t['memory_empty'] else 2
            arrays=[z,h.last_outputs['pre'],h.last_outputs['candidate'],h.last_outputs.get('half',z)]
            packed=b''.join(np.packbits(a.float().sigmoid().numpy()[0]>=.5).tobytes() for a in arrays)
            probs.write(z.float().sigmoid().numpy().astype('<f4').tobytes());probs.flush()
            if t['diagnostic']:
                diag.write(np.stack([a.float().sigmoid().numpy()[0] for a in arrays[1:]]).astype('<f4').tobytes());diag.flush()
            t['seconds']=time.perf_counter()-start;t['counts']=guard.meter.cost.copy();journal.append(packed,t)
        os.fsync(probs.fileno());os.fsync(diag.fileno())
    result=journal.complete(len(rows));save(dest/'soft_complete.json',dict(bytes=(dest/'probabilities.f32').stat().st_size,sha256=_digest(dest/'probabilities.f32'),diagnostic_bytes=(dest/'diagnostic-probabilities.f32').stat().st_size,diagnostic_sha256=_digest(dest/'diagnostic-probabilities.f32'),dtype='float32 sigmoid',visits=len(rows)));h.close();return result

def worker():
    c=config();root=Path(c['output_root']);phase=os.environ['RUN_PHASE'];start=float(os.environ['RUN_STARTED']);guard=Guard(c,phase);cpu=phase=='score';meter=None;failure=None;result=None
    torch.set_num_threads(2);torch.set_num_interop_threads(2);timer=threading.Timer(max(.1,guard.deadline-time.time()),lambda:os.killpg(os.getpgrp(),signal.SIGTERM));timer.daemon=True;timer.start()
    try:
        if cpu:
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only scorer')
            from .score import score
            result=score(c,guard)
        else:
            assignment=json.loads(os.environ['RUN_ASSIGNMENT']);gpu_policy(assignment);available_memory(assignment,5*1024**3)
            mount=subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(root.parent.parent)],text=True).strip()
            if mount not in ('nfs','nfs4') or root.stat().st_dev!=root.parent.parent.stat().st_dev:raise RuntimeError('NAS mount mismatch')
            if shutil.disk_usage(root).free<c['origin']['cache_peak_cap_bytes']+10*1024**3:raise RuntimeError('NAS capacity reserve')
            probe=root/'private'/('probe-'+str(os.getpid()))
            with probe.open('xb') as f:f.write(b'probe');f.flush();os.fsync(f.fileno())
            if probe.read_bytes()!=b'probe':raise RuntimeError('NAS readback')
            probe.unlink()
            permit,opened=filesystem_guard(c)
            with lease(root/'leases'/f'gpu{assignment["physical_id"]}',dict(experiment_id=ID,GPU=assignment)),Meter(None,guard.observe) as meter:
                guard.meter=meter
                result=profile(c,guard,permit) if phase=='preflight' else online(c,guard,permit,json.loads(os.environ['RUN_JOB']))
            save(root/'private'/f'{phase}-asset-reads.json',sorted(opened))
    except BaseException as e:failure=dict(reason=str(e),error_type=type(e).__name__);traceback.print_exc()
    finally:timer.cancel()
    cost=meter.cost.copy() if meter else {};cost.update(guard.extra);cost['gpu_seconds']=0 if cpu else time.time()-start
    row=dict(status='FAILED' if failure else 'COMPLETE',phase=phase,attempt=int(os.environ.get('RUN_ATTEMPT',0)),started=start,ended=time.time(),wall_seconds=time.time()-start,cost=cost,failure=failure,result=result,code_sha=c['code_sha'],config_sha256=sha(c));save(root/'attempts'/f'{phase}.{row["attempt"]}.json',row);return row

def ledger(c,state):
    root=Path(c['output_root']);rows=[read(p) for p in sorted((root/'attempts').glob('*.json'))]
    x=dict(experiment_id=ID,status=state['status'],T0=c['origin']['T0'],wall_seconds=time.time()-c['origin']['T0_epoch'],gpu_seconds=sum(r['cost'].get('gpu_seconds',0) for r in rows),gpu_cap_seconds=28800,attempts=rows)
    save(root/'RESOURCE_LEDGER.json',x);return x

def admission(c):
    p=read(Path(c['output_root'])/'PREFLIGHT.json')['profile'];normal=sum(2*1951*(max(v[1:])+.20) for v in p.values());diagnostic=2*len(c['science']['diagnostic_positions'])*max(0,p['G'][0]-max(p['G'][1:]));normal=1.3*(normal+8*60);diagnostic=1.3*(diagnostic+120)
    charged=amounts(c)[0];a=dict(normal_gpu_seconds=normal,diagnostic_gpu_seconds=diagnostic,already_charged=charged,safety_factor=1.3,admitted=normal<=18000 and diagnostic<=3600 and charged+normal+diagnostic<=23400,scope='all eight full trajectories; no shortening or arm deletion')
    a['admitted']=a['admitted'] and time.time()+(normal+diagnostic)/2<c['origin']['normal_compute_deadline_epoch']
    save(Path(c['output_root'])/'PROFILE_ADMISSION.json',a);return a

def supervise():
    c=config();root=Path(c['output_root']);state=read(root/'RUN_STATE.json');jobs=[dict(id=f'{a}_o{o}',arm=a,order=o) for a in ARMS for o in (0,1)]
    with lease(root/'supervisor_control',dict(experiment_id=ID,config=sha(c))):
        if (root/'execution_started.json').exists():raise ValueError('execution already started')
        save(root/'execution_started.json',dict(at=time.time(),pid=os.getpid(),config_sha256=sha(c)))
        try:
            state.update(status='PREFLIGHT_RUNNING');save(root/'RUN_STATE.json',state)
            r=base.run_task(c,'preflight',c['gpu_assignments'][0],1800);state['jobs']['preflight']=r['status'];ledger(c,state)
            if r['status']!='COMPLETE':raise RuntimeError('NOT_RUN_QUALIFICATION')
            a=admission(c)
            if not a['admitted']:raise RuntimeError('NOT_RUN_BUDGET')
            state.update(status='TARGET_RUNNING');state['jobs'].update({j['id']:'NOT_RUN' for j in jobs});save(root/'RUN_STATE.json',state)
            for offset in range(0,8,2):
                for j in jobs[offset:offset+2]:state['jobs'][j['id']]='RUNNING'
                save(root/'RUN_STATE.json',state)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fs={ex.submit(base.run_task,c,'online_'+j['id'],gpu,10800,j):j for j,gpu in zip(jobs[offset:offset+2],c['gpu_assignments'])}
                    for f in concurrent.futures.as_completed(fs):
                        j=fs[f];r=f.result();state['jobs'][j['id']]=r['status'];save(root/'RUN_STATE.json',state);ledger(c,state)
                if any(state['jobs'][j['id']]=='FAILED' for j in jobs[offset:offset+2]):raise RuntimeError('target failure; preserve exact evidence before any recovery')
            state.update(status='TARGET_MATRIX_TERMINAL');save(root/'RUN_STATE.json',state)
            r=base.run_task(c,'score',None,7200);state['jobs']['score']=r['status'];state.update(status='COMPLETE' if r['status']=='COMPLETE' else 'SCORE_FAILED',target_scores_embargoed=r['status']!='COMPLETE')
        except BaseException as e:
            traceback.print_exc();state.update(status=str(e) if str(e).startswith('NOT_RUN_') else 'STOPPED',reason=str(e));state['jobs']={k:'NOT_RUN_STOPPED' if v in ('NOT_RUN','RUNNING') else v for k,v in state['jobs'].items()}
        state.update(ended=time.time(),delivery='PENDING_LOCAL_GITHUB');save(root/'RUN_STATE.json',state);ledger(c,state)
        from .score import report
        report(c,state)

def main():
    mode=os.environ['RUN_MODE']
    if mode=='worker':r=worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='supervise':supervise()
    elif mode=='watch':base.config=config;base.watch()
    else:raise ValueError('unknown mode')
