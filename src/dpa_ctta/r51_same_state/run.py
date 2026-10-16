"""One bounded GPU process, then an independent CPU scorer; never resume/retry."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback


def save(path, value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def worker(c):
    import numpy as np
    import torch
    from ..r37_cw_orientation.method import Host,candidates
    from ..r19_model_only.runtime import weights,image
    from ..r19_model_only.method import equal
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    from .diagnostic import one_pair,mask,assert_native_equivalence,BRANCHES

    root=Path(c['output_root']);start=float(os.environ['DIAG_STARTED'])
    deadline=start+c['gpu_cap_seconds']-20
    status={'status':'QUALIFYING','formal_pairs':0,'started':start}
    save(root/'RUN_STATE.json',status)
    timer=threading.Timer(max(.1,deadline-time.time()),lambda:os.kill(os.getpid(),signal.SIGTERM));timer.daemon=True;timer.start()
    host=None;phase_deadline=start+c['qualification_cap_seconds']
    def guard():
        if time.time()>min(deadline,phase_deadline)-5:
            raise TimeoutError('fixed phase/total budget; no retry')
    guard.extra={'image_accesses':0}
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        gpu_policy(c['gpu'])
        permit,opened,rejected=filesystem_guard(c)
        state=weights(c)
        anchors=[]
        for order in c['orders']:
            raw=(root/'private'/f'anchor_o{order}.pt').read_bytes()
            meta=read(root/'private'/f'anchor_o{order}.json')
            # Preserve the existing journal's mandatory integrity mechanism.
            if hashlib.sha256(raw).hexdigest()!=meta['sha256']:
                raise ValueError('existing anchor integrity mismatch')
            payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            anchor=payload['host']
            if payload['schema']!='R8_TARGET_JOURNAL_V1' or meta['visits']!=1900 or anchor['visits']!=1900 or anchor['steps']!=1900:
                raise ValueError('only the two frozen visit1900 anchors are admitted')
            anchors.append(anchor)
        reader=TargetReader(c['target_root'],256*1024**2,'image')
        qualification=read(root/'private/qualification.json');panel=read(root/'private/images.json')
        if len(panel)!=48 or len(anchors)!=2 or c['total_pairs']!=96:
            raise ValueError('frozen matrix size')
        checks=[];timings=[]
        for order,anchor in enumerate(anchors):
            guard()
            host=Host(state,candidates()[3],17011,anchor['context']['sha256'],device='cuda:0')
            host.restore(anchor)
            if not equal(anchor,host.snapshot()):raise ValueError('restored snapshot differs from serialized anchor')
            pixels=image(reader,qualification[order],permit,guard)
            original,_=host.step(pixels);reference=host.snapshot()
            historical=np.unpackbits(np.frombuffer((root/'private'/f'qualification_o{order}.bits').read_bytes(),dtype=np.uint8)).reshape(2,512,512).astype(bool)
            if not np.array_equal(mask(original).numpy(),historical):raise ValueError('restored native prediction differs from saved historical next-image prediction')
            host.restore(anchor);again,_=host.step(pixels)
            if not torch.equal(original,again) or not equal(reference,host.snapshot()):raise ValueError('restore/native-step repeatability failed')
            host.restore(anchor);tick=time.monotonic()
            outputs,info,qualification_vectors=one_pair(host,pixels);torch.cuda.synchronize();timings.append(time.monotonic()-tick)
            torch.save(qualification_vectors,root/'private'/f'qualification_vectors_o{order}.pt')
            if not torch.equal(outputs['native'],mask(original)):raise ValueError('instrumented native prediction mismatch')
            assert_native_equivalence(reference,host.snapshot())
            checks.append(dict(order=order,status='PASS',restore_exact=True,historical_mask_exact=True,native_repeat_exact=True,
                               native_state_equivalent=True,matching=info['committed_matching'],
                               gradient_decomposition_relative_error=info['gradient_decomposition_relative_error'],pair_seconds=timings[-1]))
            save(root/'QUALIFICATION.json',dict(status='RUNNING',checks=checks))
            host.close();host=None
            print(f'qualification anchor {order}: PASS',flush=True)
        # A conservative same-run estimate, with fixed 50% runtime headroom.
        estimate=96*max(timings)*1.5+60
        admitted=estimate<c['formal_cap_seconds'] and time.time()+estimate<deadline-30
        qualification_result=dict(status='PASS' if admitted else 'BUDGET_NOT_ADMITTED',checks=checks,
                                  elapsed_seconds=time.time()-start,projected_formal_seconds=estimate,
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),admitted=admitted)
        save(root/'QUALIFICATION.json',qualification_result)
        if not admitted:raise TimeoutError('qualified but fixed matrix cannot fit budget; no formal pairs')
        guard()
        phase_deadline=time.time()+c['formal_cap_seconds']
        status['status']='RUNNING_FIXED_PAIRS';save(root/'RUN_STATE.json',status)
        artifact_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        with (root/'private/pairs.jsonl').open('x') as trace:
            for order,anchor in enumerate(anchors):
                host=Host(state,candidates()[3],17011,anchor['context']['sha256'],device='cuda:0')
                for index,row in enumerate(panel):
                    guard();host.restore(anchor)
                    pixels=image(reader,row,permit,guard);tick=time.monotonic()
                    outputs,info,vectors=one_pair(host,pixels);torch.cuda.synchronize()
                    pair_id=order*48+index
                    with (root/'predictions'/f'{pair_id:03d}.bits').open('xb') as out:
                        for branch in BRANCHES:out.write(np.packbits(outputs[branch].numpy()).tobytes())
                    torch.save(vectors,root/'vectors'/f'{pair_id:03d}.pt')
                    artifact_bytes += (root/'predictions'/f'{pair_id:03d}.bits').stat().st_size+(root/'vectors'/f'{pair_id:03d}.pt').stat().st_size
                    if artifact_bytes > c['output_cap_bytes']-16*1024**2:
                        raise OSError('artifact cap with receipt reserve; no deletion or retry')
                    trace.write(json.dumps(dict(pair=pair_id,order=order,panel_index=index,domain=row['domain'],
                                               seconds=time.monotonic()-tick,diagnostics=info),allow_nan=False)+'\n');trace.flush()
                    status['formal_pairs']+=1
                    if status['formal_pairs']%12==0:
                        save(root/'RUN_STATE.json',status);print(f'fixed pairs: {status["formal_pairs"]}/96',flush=True)
                host.close();host=None
        if status['formal_pairs']!=96:raise ValueError('formal coverage mismatch')
        # Reads already used the existing integrity-checked reader; do not reopen past images.
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input metadata changed after read')
        save(root/'private/ACCESS_AUDIT.json',dict(image_accesses=guard.extra['image_accesses'],label_reads=0,
             source_image_reads=0,opened_paths=sorted(opened),deliberate_rejections=rejected))
        status.update(status='GPU_COMPLETE',peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',error_type=type(e).__name__,reason=str(e),numerical=getattr(e,'diagnostic',None))
        traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        timer.cancel();status['ended']=time.time();status['gpu_worker_seconds']=time.time()-start
        save(root/'RUN_STATE.json',status)


def run_child(c,mode,cap,gpu):
    root=Path(c['output_root']);start=time.time()
    env=dict(os.environ,DIAG_MODE=mode,DIAG_STARTED=str(start),CUDA_VISIBLE_DEVICES=str(c['gpu']['physical_id']) if gpu else '')
    with (root/'logs'/f'{mode}.log').open('x') as log:
        process=subprocess.Popen([sys.executable,os.environ['DIAG_ENTRY']],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1,start_new_session=True)
        identity=dict(pid=process.pid,active=True,started=start,mode=mode)
        save(root/'processes'/f'{mode}.json',identity)
        def terminate():
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        timer=threading.Timer(cap-10 if gpu else cap-5,terminate);timer.daemon=True;timer.start()
        try:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            process.wait()
        finally:timer.cancel()
    elapsed=time.time()-start
    identity.update(active=False,ended=time.time(),exit_code=process.returncode);save(root/'processes'/f'{mode}.json',identity)
    result=dict(mode=mode,status='COMPLETE' if process.returncode==0 else 'FAILED',exit_code=process.returncode,
                started=start,ended=time.time(),gpu_worker_seconds=elapsed if gpu else 0.,cpu_worker_seconds=0. if gpu else elapsed)
    save(root/'attempts'/f'{mode}.json',result)
    return result


def controller(c):
    root=Path(c['output_root'])
    with (root/'started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    if read(root/'CPU_CHECK.json')['status']!='PASS' or c['source_sha']=='PREPARATION':raise ValueError('frozen source and CPU gate required')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus={int(i):(u.strip(),int(m)) for i,u,m in (line.split(',') for line in raw.splitlines())}
    gpu=c['gpu'];uuid,free=gpus[gpu['physical_id']]
    if gpu['physical_id'] not in (4,5,6,7) or uuid!=gpu['uuid'] or free<6144:raise RuntimeError('GPU admission failed; no fallback/retry')
    save(root/'GPU_ADMISSION.json',dict(physical_id=gpu['physical_id'],free_MiB=free,uuid=uuid))
    attempts=[]
    try:
        print('one bounded GPU worker starting; qualification precedes matrix',flush=True)
        attempts.append(run_child(c,'worker',c['gpu_cap_seconds'],True))
        if attempts[-1]['status']=='COMPLETE' and read(root/'RUN_STATE.json')['status']=='GPU_COMPLETE':
            attempts.append(run_child(c,'score',c['cpu_cap_seconds'],False))
    finally:
        save(root/'RESOURCE_LEDGER.json',dict(source_sha=c['source_sha'],attempts=attempts,
             gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),gpu_cap_seconds=c['gpu_cap_seconds'],
             cpu_worker_seconds=sum(a['cpu_worker_seconds'] for a in attempts),
             all_workers_retired=True,no_retry=True,no_replay=True))
    return 0 if len(attempts)==2 and all(a['status']=='COMPLETE' for a in attempts) else 1


def main():
    c=read(os.environ['DIAG_CONFIG']);mode=os.environ['DIAG_MODE']
    if mode=='worker':return worker(c)
    if mode=='controller':
        try:return controller(c)
        except BaseException as e:
            root=Path(c['output_root'])
            save(root/'CONTROLLER_FAILURE.json',dict(status='NOT_EVALUABLE',error_type=type(e).__name__,reason=str(e)))
            if not (root/'RESOURCE_LEDGER.json').exists():
                save(root/'RESOURCE_LEDGER.json',dict(attempts=[],gpu_worker_seconds=0.,gpu_cap_seconds=c['gpu_cap_seconds'],all_workers_retired=True))
            traceback.print_exc();return 1
    if mode=='score':
        from .score import score
        return score(c)
    raise ValueError('unknown diagnostic mode')


if __name__=='__main__':sys.exit(main())
