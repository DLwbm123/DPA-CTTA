"""Bounded ENGRAM module transfer; existing guarded host and independent scoring."""
import json
import os
import subprocess
from pathlib import Path
import time
import traceback
import numpy as np
from .method import Host, RetainedHost, ARMS, configuration, self_check
from ..r51_same_state.run import read, save, run_child

WIDTH = 65536


def worker(c):
    import torch
    from ..r19_model_only.runtime import weights, image
    from ..r19_model_only.method import equal
    from ..r7_target_screen.runner import TargetReader, file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    root = Path(c['output_root']); start = time.time(); host = None
    status = dict(status='QUALIFYING', arrivals=0, completed_trajectories=0, started=start)
    save(root/'RUN_STATE.json', status)
    def guard():
        if time.time()-start > c['gpu_cap_seconds']-30:
            raise TimeoutError('registered GPU cap')
    guard.extra = {'image_accesses': 0}
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(2); gpu_policy(c['gpu'])
        permit, opened, rejected = filesystem_guard(c)
        if c['output_cap_bytes'] < 768*WIDTH + 64*1024**2:
            raise ValueError('insufficient fixed output allowance')
        state = weights(c)
        reader = TargetReader(c['target_root'], 256*1024**2, 'image')
        checks = []; durations = []
        for order in (0, 1):
            rows = read(root/'private'/f'SCREEN_o{order}.json')
            if len(rows) != 192: raise ValueError('fixed cohort length')
            x = image(reader, rows[0], permit, guard)
            host = RetainedHost(state, configuration('NATIVE'), c['seed'], 'qualification')
            z_ref, _ = host.step(x); s_ref = host.snapshot(); host.close(); host = None
            for arm in ARMS:
                host = Host(state, configuration(arm), c['seed'], 'qualification')
                initial = host.snapshot(); tick = time.time()
                z, detail = host.step(x); torch.cuda.synchronize(); durations.append(time.time()-tick)
                after = host.snapshot()
                if arm == 'NATIVE' and (not torch.equal(z, z_ref) or not equal(after, s_ref)):
                    raise ValueError('retained host parity failed')
                host.restore(initial); again, _ = host.step(x)
                if not torch.equal(z, again) or not equal(host.snapshot(), after):
                    raise ValueError('repeat state parity failed: '+arm)
                host.check_frozen(True); host.close(); host = None
                checks.append(dict(order=order, arm=arm, repeated_state_exact=True,
                                   retained_parity=True if arm=='NATIVE' else None))
        # Fixed-key recall is exactly zero, including after memory growth.
        const = configuration('ENGRAM'); const['constant_key'] = True
        host = Host(state, const, c['seed'], 'constant-key')
        reference = RetainedHost(state, configuration('NATIVE'), c['seed'], 'constant-key')
        try:
            for _ in range(12):
                z0, _ = reference.step(x); z1, _ = host.step(x)
                hs = host.snapshot(); hs.pop('associative_memory')
                if not torch.equal(z0, z1) or not equal(hs, reference.snapshot()):
                    raise ValueError('constant-key host parity failed')
            if host.memory.U.shape[1] < 1:
                raise ValueError('constant-key qualification did not exercise growth')
            checks.append(dict(constant_key_12_steps_exact=True, grown_rank=host.memory.U.shape[1]))
        finally:
            host.close(); reference.close(); host = None
        estimate = max(durations)*768*1.5
        peak = torch.cuda.max_memory_reserved()
        if time.time()-start+estimate > c['gpu_cap_seconds']-30:
            raise TimeoutError('full pilot not admitted within budget')
        print('real-model qualification PASS; starting fixed streams', flush=True)
        save(root/'QUALIFICATION.json', dict(status='PASS', checks=checks, estimated_formal_seconds=estimate,
                                             peak_reserved_bytes=peak, online_label_reads=0))
        del initial, after, s_ref
        status['status'] = 'RUNNING_FIXED_STREAMS'; save(root/'RUN_STATE.json',status)
        for order in (0,1):
            rows = read(root/'private'/f'SCREEN_o{order}.json')
            for arm in ARMS:
                guard(); host = Host(state, configuration(arm), c['seed'], f'{arm}_o{order}')
                key = f'{arm}_o{order}'
                with (root/'predictions'/f'{key}.bits').open('xb') as bits, (root/'private'/f'{key}.jsonl').open('x') as trace:
                    for i,row in enumerate(rows):
                        guard(); x = image(reader,row,permit,guard); tick=time.time()
                        z,detail = host.step(x); torch.cuda.synchronize()
                        bits.write(np.packbits((z.numpy()[0]>=0)).tobytes())
                        trace.write(json.dumps(dict(visit=i+1,seconds=time.time()-tick,
                                                   update_norm=detail['diagnostics']['BN_update_norm'],
                                                   recall_norm=detail['diagnostics'].get('recall_norm',0.),
                                                   key_shift=detail['diagnostics'].get('key_shift',0.),
                                                   memory_rank=detail['diagnostics'].get('memory_rank',0),
                                                   grew=detail['diagnostics'].get('grew',0)),allow_nan=False)+'\n')
                        trace.flush(); status['arrivals']+=1
                        if (i+1)%16==0:
                            status.update(arm=arm,order=order,visit=i+1);save(root/'RUN_STATE.json',status)
                    if bits.tell()!=192*WIDTH:raise ValueError('prediction size mismatch')
                host.check_frozen(True); host.close();host=None
                save(root/'private'/f'{key}.complete.json',dict(arrivals=192,online_label_reads=0))
                status['completed_trajectories']+=1;save(root/'RUN_STATE.json',status)
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input changed')
        save(root/'private/ACCESS_AUDIT.json',dict(label_reads=0,source_image_reads=0,
             image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),deliberate_rejections=rejected))
        status['status']='GPU_COMPLETE';return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e));traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        status.update(ended=time.time(),gpu_worker_seconds=time.time()-start);save(root/'RUN_STATE.json',status)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import interval,csvout
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer required')
    root=Path(c['output_root']);state=read(root/'RUN_STATE.json');retired=read(root/'processes/worker.json')
    if state['status']!='GPU_COMPLETE' or state['arrivals']!=768 or state['completed_trajectories']!=4 or retired['active'] or retired['exit_code']!=0:
        raise ValueError('complete successful retirement required')
    panels=[read(root/'scorer'/f'SCREEN_o{o}.json') for o in (0,1)]
    ids=sorted(r['image_sha256'] for r in panels[0]);index={v:i for i,v in enumerate(ids)}
    if len(index)!=192 or sorted(r['image_sha256'] for r in panels[1])!=ids:raise ValueError('paired cohort mismatch')
    domains=np.array([next(r['domain'] for r in panels[0] if r['image_sha256']==v) for v in ids])
    reader=TargetReader(c['target_root'],256*1024**2,'mask');values=np.empty((192,2,2,2));diagnostics=[]
    for order,rows in enumerate(panels):
        for b,arm in enumerate(ARMS):
            key=f'{arm}_o{order}';receipt=read(root/'private'/f'{key}.complete.json')
            traces=[json.loads(x) for x in (root/'private'/f'{key}.jsonl').read_text().splitlines()]
            if receipt['arrivals']!=192 or [t['visit'] for t in traces]!=list(range(1,193)):raise ValueError('trajectory coverage')
            with (root/'predictions'/f'{key}.bits').open('rb') as f:
                for row in rows:
                    raw=f.read(WIDTH)
                    if len(raw)!=WIDTH:raise ValueError('short predictions')
                    masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool)
                    values[index[row['image_sha256']],order,b]=[100*m['dice'] for m in hard_metrics(masks,reader.read(row))]
                if f.read(1):raise ValueError('extra predictions')
            diagnostics.append(dict(arm=arm,order=order,seconds_per_image=float(np.mean([t['seconds'] for t in traces])),
                                    mean_update_norm=float(np.mean([t['update_norm'] for t in traces])),
                                    mean_recall_norm=float(np.mean([t['recall_norm'] for t in traces])),
                                    mean_key_shift=float(np.mean([t['key_shift'] for t in traces])),
                                    final_memory_rank=traces[-1]['memory_rank'],
                                    growth_events=sum(t['grew'] for t in traces)))
    reader.after_check();np.save(root/'scorer/scores.private.npy',values)
    names=sorted(set(domains));summaries=[];cells=[]
    for name,a,b in [('ENGRAM-NATIVE',1,0)]:
        delta=values[:,:,a]-values[:,:,b]
        for d in names:
            for o in (0,1):
                for ch,label in enumerate(('OD','OC')):
                    cells.append(dict(contrast=name,domain=d,order=o,channel=label,n_contents=int(sum(domains==d)),delta_pp=float(delta[domains==d,o,ch].mean())))
        orders=np.mean([delta[domains==d].mean((0,2)) for d in names],axis=0)
        summaries.append(dict(contrast=name,macro_delta_pp=float(orders.mean()),order_delta_pp=orders.tolist(),
                              ci95_pp=interval(delta,domains),imageweighted_delta_pp=float(delta.mean()),
                              worst_cell_pp=min(r['delta_pp'] for r in cells if r['contrast']==name)))
    cycles=np.array([next(r['cycle'] for r in panels[0] if r['image_sha256']==v) for v in ids])
    delta=values[:,:,1]-values[:,:,0];cycle_summaries=[]
    for cycle in (0,1):
        chosen=cycles==cycle; ds=domains[chosen]; diff=delta[chosen]
        order_means=np.mean([diff[ds==d].mean((0,2)) for d in names],axis=0)
        cycle_summaries.append(dict(cycle=cycle,macro_delta_pp=float(order_means.mean()),
            order_delta_pp=order_means.tolist(),ci95_pp=interval(diff,ds),
            worst_cell_pp=min(float(diff[ds==d,o,ch].mean()) for d in names for o in (0,1) for ch in (0,1))))
    primary=cycle_summaries[1]
    gate=(primary['macro_delta_pp']>=.2 and primary['ci95_pp'][0]>0 and min(primary['order_delta_pp'])>0
          and primary['worst_cell_pp']>=-2 and summaries[0]['macro_delta_pp']>=0)
    result=dict(status='COMPLETE',allocation_gate=bool(gate),contrasts=summaries,cycles=cycle_summaries,
                primary='second_cycle',independent_validation=False,full_ENGRAM_reproduction=False)
    out=root/'public';save(out/'RESULTS.json',result);csvout(out/'ALL_CELLS.csv',cells);csvout(out/'MECHANISM.csv',diagnostics)
    csvout(out/'ARM_SCORES.csv',[dict(arm=arm,macro_Dice_percent=float(np.mean([values[domains==d,:,b].mean() for d in names])),imageweighted_Dice_percent=float(values[:,:,b].mean())) for b,arm in enumerate(ARMS)])
    save(out/'QUALIFICATION.json',read(root/'QUALIFICATION.json'))
    save(out/'COMPLETION_AUDIT.json',dict(contents=192,orders=2,arms=2,arrivals=768,scored_predictions=768,online_label_reads=0,all_gpu_workers_retired=True,all_workers_retired=False))
    lines=['# ENGRAM module pilot', '', 'Completed fixed development pilot. See PROTOCOL.md for scope.', '',
           'Primary: second-cycle ENGRAM minus NATIVE.', '',
           'Allocation gate: '+str(bool(gate)), '', 'Results:', '```json', json.dumps(result,indent=2), '```', '',
           'Full paper not reproduced: frozen source convolutional descriptor replaces standardized VGG. No RDC.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    state.update(status='COMPLETE',scored_predictions=768,ended=time.time());save(root/'RUN_STATE.json',state)


def main():
    mode=os.environ['DIAG_MODE']
    if mode=='self_check':self_check();return 0
    c=read(os.environ['DIAG_CONFIG']);root=Path(c['output_root'])
    if mode=='import_check':
        import torch
        from ..r19_model_only.runtime import weights, image
        from ..r7_target_screen.runner import TargetReader
        from ..r24_c_context.run import hard_metrics
        from ..r51_same_state.score import interval, csvout
        from ..b1_host import official
        from ..integrations.ctta_suite import load_reference_interfaces
        start=time.time();official();load_reference_interfaces()
        if torch.cuda.is_initialized():raise ValueError('CPU import check initialized CUDA')
        save(root/'IMPORT_CHECK.json',dict(status='PASS',cpu_seconds=time.time()-start,cuda_initialized=False))
        print('ENGRAM_TRANSFER_IMPORT_CHECK_PASS',flush=True)
        return 0
    if mode=='worker':return worker(c)
    if mode=='score':score(c);return 0
    if mode!='controller':raise ValueError('unknown mode')
    with (root/'started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    if read(root/'CPU_CHECK.json')['status']!='PASS' or read(root/'IMPORT_CHECK.json')['status']!='PASS' or c['source_sha']=='PREPARATION':
        raise ValueError('CPU/import checks or frozen source missing')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    devices={int(i):(u.strip(),int(m)) for i,u,m in (line.split(',') for line in raw.splitlines())}
    gpu=c['gpu'];uuid,free=devices[gpu['physical_id']]
    if uuid!=gpu['uuid'] or free<8192:raise RuntimeError('GPU admission failed; no fallback')
    save(root/'GPU_ADMISSION.json',dict(physical_id=gpu['physical_id'],uuid=uuid,free_MiB=free))
    attempts=[]
    try:
        attempts.append(run_child(c,'worker',c['gpu_cap_seconds'],True))
        if attempts[-1]['exit_code']==0:attempts.append(run_child(c,'score',c['cpu_cap_seconds'],False))
    finally:
        ledger=dict(attempts=attempts,gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),
                    cpu_worker_seconds=c['prior_cpu_seconds']+sum(a['cpu_worker_seconds'] for a in attempts),
                    preparation_cpu_seconds=c['prior_cpu_seconds'],all_workers_retired=True)
        save(root/'RESOURCE_LEDGER.json',ledger);save(root/'public/RESOURCE_LEDGER.json',ledger)
        if (root/'public/COMPLETION_AUDIT.json').exists():
            audit=read(root/'public/COMPLETION_AUDIT.json')
            audit['all_workers_retired']=len(attempts)==2 and all(a['exit_code']==0 for a in attempts)
            save(root/'public/COMPLETION_AUDIT.json',audit)
    return 0 if len(attempts)==2 and all(a['exit_code']==0 for a in attempts) else 1


if __name__=='__main__':raise SystemExit(main())
