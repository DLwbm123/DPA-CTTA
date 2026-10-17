"""Bounded sequential pilot using existing guarded host and retirement machinery."""
import json
import os
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
        estimate = max(durations)*1152*1.5
        peak = torch.cuda.max_memory_reserved()
        if time.time()-start+estimate > c['gpu_cap_seconds']-30:
            raise TimeoutError('full pilot not admitted within budget')
        save(root/'QUALIFICATION.json', dict(status='PASS', checks=checks, estimated_formal_seconds=estimate,
                                             peak_reserved_bytes=peak, online_label_reads=0))
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
                                                   opposite_fraction=detail['diagnostics'].get('pixel_gradient_opposite_fraction',0.)),allow_nan=False)+'\n')
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
    if state['status']!='GPU_COMPLETE' or state['arrivals']!=1152 or state['completed_trajectories']!=6 or retired['active'] or retired['exit_code']!=0:
        raise ValueError('complete successful retirement required')
    panels=[read(root/'scorer'/f'SCREEN_o{o}.json') for o in (0,1)]
    ids=sorted(r['image_sha256'] for r in panels[0]);index={v:i for i,v in enumerate(ids)}
    if len(index)!=192 or sorted(r['image_sha256'] for r in panels[1])!=ids:raise ValueError('paired cohort mismatch')
    domains=np.array([next(r['domain'] for r in panels[0] if r['image_sha256']==v) for v in ids])
    reader=TargetReader(c['target_root'],256*1024**2,'mask');values=np.empty((192,2,3,2));diagnostics=[]
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
                                    mean_pixel_gradient_opposite_fraction=float(np.mean([t['opposite_fraction'] for t in traces]))))
    reader.after_check();np.save(root/'scorer/scores.private.npy',values)
    names=sorted(set(domains));summaries=[];cells=[]
    for name,a,b in [('FCE-NATIVE',2,0),('FCE-HARD',2,1),('HARD-NATIVE',1,0)]:
        delta=values[:,:,a]-values[:,:,b]
        for d in names:
            for o in (0,1):
                for ch,label in enumerate(('OD','OC')):
                    cells.append(dict(contrast=name,domain=d,order=o,channel=label,n_contents=int(sum(domains==d)),delta_pp=float(delta[domains==d,o,ch].mean())))
        orders=np.mean([delta[domains==d].mean((0,2)) for d in names],axis=0)
        summaries.append(dict(contrast=name,macro_delta_pp=float(orders.mean()),order_delta_pp=orders.tolist(),
                              ci95_pp=interval(delta,domains),imageweighted_delta_pp=float(delta.mean()),
                              worst_cell_pp=min(r['delta_pp'] for r in cells if r['contrast']==name)))
    def passed(s,threshold):
        return s['macro_delta_pp']>=threshold and s['ci95_pp'][0]>0 and min(s['order_delta_pp'])>0
    gate=passed(summaries[0],.2) and summaries[0]['worst_cell_pp']>=-2 and summaries[0]['imageweighted_delta_pp']>=0 and passed(summaries[1],.1)
    result=dict(status='COMPLETE',allocation_gate=bool(gate),contrasts=summaries,independent_validation=False,full_TeSLA_reproduction=False)
    out=root/'public';save(out/'RESULTS.json',result);csvout(out/'ALL_CELLS.csv',cells);csvout(out/'MECHANISM.csv',diagnostics)
    csvout(out/'ARM_SCORES.csv',[dict(arm=arm,macro_Dice_percent=float(np.mean([values[domains==d,:,b].mean() for d in names])),imageweighted_Dice_percent=float(values[:,:,b].mean())) for b,arm in enumerate(ARMS)])
    save(out/'QUALIFICATION.json',read(root/'QUALIFICATION.json'))
    save(out/'COMPLETION_AUDIT.json',dict(contents=192,orders=2,arms=3,arrivals=1152,scored_predictions=1152,online_label_reads=0,all_workers_retired=True))
    state.update(status='COMPLETE',scored_predictions=1152,ended=time.time());save(root/'RUN_STATE.json',state)


def main():
    mode=os.environ['DIAG_MODE']
    if mode=='self_check':self_check();return 0
    c=read(os.environ['DIAG_CONFIG']);root=Path(c['output_root'])
    if mode=='worker':return worker(c)
    if mode=='score':score(c);return 0
    if mode!='controller':raise ValueError('unknown mode')
    with (root/'started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU check missing')
    attempts=[]
    try:
        attempts.append(run_child(c,'worker',c['gpu_cap_seconds'],True))
        if attempts[-1]['exit_code']==0:attempts.append(run_child(c,'score',c['cpu_cap_seconds'],False))
    finally:
        ledger=dict(attempts=attempts,gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),
                    cpu_worker_seconds=c['prior_cpu_seconds']+sum(a['cpu_worker_seconds'] for a in attempts),
                    preparation_cpu_seconds=c['prior_cpu_seconds'],all_workers_retired=True)
        save(root/'RESOURCE_LEDGER.json',ledger);save(root/'public/RESOURCE_LEDGER.json',ledger)
    return 0 if len(attempts)==2 and all(a['exit_code']==0 for a in attempts) else 1


if __name__=='__main__':raise SystemExit(main())
