"""Sparse paired block decisions; KEEP shares the verified native trajectory."""
import json
import os
import sys
import time
from pathlib import Path
import torch
from .method import BLOCK, PIVOTS, rollback
from ..r30_state_history.method import StateHost
from ..r30_state_history.online import pack, reference_mask, inventory
from ..r32_history_origin.online import coupled_step
from ..r19_model_only.method import equal
from ..r19_model_only.runtime import weights, image
from ..r7_target_screen.runner import TargetReader
from ..r10_12h_core.run import read, save


def access_guard(c):
    root = Path(c['output_root']).resolve(); checkpoint = Path(c['checkpoint_path']).resolve()
    reference = Path(c['snapshot_input_root']).resolve()
    libraries = [Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    current = {'image': None}
    def audit(event,args):
        if event != 'open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        if any(part in ('scorer','scores') for part in p.parts):
            raise PermissionError('online label/score barrier')
        if p.suffix.lower() in ('.png','.jpg','.jpeg','.pt','.pth','.npy','.npz','.bits'):
            permitted_reference = p.suffix=='.bits' and p.is_relative_to(reference)
            if p not in (checkpoint,current['image']) and not permitted_reference and not p.is_relative_to(root/'target') and not any(p.is_relative_to(x) for x in libraries):
                raise PermissionError('unregistered model/data access')
    sys.addaudithook(audit)
    try: (root/'scorer/denial_probe.json').read_bytes()
    except PermissionError: pass
    else: raise AssertionError('label guard did not reject')
    return current


def stream(c,job,guard,permit,profiling=False):
    root=Path(c['output_root']);rows=read(root/f'private/ONLINE_o{job["order"]}.json')
    if len(rows)!=1951: raise ValueError('registered stream changed')
    pivots=PIVOTS[:1] if profiling else PIVOTS
    lengths={t:min(8 if profiling else 256,len(rows)-t) for t in pivots}
    end=72 if profiling else len(rows)
    dest=root/'target'/job['id'];dest.mkdir(exist_ok=False)
    tick=time.perf_counter();h=StateHost(weights(c),'C_CONT',job['seed'],job['id'])
    guard.meter.attach(h.native.model);init=time.perf_counter()-tick
    save(dest/'inventory.json',inventory(h))
    reader=TargetReader(c['target_root'],256*1024**2,'image')
    snapshots={};schedule={};replay_times=[];future_times=[];windows=[]
    needed=set(pivots)|{t-BLOCK for t in pivots}
    reference=Path(c['snapshot_input_root'])/f'B_C_CONT_s{job["seed"]}_o{job["order"]}'/'predictions.bits'
    try:
        with (dest/'native.bits').open('xb') as bits, (dest/'native_traces.jsonl').open('x') as trace:
            for visit,row in enumerate(rows[:end],1):
                guard();tick=time.perf_counter();z,details,rng=coupled_step(h,image(reader,row,permit,guard))
                raw=pack(z)
                if raw!=reference_mask(reference,visit): raise ValueError('native replay mismatch against R30')
                bits.write(raw);schedule[visit]=rng
                if visit in needed: snapshots[visit]=h.snapshot()
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick;replay_times.append(elapsed)
                trace.write(json.dumps(dict(visit=visit,seconds=elapsed,consistency_loss=details['native'][0]['diagnostics']['cal_consis_loss'],update_norm=details['diagnostics']['BN_update_norm']))+'\n')
                guard.extra['replay_arrivals']=visit
            bits.flush();trace.flush()
        for t in pivots:
            before,after=snapshots[t-BLOCK],snapshots[t]
            state=rollback(before,after);h.restore(state)
            if not equal(state,h.snapshot()): raise ValueError('rollback restore mismatch')
            with (dest/f'rollback_{t}.bits').open('xb') as bits:
                for offset in range(1,lengths[t]+1):
                    visit=t+offset;guard();tick=time.perf_counter()
                    z,_,_=coupled_step(h,image(reader,rows[visit-1],permit,guard),schedule[visit])
                    bits.write(pack(z));bits.flush();torch.cuda.synchronize()
                    future_times.append(time.perf_counter()-tick);guard.extra['rollback_future_arrivals']+=1
            if h.visits!=t+lengths[t] or h.native.steps!=t-BLOCK+lengths[t]:
                raise ValueError('rollback local/global clock mismatch')
            if not equal(state,rollback(before,after)): raise ValueError('donor snapshot mutated')
            windows.append(dict(pivot=t,block=BLOCK,available_future=lengths[t],retained_start_step=before['steps'],global_start=t,shared_block_predictions=True))
            guard.extra['windows_complete']+=1
        h.check_frozen(True)
        updates=end+sum(lengths.values());expected=dict(optimizer_steps=updates,backward_calls=updates,model_forwards=8*updates,vjp_calls=0)
        if any(guard.meter.cost[k]!=v for k,v in expected.items()): raise ValueError('physical operation accounting')
        result=dict(status='COMPLETE',initialization_seconds=init,replay_arrivals=end,replay_seconds=replay_times,reference_checks=end,
                    windows=windows,rollback_future_arrivals=sum(lengths.values()),future_seconds=future_times,expected_operations=expected,
                    label_reads=0,peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(dest/'COMPLETE.json',result);return result
    finally: h.close()


def profile(c,job,guard,permit):
    return stream(c,dict(job,seed=c['seeds'][0],order=0),guard,permit,profiling=True)
