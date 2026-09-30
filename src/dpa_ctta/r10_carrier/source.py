"""Fixed source parity and paired controls; exact historical R10 val simulator."""
import time
from contextlib import contextmanager
from pathlib import Path
from collections import Counter
import torch
from ..r10_12h_core.run import save,sha
from ..r10_use_write_rl.assets import open_source
from ..r10_use_write_rl.source import Source
from ..r9_current_first.validation import dice
from ..r7_shared.context import tensor_digest
from .diagnostic import construct,resolved,source_item,equal_state
from .run import SOURCE,lock

@contextmanager
def data(c,guard):
    with open_source(c['bindings'],c['gpu_assignments'][0],guard) as (ds,seg,oracles,controller):
        yield Source(ds,oracles,seg,controller,sha(c['bindings']))

def bn_record(seg):
    return {n:dict(training=m.training,track_running_stats=m.track_running_stats,
        running_mean=None if m.running_mean is None else m.running_mean.tolist(),
        running_var=None if m.running_var is None else m.running_var.tolist(),
        num_batches_tracked=None if m.num_batches_tracked is None else int(m.num_batches_tracked),
        affine=m.affine,eps=m.eps,dtype=str(m.weight.dtype))
        for n,m in seg.model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)}

def parameter_stamp(seg):return tensor_digest(list(seg.model.state_dict().items()))

@torch.no_grad()
def preflight(c,guard):
    from ..r10_use_write_rl.factory import construct as native
    root=Path(c['output_root']);names=(*SOURCE,'B_ZERO');hosts={};handles=[];rows=[];timings={k:0. for k in (*names,'NATIVE_B')};start=time.time()
    try:
        for name in names:
            hosts[name]=construct(name,c,lock(c,name));handles.append(guard.meter.attach(hosts[name][0].segmenter.model))
        hosts['NATIVE_B']=native(resolved('B_FULL_1'),c,c['output_root'],lock(c,'B_FULL_1'));handles.append(guard.meter.attach(hosts['NATIVE_B'][0].segmenter.model))
        before={k:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for k,(h,_) in hosts.items()}
        if any(x['BN']!=before['C0']['BN'] or x['parameters']!=before['C0']['parameters'] for x in before.values()):raise ValueError('backbone/BN path mismatch')
        if any(m['training'] or m['track_running_stats'] or m['running_mean'] is not None or m['running_var'] is not None for m in before['C0']['BN'].values()):raise ValueError('not current-statistics eval BN')
        with data(c,guard) as src:
            indices=[16*m for m in range(4)]
            for index in indices:
                for visit in (0,4,8,12,16,20,24,28):
                    guard();image,_,mode=source_item(src,index,visit);outputs={}
                    for name,(h,_) in hosts.items():
                        t=time.perf_counter();outputs[name]=h.step(image)[0];timings[name]+=time.perf_counter()-t
                    repeat=hosts['C0'][0].segmenter(image)
                    zero_obs=hosts['B_ZERO'][0].segmenter(image,observe=True)[0]
                    c0=outputs['C0'];zero=outputs['B_ZERO']
                    full_state=hosts['B_FULL_1'][0].state
                    if not all(equal_state(full_state,hosts[k][0].state) for k in ('B_ZERO','B_FULL_READOUT_025','NATIVE_B')):raise ValueError('READOUT_ONLY changed FULL recurrence')
                    if not equal_state(hosts['B_RESET_1'][0].state,hosts['B_RESET_READOUT_025'][0].state):raise ValueError('readout changed RESET recurrence')
                    if not torch.equal(outputs['B_FULL_1'],outputs['NATIVE_B']):raise ValueError('alpha=1 changed native B outputs')
                    rows.append(dict(episode=index,visit=visit,mode=mode,repeat_logits_max=float((c0-repeat).abs().max()),repeat_probability_max=float((c0.sigmoid()-repeat.sigmoid()).abs().max()),zero_logits_max=float((zero-c0).abs().max()),zero_logits_mean=float((zero-c0).abs().mean()),zero_probability_max=float((zero.sigmoid()-c0.sigmoid()).abs().max()),zero_mask_mismatch=int(((zero>=0)!=(c0>=0)).sum()),observation_vs_prediction_logits_max=float((zero_obs-zero).abs().max())))
                    if any(h.segmenter.v is not None or h.segmenter.capture or h.segmenter.cache for h,_ in hosts.values()):raise ValueError('observation leaked persistent segmenter cache')
        after={k:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for k,(h,_) in hosts.items()}
        unchanged=before==after;tol_logit=max(r['repeat_logits_max'] for r in rows);tol_prob=max(r['repeat_probability_max'] for r in rows)
        passed=unchanged and all(r['zero_logits_max']<=tol_logit and r['zero_probability_max']<=tol_prob and r['observation_vs_prediction_logits_max']<=tol_logit and r['zero_mask_mismatch']==0 for r in rows)
        result=dict(schema='CARRIER_ZERO_PARITY_V1',passed=passed,visits=32,modes=dict(Counter(r['mode'] for r in rows)),rows=rows,criteria=dict(source='same-path repeat numerical error frozen before targets',logits_max=tol_logit,probability_max=tol_prob,hard_mask_mismatch=0),backbone_and_BN=before,parameters_and_buffers_unchanged=unchanged,alpha_one_native_B_exact=True,FULL_state_same_across_alpha=True,RESET_state_same_across_alpha=True,observation_no_persistent_side_effect=True,timings=timings,wall_seconds=time.time()-start,target_direct_pixel_parity='Pending new B_ZERO probability seal compared with compatible old C0 seal; old probabilities were retired.',parent=c['parent']['artifact'],config_sha256=sha(c))
        save(root/'ZERO_PARITY.json',result)
        if not passed:raise ValueError('zero-modulation parity failed; no further carrier interpretation')
        return dict(passed=True,visits=32,wall_seconds=result['wall_seconds'])
    finally:
        for h in handles:h.remove()
        for _,close in hosts.values():close()

@torch.no_grad()
def comparison(c,guard):
    root=Path(c['output_root']);rows=[];diagnostics=[];start=time.time();hosts={};handles=[]
    try:
        for k in SOURCE:hosts[k]=construct(k,c,lock(c,k));handles.append(guard.meter.attach(hosts[k][0].segmenter.model))
        with data(c,guard) as src:
            modecounts=Counter()
            for index in c['val_indices']:
                for h,_ in hosts.values():h.state=None if h.method is None else h.method.initial();h.visits=0
                collected={k:[] for k in SOURCE};soft={k:[] for k in SOURCE}
                for visit in range(32):
                    guard();image,label,mode=source_item(src,index,visit)
                    for k,(host,_) in hosts.items():
                        logits,trace=host.step(image);hard,sd=dice(logits.sigmoid(),label);collected[k].append(hard);soft[k].append(sd)
                        diagnostics.append(dict(condition=k,episode=index,visit=visit,mode=mode,**{k:v for k,v in trace.items() if k not in ('counts','B_state_sha256')}))
                    if not equal_state(hosts['B_FULL_1'][0].state,hosts['B_FULL_READOUT_025'][0].state):raise ValueError('source FULL readout changed native state')
                    if not equal_state(hosts['B_RESET_1'][0].state,hosts['B_RESET_READOUT_025'][0].state):raise ValueError('source RESET readout changed native state')
                modecounts[mode]+=1
                for k in SOURCE:
                    h=torch.tensor(collected[k],dtype=torch.float64).mean(0);s=torch.tensor(soft[k],dtype=torch.float64).mean(0)
                    rows.append(dict(condition=k,episode=index,mode=mode,visits=32,hard_OD=float(h[0]),hard_OC=float(h[1]),hard_Dice=float(h.mean()),soft_OD=float(s[0]),soft_OC=float(s[1]),soft_Dice=float(s.mean())))
                save(root/'SOURCE_COMPARISON.json',dict(status='RUNNING',rows=rows,completed_episodes=len(modecounts) and sum(modecounts.values()),planned_episodes=16))
            if len(rows)!=80 or len(modecounts)!=4 or any(n!=4 for n in modecounts.values()):raise ValueError('balanced paired source validation')
        result=dict(status='COMPLETE',rows=rows,diagnostics=diagnostics,visits=2560,episodes=16,modes=dict(modecounts),seed=20260924,indices=c['val_indices'],B_ZERO_alias='C0 based on separately verified source parity; no sixth validation arm',readout_state_equal_every_visit=True,wall_seconds=time.time()-start)
        save(root/'SOURCE_COMPARISON.json',result);return dict(visits=2560,episodes=16)
    finally:
        for h in handles:h.remove()
        for _,close in hosts.values():close()
