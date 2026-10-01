"""Paired fixed source episodes, with identity/immutable-backbone qualification."""
import time
from collections import Counter
from pathlib import Path
import torch
from ..r10_12h_core.run import save,sha
from ..r10_carrier.source import data,bn_record,parameter_stamp
from ..r10_carrier.diagnostic import source_item
from ..r9_current_first.validation import dice
from .run import SOURCE,lock
from .view import construct

@torch.no_grad()
def preflight(c,guard):
    root=Path(c['output_root']);names=SOURCE;hosts={};handles=[];timings={k:0. for k in names};rows=[];start=time.time()
    try:
        for name in names:
            hosts[name]=construct(name,c,lock(c,name));handles.append(guard.meter.attach(hosts[name][0].segmenter.model))
        before={k:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for k,(h,_) in hosts.items()}
        if any(v!=before['C0'] for v in before.values()):raise ValueError('different backbones')
        if any(m['training'] or m['track_running_stats'] or m['running_mean'] is not None or m['running_var'] is not None for m in before['C0']['BN'].values()):raise ValueError('current-statistics BN mismatch')
        with data(c,guard) as src:
            for index in (0,16,32,48):
                for visit in (0,4,8,12,16,20,24,28):
                    guard();image,_,mode=source_item(src,index,visit);out={}
                    for name,(h,_) in hosts.items():
                        torch.cuda.synchronize();t=time.perf_counter();out[name]=h.step(image)[0];torch.cuda.synchronize();timings[name]+=time.perf_counter()-t
                    raw=hosts['C0'][0].segmenter(image)
                    exact=torch.equal(raw,out['C0']) and torch.equal(raw.sigmoid(),out['C0'].sigmoid())
                    expected=torch.logit((.75*out['C0'].sigmoid()+.25*out['H_ONLY'].sigmoid()).clamp(1e-6,1-1e-6))
                    if not torch.equal(expected,out['CV_H025']):raise ValueError('fixed quarter-weight equation mismatch')
                    rows.append(dict(episode=index,visit=visit,mode=mode,identity_exact=exact,quarter_equation_exact=True))
                    if not exact:raise ValueError('single view differs from unchanged C0')
                    for h,_ in hosts.values():h.check_frozen(True)
        after={k:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for k,(h,_) in hosts.items()}
        passed=before==after and all(r['identity_exact'] for r in rows)
        result=dict(schema='R12_SOURCE_PARITY_V1',passed=passed,visits=32,rows=rows,modes=dict(Counter(r['mode'] for r in rows)),backbone_and_BN=before,parameters_and_buffers_unchanged=before==after,timings=timings,wall_seconds=time.time()-start,config_sha256=sha(c))
        save(root/'ZERO_PARITY.json',result)
        if not passed:raise ValueError('identity/backbone parity failed')
        return dict(passed=True,visits=32)
    finally:
        for h in handles:h.remove()
        for _,close in hosts.values():close()

@torch.no_grad()
def comparison(c,guard):
    root=Path(c['output_root']);hosts={};handles=[];rows=[];modes=Counter();start=time.time()
    try:
        for name in SOURCE:
            hosts[name]=construct(name,c,lock(c,name));handles.append(guard.meter.attach(hosts[name][0].segmenter.model))
        with data(c,guard) as src:
            for index in c['val_indices']:
                collected={k:[] for k in SOURCE};soft={k:[] for k in SOURCE}
                for visit in range(32):
                    guard();image,label,mode=source_item(src,index,visit)
                    for k,(host,_) in hosts.items():
                        logits,_=host.step(image);hd,sd=dice(logits.sigmoid(),label);collected[k].append(hd);soft[k].append(sd)
                modes[mode]+=1
                for k,(host,_) in hosts.items():
                    host.check_frozen(True);hd=torch.tensor(collected[k],dtype=torch.float64).mean(0);sd=torch.tensor(soft[k],dtype=torch.float64).mean(0)
                    rows.append(dict(condition=k,episode=index,mode=mode,visits=32,hard_OD=float(hd[0]),hard_OC=float(hd[1]),hard_Dice=float(hd.mean()),soft_OD=float(sd[0]),soft_OC=float(sd[1]),soft_Dice=float(sd.mean())))
                save(root/'SOURCE_COMPARISON.json',dict(status='RUNNING',rows=rows,completed_episodes=sum(modes.values()),planned_episodes=16))
        if len(rows)!=64 or sorted(modes.values())!=[4]*4:raise ValueError('paired source coverage')
        reference={(r['condition'],r['episode']):r for r in c['source_reference']};differences=[]
        for row in rows:
            if row['condition'] in ('C0','CV_H2'):
                ref=reference[row['condition'],row['episode']]
                if row['mode']!=ref['mode']:raise ValueError('source mode reference changed')
                differences.extend(abs(row[k]-ref[k]) for k in ('hard_OD','hard_OC','hard_Dice','soft_OD','soft_OC','soft_Dice'))
        proof=dict(passed=max(differences)<=1e-12,max_metric_delta=max(differences),paired_control_episodes=32,tolerance=1e-12,selection=False)
        save(root/'SOURCE_REFERENCE_PARITY.json',proof)
        if not proof['passed']:raise ValueError('fresh C0/half source replay differs from prior fixed source results')
        save(root/'SOURCE_COMPARISON.json',dict(status='COMPLETE',rows=rows,visits=2048,episodes=16,modes=dict(modes),seed=20260924,indices=c['val_indices'],wall_seconds=time.time()-start))
        return dict(visits=2048,episodes=16)
    finally:
        for h in handles:h.remove()
        for _,close in hosts.values():close()
