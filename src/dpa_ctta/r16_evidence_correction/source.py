"""Only registered source labels are available here; target fitting is absent."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..r10_carrier.source import data,bn_record,parameter_stamp
from ..r10_carrier.diagnostic import source_item
from ..r9_current_first.validation import dice
from ..r10_12h_core.run import save,read,sha
from .methods import Engine,ResidualHead,residual,normalize_input,STATIC,prototypes,seeds
from .zero_order import Adapter,COMBINATIONS,SEED
from .structure import hard,topology,containment

INDICES=[16*m+i for m in range(4) for i in (0,4,10,13)]

def metric(z,y):
    h,s=dice(z.float().sigmoid(),y)
    return dict(hard_OD=h[0],hard_OC=h[1],hard_Dice=sum(h)/2,soft_OD=s[0],soft_OC=s[1],soft_Dice=sum(s)/2)

def freeze(seg):return dict(parameters=parameter_stamp(seg),BN=bn_record(seg))

def preflight(c,guard):
    root=Path(c['output_root']);rows=[];static_seconds=[];z_seconds=[];io_seconds=[];source_ids=[]
    torch.manual_seed(SEED)
    heads={k:ResidualHead().cuda() for k in ('D_LOGIT','D_CONTEXT')};heads['D_CONTEXT'].load_state_dict(heads['D_LOGIT'].state_dict())
    stats=dict(mean=torch.zeros(1,269,1,1),std=torch.ones(1,269,1,1))
    with data(c,guard) as src:
        seg=src.segmenter;guard.meter.attach(seg.model);before=freeze(seg)
        if any(m['training'] or m['track_running_stats'] for m in before['BN'].values()):raise ValueError('current-stat eval BN required')
        engine=Engine(seg)
        head_handles=[h.register_forward_pre_hook(lambda *_:guard.extra.__setitem__('head_forwards',guard.extra['head_forwards']+1)) for h in heads.values()]
        try:
            for episode in (0,16,32,48):
                for visit in (0,4,8,12,16,20,24,28):
                    guard();image,label,mode=source_item(src,episode,visit,SEED)
                    guard.extra['image_accesses']+=1
                    if (label[:,1]>label[:,0]).any():raise ValueError('OC must be contained in OD in source annotation')
                    torch.cuda.synchronize();start=time.perf_counter()
                    out,diag,x=engine.outputs(image,heads,stats);torch.cuda.synchronize();static_seconds.append(time.perf_counter()-start)
                    repeat=seg(image)
                    if not torch.equal(repeat,out['C0']):raise ValueError('capture hook changed native logits')
                    for kind in ('D_LOGIT','D_CONTEXT','D_VERIFY'):
                        if not torch.equal(out[kind],out['DS']):raise ValueError('zero residual not identical')
                    adapter=Adapter(seg,'Z_PROBE',1e-3,1e-4,SEED,episode)
                    try:
                        if not torch.equal(seg(image),out['C0']):raise ValueError('zero adapter changed native logits')
                        start=time.perf_counter();z,trace=adapter.step(image);torch.cuda.synchronize();z_seconds.append(time.perf_counter()-start)
                    finally:adapter.close()
                    start=time.perf_counter();p=root/'private/profile-io.npz'
                    np.savez_compressed(p,**{k:np.packbits(hard(v)) for k,v in out.items()});p.read_bytes();p.unlink()
                    p=root/'private/profile-trace.json';save(p,diag);p.read_bytes();p.unlink();io_seconds.append(time.perf_counter()-start)
                    rows.append(dict(episode=episode,visit=visit,mode=mode,metrics={k:metric(v,label) for k,v in out.items()},
                                     prototype=diag['prototype'],zero_residual_exact=True,zero_adapter_exact=True,DS_quarter_hard_exact=True))
            if before!=freeze(seg):raise ValueError('preflight modified backbone or BN')
            # Short mechanical training/profile; discarded heads, no condition selection.
            optim={k:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for k,h in heads.items()};train_seconds=[]
            inp=x['inputs'].repeat(8,1,1,1).cuda();baseline=x['DS'].repeat(8,1,1,1).cuda();y=label.repeat(8,1,1,1).cuda();m=torch.from_numpy(x['editable'])[None].repeat(8,1,1,1).cuda()
            for _ in range(4):
                guard();torch.cuda.synchronize();start=time.perf_counter()
                for kind,head in heads.items():
                    optim[kind].zero_grad(set_to_none=True);r=head(normalize_input(inp,stats,kind));z=torch.where(m,baseline+.25*torch.tanh(F.interpolate(r,(512,512),mode='bilinear',align_corners=False)),baseline)
                    loss=training_loss(z,baseline,y);loss.backward();optim[kind].step()
                torch.cuda.synchronize();train_seconds.append(time.perf_counter()-start)
            p=root/'private/profile-cache.pt';torch.save(dict(inputs=x['inputs'][0],baseline=x['DS'][0],editable=torch.from_numpy(x['editable']),label=label[0]),p)
            start=time.perf_counter()
            for _ in range(8):torch.load(p,weights_only=True)
            cache_batch_read_seconds=time.perf_counter()-start;p.unlink()
        finally:
            engine.close()
            for h in head_handles:h.remove()
        source_ids=list(src.data.folds['fit']);val_ids=list(src.data.folds['val'])
        if set(source_ids)&set(val_ids):raise ValueError('source fit/val overlap')
    result=dict(passed=True,source_images=32,rows=rows,source_fit_groups=len(source_ids),source_val_groups=len(val_ids),
                OD_contains_OC=True,ROI='entire 512x512 grid, as existing scorer; no FOV exclusion',public_postprocessing='none',
                parameters_BN_unchanged=True,profile=dict(static_seconds=static_seconds,z_pair_seconds=z_seconds,IO_seconds=io_seconds,
                train_pair_seconds=train_seconds,cache_batch_read_seconds=cache_batch_read_seconds,
                max_reserved_bytes=torch.cuda.max_memory_reserved()),target_labels_read=0,config_sha256=sha(c))
    save(root/'SOURCE_PRECHECK.json',result);return dict(passed=True,source_images=32)

def calibrate(c,guard):
    root=Path(c['output_root']);results=[]
    with data(c,guard) as src:
        seg=src.segmenter;guard.meter.attach(seg.model);before=freeze(seg)
        for mu,eta in COMBINATIONS:
            for kind in ('Z_CORE','Z_BOUND'):
                for episode in INDICES:
                    adapter=Adapter(seg,kind,mu,eta,SEED,episode);values=[];soft=[]
                    try:
                        for visit in range(32):
                            guard();image,label,mode=source_item(src,episode,visit,SEED);z,trace=adapter.step(image)
                            guard.extra['image_accesses']+=1;guard.extra['zero_order_updates']+=1
                            h,s=dice(z.sigmoid(),label);values.append(h);soft.append(s)
                        a=np.mean(values,axis=0);b=np.mean(soft,axis=0)
                        results.append(dict(mu=mu,eta=eta,condition=kind,episode=episode,mode=mode,hard_OD=float(a[0]),hard_OC=float(a[1]),hard_Dice=float(a.mean()),soft_OD=float(b[0]),soft_OC=float(b[1]),adapter_rms=trace['adapter_rms']))
                        save(root/'SOURCE_Z.json',dict(status='RUNNING',rows=results))
                    finally:adapter.close()
        if before!=freeze(seg):raise ValueError('Z source changed frozen backbone')
    means=[]
    for mu,eta in COMBINATIONS:
        rs=[r['hard_Dice'] for r in results if (r['mu'],r['eta'])==(mu,eta)]
        if len(rs)!=32:raise ValueError('source Z matrix coverage')
        means.append(dict(mu=mu,eta=eta,hard_Dice=float(np.mean(rs))))
    selected=max(means,key=lambda r:(r['hard_Dice'],-r['eta'],-r['mu']))
    result=dict(status='COMPLETE',rows=results,combinations=means,selected=selected,seed=SEED,indices=INDICES,images=4096,full_forwards=8192,
                selection='mode-balanced source hard Dice averaged across CORE and BOUND; exact tie lower eta then mu',target_access=0)
    save(root/'SOURCE_Z.json',result);return result

def source_error(z,b,y):
    a=hard(z);old=hard(b);gt=y.bool().numpy()[0];changed=a!=old
    return dict(correct_to_error=int((changed&(old==gt)&(a!=gt)).sum()),error_to_correct=int((changed&(old!=gt)&(a==gt)).sum()),
                edited=int(changed.sum()),containment=containment(a),topology=topology(a))

def cache_source(c,guard,src):
    root=Path(c['output_root']);cache=root/'source_cache';cache.mkdir(exist_ok=False)
    engine=Engine(src.segmenter);records=[];means=[];variances=[];metrics=[];logical=0
    used=0
    try:
        entries=[('fit',None,None,g) for g in src.data.folds['fit']]+[('val',i,v,None) for i in INDICES for v in range(32)]
        for index,(fold,episode,visit,group) in enumerate(entries):
            guard()
            guard.extra['image_accesses']+=1
            if fold=='fit':row=src.data.get(group,'fit');image,label,mode=row.image,row.label,'identity'
            else:image,label,mode=source_item(src,episode,visit,SEED);group=src.schedule('val',episode,SEED)[1][visit][1]
            with torch.no_grad():out,diag,x=engine.outputs(image)
            payload=dict(inputs=x['inputs'][0],baseline=x['DS'][0],editable=torch.from_numpy(x['editable']),label=label[0],
                         fold=fold,group=group,episode=episode,visit=visit,mode=mode)
            payload['label']=payload['label'].to(torch.uint8)
            path=cache/f'{index:04d}.pt';torch.save(payload,path);used+=path.stat().st_size
            if used>c['origin']['cache_peak_cap_bytes']-512*1024**2:raise RuntimeError('source cache cap; target reserve retained')
            records.append(dict(index=index,file=str(path),fold=fold,episode=episode,visit=visit,mode=mode,group=group))
            if fold=='fit':
                a=x['inputs'].double();means.append(a.mean((0,2,3)));variances.append(a.square().mean((0,2,3)))
            else:
                metrics.extend(dict(condition=k,episode=episode,visit=visit,mode=mode,**metric(z,label),**source_error(z,x['DS'],label)) for k,z in out.items())
                if episode in (0,16,32,48) and visit in (0,4,8,12,16,20,24,28):
                    # Source-only fixed spatial permutation diagnostics reuse this image's detached feature computation.
                    shuffled,_,_=engine.outputs(image,shuffle=True)
                    metrics.append(dict(condition='P_SHUFFLED_SOURCE_DIAGNOSTIC',episode=episode,visit=visit,mode=mode,**metric(shuffled['P'],label)))
            logical+=1
            if index%16==0:save(root/'SOURCE_CACHE_STATE.json',dict(completed=logical,planned=len(entries),bytes=used,metrics=metrics[-32:]))
        mean=torch.stack(means).mean(0);var=torch.stack(variances).mean(0)-mean.square()
        statistics=dict(mean=mean.float().reshape(1,-1,1,1),std=var.clamp_min(1e-12).sqrt().float().reshape(1,-1,1,1))
        torch.save(statistics,root/'private/head-statistics.pt')
        save(root/'private/source-cache-index.json',records);save(root/'SOURCE_STATIC.json',dict(status='COMPLETE',rows=metrics,seed=SEED,indices=INDICES))
        save(root/'SOURCE_CACHE_STATE.json',dict(status='COMPLETE',completed=logical,bytes=used,fit=len(src.data.folds['fit']),val=512))
        return records,statistics
    finally:engine.close()

def training_loss(z,b,y):
    cells=[]
    for c in range(2):
        gt=y[:,c].bool();loss=F.binary_cross_entropy_with_logits(z[:,c],y[:,c],reduction='none');groups=[]
        for a in (gt,~gt):
            if a.any():groups.append(loss[a].mean())
        cells.append(torch.stack(groups).mean())
    p=z.sigmoid();dice_loss=1-((2*(p*y).sum((-2,-1))+1e-6)/(p.sum((-2,-1))+y.sum((-2,-1))+1e-6)).mean()
    correct=(b.sigmoid()>=.5)==y.bool();penalty=(p-b.sigmoid()).abs()[correct].mean() if correct.any() else z.sum()*0
    return torch.stack(cells).mean()+dice_loss+.1*penalty

def train_heads(c,guard):
    root=Path(c['output_root']);torch.manual_seed(SEED);heads={k:ResidualHead().cuda() for k in ('D_LOGIT','D_CONTEXT')}
    heads['D_CONTEXT'].load_state_dict(heads['D_LOGIT'].state_dict());optim={k:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for k,h in heads.items()}
    with data(c,guard) as src:
        guard.meter.attach(src.segmenter.model);before=freeze(src.segmenter);records,stats=cache_source(c,guard,src)
        if before!=freeze(src.segmenter):raise ValueError('D caching changed backbone')
    fit=[r for r in records if r['fold']=='fit'];val=[r for r in records if r['fold']=='val'];selected={};history=[]
    gen=torch.Generator().manual_seed(SEED);schedule=[]
    while len(schedule)<16000:schedule.extend(torch.randperm(len(fit),generator=gen).tolist())
    for step in range(1,2001):
        guard();batch=[torch.load(fit[i]['file'],weights_only=True) for i in schedule[8*(step-1):8*step]]
        inputs=torch.stack([a['inputs'] for a in batch]).cuda();b=torch.stack([a['baseline'] for a in batch]).cuda();y=torch.stack([a['label'] for a in batch]).float().cuda();m=np.stack([a['editable'].numpy() for a in batch])
        for kind,head in heads.items():
            optim[kind].zero_grad(set_to_none=True);r=head(normalize_input(inputs,stats,kind));guard.extra['head_forwards']+=1
            # Batch editable support is explicit; the same exact native logits remain outside it.
            z=torch.where(torch.from_numpy(m).to(b.device),b+.25*torch.tanh(F.interpolate(r,b.shape[-2:],mode='bilinear',align_corners=False)),b)
            loss=training_loss(z,b,y)
            if not torch.isfinite(loss):raise ValueError('nonfinite source head loss')
            loss.backward();optim[kind].step()
        if step%100==0:save(root/'SOURCE_D_PROGRESS.json',dict(updates_per_head=step,max_updates=2000))
        if step in (500,1000,1500,2000):
            values={k:[] for k in heads};detail=[]
            with torch.no_grad():
                for row in val:
                    guard();a=torch.load(row['file'],weights_only=True);inp=a['inputs'][None].cuda();b=a['baseline'][None];label=a['label'][None]
                    for kind,head in heads.items():
                        r=head(normalize_input(inp,stats,kind)).cpu();guard.extra['head_forwards']+=1;z=residual(b,a['editable'].numpy(),r);v=metric(z,label);values[kind].append(v['hard_Dice']);detail.append(dict(condition=kind,step=step,episode=row['episode'],visit=row['visit'],mode=row['mode'],**v))
                for kind in heads:
                    score=float(np.mean(values[kind]));path=root/'private'/f'{kind}.{step}.pt';torch.save(heads[kind].state_dict(),path)
                    if kind not in selected or score>selected[kind]['hard_Dice']:selected[kind]=dict(step=step,hard_Dice=score,file=str(path))
            history.extend(detail);save(root/'SOURCE_D.json',dict(status='RUNNING',rows=history,selected=selected,updates_per_head=step))
    for kind,head in heads.items():head.load_state_dict(torch.load(selected[kind]['file'],weights_only=True));head.eval().requires_grad_(False)
    final=[]
    with data(c,guard) as src:
        guard.meter.attach(src.segmenter.model);engine=Engine(src.segmenter);before=freeze(src.segmenter)
        try:
            for episode in INDICES:
                for visit in range(32):
                    guard();image,label,mode=source_item(src,episode,visit,SEED);out,diag,x=engine.outputs(image,heads,stats);guard.extra['head_forwards']+=2
                    guard.extra['image_accesses']+=1
                    for kind in ('DS','D_LOGIT','D_CONTEXT','D_VERIFY'):
                        final.append(dict(condition=kind,episode=episode,visit=visit,mode=mode,**metric(out[kind],label),**source_error(out[kind],x['DS'],label)))
            if before!=freeze(src.segmenter):raise ValueError('final source evaluation changed backbone')
        finally:engine.close()
    result=dict(status='COMPLETE',rows=history,final_rows=final,selected=selected,updates_per_head=2000,seed=SEED,parameter_counts={k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()},same_batch_schedule=True,fit_groups=len(fit),val_images=len(val),statistics_fold='fit_only')
    save(root/'SOURCE_D.json',result);return result
