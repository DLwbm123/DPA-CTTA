"""Balanced C/G source trajectories, two equal heads and frozen checkpoint choice."""
import gc,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..r10_carrier.source import data
from ..r7_shared.source import simulate
from ..r10_12h_core.run import save,read,sha
from ..r10_use_write_rl.factory import construct as native_construct
from ..r16_evidence_correction.source import metric,training_loss
from ..r16_evidence_correction.methods import ResidualHead,normalize_input,residual
from .method import construct,SEED,TARGET_SEED,LAMBDA

EPISODES=(0,16,32,48)


def item(src,fold,episode,visit):
    ids,roles,mode=src.schedule(fold,episode,SEED);group=roles[visit][1];r=src.data.get(group,fold)
    image=simulate(r.image,src.banks[fold][ids[visit]],f'R10_QUERY|{fold}|{SEED}|{episode}|{visit}|{group}')
    return image,r.label,mode,group


def model(c,kind,guard,observe=None,head=None,stats=None,weight=LAMBDA):
    h,corr,close=construct(c,kind,TARGET_SEED,dict(sha256=sha(c)),head,stats,weight,observe)
    met=guard.meter.attach(h.native.model)
    def done():met.remove();close()
    return h,corr,done


def preflight(c,guard):
    root=Path(c['output_root']);checks=[];timings=[];captured=[]
    with data(c,guard) as src:
        for kind in ('C_CONTEXT','G_CONTEXT'):
            native,nclose=native_construct(dict(arm='C_CTTA' if kind[0]=='C' else 'G_CTTA',seed=TARGET_SEED,source_job=None,artifact=None),c,c['output_root'],dict(sha256=sha(c)))
            met=guard.meter.attach(native.native.model)
            h,corr,close=model(c,kind,guard,observe=lambda a:captured.append(a),weight=0)
            try:
                for episode in EPISODES:
                    guard();image,label,mode,_=item(src,'val',episode,0);guard.extra['image_accesses']+=2
                    z0,_=native.step(image);start=time.perf_counter();z,t=h.step(image);torch.cuda.synchronize();timings.append(time.perf_counter()-start)
                    error=float((z-z0).abs().max());same=all(torch.equal(a,b) for a,b in zip(native.native.params,h.native.params))
                    if not torch.equal(z,z0) or not same:raise ValueError('lambda zero differs from official native logits/BN')
                    from ..host_diagnostic import close as check_state
                    check_state(native.native.rng,h.native.rng,exact=True)
                    if native.native.base.state_dict()['param_groups']!=h.native.base.state_dict()['param_groups']:raise ValueError('optimizer group parity')
                    for key,state in native.native.base.state_dict()['state'].items():
                        for name,v in state.items():
                            w=h.native.base.state_dict()['state'][key][name]
                            if torch.is_tensor(v) and not torch.equal(v,w):raise ValueError('Adam state parity')
                    checks.append(dict(update=kind[0],source_mode=mode,max_logit_error=error,BN_exact=same,hard_soft=metric(z,label)))
            finally:met.remove();nclose();close();gc.collect()
        a=captured[-1];del captured[:-1]
        torch.manual_seed(SEED);heads={k:ResidualHead().cuda() for k in ('D_LOGIT','D_CONTEXT')};heads['D_CONTEXT'].load_state_dict(heads['D_LOGIT'].state_dict())
        stats=dict(mean=torch.zeros(1,269,1,1),std=torch.ones(1,269,1,1));pair=[]
        opts={k:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for k,h in heads.items()}
        for _ in range(4):
            guard();start=time.perf_counter()
            for k,head in heads.items():
                opts[k].zero_grad(set_to_none=True);r=head(normalize_input(a['inputs'][None].repeat(8,1,1,1).cuda(),stats,k));guard.extra['head_forwards']+=1
                b=a['baseline'][None].repeat(8,1,1,1).cuda();m=a['editable'][None].repeat(8,1,1,1).cuda();y=label.repeat(8,1,1,1).cuda()
                z=torch.where(m,b+.25*torch.tanh(F.interpolate(r,(512,512),mode='bilinear',align_corners=False)),b);loss=training_loss(z,b,y);loss.backward();opts[k].step()
            torch.cuda.synchronize();pair.append(time.perf_counter()-start)
        # Exercise an active deterministic correction during actual C and G updates.
        active=[]
        for kind in ('C_CONTEXT','G_CONTEXT'):
            h,corr,close=model(c,kind,guard,head=heads['D_CONTEXT'].eval().requires_grad_(False),stats=stats)
            try:
                image,_,_,_=item(src,'val',0,0);guard();start=time.perf_counter();z,t=h.step(image);torch.cuda.synchronize();active.append(time.perf_counter()-start);guard.extra['image_accesses']+=1;guard.extra['head_forwards']+=1
                if t['correction']['mean_probability_change']<=0:raise ValueError('nonzero head produced no correction signal')
                snap=h.snapshot();restored,rc,rdone=model(c,kind,guard,head=heads['D_CONTEXT'],stats=stats)
                try:
                    restored.restore(snap);image,_,_,_=item(src,'val',0,1);left,lt=h.step(image);right,rt=restored.step(image);guard.extra['image_accesses']+=2;guard.extra['head_forwards']+=2
                    if not torch.equal(left,right):raise ValueError('integrated snapshot continuation logits mismatch')
                    check_state(h.native.rng,restored.native.rng,exact=True)
                    if not all(torch.equal(a,b) for a,b in zip(h.native.params,restored.native.params)):raise ValueError('integrated continuation BN mismatch')
                finally:rdone()

            finally:close();gc.collect()
    result=dict(passed=True,rows=checks,lambda_zero_official_logits_BN_Adam_exact=True,integrated_snapshot_resume_exact=True,source_accesses=22,target_accesses=0,
          profile=dict(native_seconds=timings,active_seconds=active,train_pair_seconds=pair,max_reserved_bytes=torch.cuda.max_memory_reserved()))
    save(root/'SOURCE_PRECHECK.json',result);return result


def train(c,guard):
    root=Path(c['output_root']);cache=root/'source_cache';cache.mkdir(exist_ok=False);records=[];means=[];variances=[];raw=[];used=0
    with data(c,guard) as src:
        for fold in ('fit','val'):
            for update in ('C','G'):
                for episode in EPISODES:
                    captured=[];h,corr,close=model(c,update+'_CONTEXT',guard,observe=lambda a:captured.append(a),weight=0)
                    try:
                        for visit in range(32):
                            guard();image,label,mode,group=item(src,fold,episode,visit);z,trace=h.step(image);guard.extra['image_accesses']+=1
                            if len(captured)!=1:raise ValueError('source current-view capture count')
                            a=captured.pop();a['label']=label[0].to(torch.uint8);f=cache/f'{len(records):04d}.pt';torch.save(a,f);used+=f.stat().st_size
                            if used>c['origin']['cache_peak_cap_bytes']-1024**3:raise RuntimeError('source cache cap with target reserve')
                            records.append(dict(file=str(f),fold=fold,update=update,episode=episode,visit=visit,mode=mode,group=group))
                            raw.append(dict(fold=fold,update=update,episode=episode,visit=visit,mode=mode,**metric(z,label)))
                            if fold=='fit':
                                v=a['inputs'].double();means.append(v.mean((1,2)));variances.append(v.square().mean((1,2)))
                            if len(records)%16==0:save(root/'SOURCE_CACHE_STATE.json',dict(status='RUNNING',completed=len(records),planned=512,bytes=used))
                    finally:close();gc.collect()
        if {r['group'] for r in records if r['fold']=='fit'}&{r['group'] for r in records if r['fold']=='val'}:raise ValueError('source image fold contamination')
    mean=torch.stack(means).mean(0);var=torch.stack(variances).mean(0)-mean.square();stats=dict(mean=mean.float().reshape(1,-1,1,1),std=var.clamp_min(1e-12).sqrt().float().reshape(1,-1,1,1))
    torch.save(stats,root/'private/head-statistics.pt');save(root/'private/source-cache-index.json',records)
    save(root/'SOURCE_CACHE_STATE.json',dict(status='COMPLETE',completed=512,fit=256,val=256,bytes=used));save(root/'SOURCE_TRAJECTORIES.json',dict(status='COMPLETE',rows=raw,source_fit_groups=len({r['group'] for r in records if r['fold']=='fit'}),source_val_groups=len({r['group'] for r in records if r['fold']=='val'})))
    torch.manual_seed(SEED);heads={k:ResidualHead().cuda() for k in ('D_LOGIT','D_CONTEXT')};heads['D_CONTEXT'].load_state_dict(heads['D_LOGIT'].state_dict());opts={k:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for k,h in heads.items()}
    fit=[r for r in records if r['fold']=='fit'];val=[r for r in records if r['fold']=='val'];gen=torch.Generator().manual_seed(SEED);schedule=[]
    while len(schedule)<16000:schedule.extend(torch.randperm(len(fit),generator=gen).tolist())
    selected={};history=[]
    for step in range(1,2001):
        guard();batch=[torch.load(fit[i]['file'],weights_only=True) for i in schedule[8*(step-1):8*step]]
        inp=torch.stack([a['inputs'] for a in batch]).cuda();b=torch.stack([a['baseline'] for a in batch]).cuda();y=torch.stack([a['label'] for a in batch]).float().cuda();m=torch.stack([a['editable'] for a in batch]).cuda()
        for kind,head in heads.items():
            opts[kind].zero_grad(set_to_none=True);r=head(normalize_input(inp,stats,kind));guard.extra['head_forwards']+=1
            z=torch.where(m,b+.25*torch.tanh(F.interpolate(r,(512,512),mode='bilinear',align_corners=False)),b);loss=training_loss(z,b,y)
            if not torch.isfinite(loss):raise ValueError('nonfinite source loss')
            loss.backward();opts[kind].step()
        if step%100==0:save(root/'SOURCE_D_PROGRESS.json',dict(updates_per_head=step,max_updates=2000))
        if step in (500,1000,1500,2000):
            values={k:[] for k in heads}
            with torch.no_grad():
                for row in val:
                    guard();a=torch.load(row['file'],weights_only=True)
                    for kind,head in heads.items():
                        out=residual(a['baseline'][None],a['editable'].numpy(),head(normalize_input(a['inputs'][None].cuda(),stats,kind)).cpu());guard.extra['head_forwards']+=1
                        v=metric(out,a['label'][None]);values[kind].append(v['hard_Dice']);history.append(dict(condition=kind,step=step,update=row['update'],mode=row['mode'],episode=row['episode'],visit=row['visit'],**v))
                for kind,head in heads.items():
                    score=float(np.mean(values[kind]));f=root/'private'/f'{kind}.{step}.pt';torch.save(head.state_dict(),f)
                    if kind not in selected or score>selected[kind]['hard_Dice']:selected[kind]=dict(step=step,hard_Dice=score,file=str(f))
            save(root/'SOURCE_D.json',dict(status='RUNNING',rows=history,selected=selected,updates_per_head=step))
    result=dict(status='COMPLETE',rows=history,selected=selected,updates_per_head=2000,head_seed=SEED,native_augmentation_seed=TARGET_SEED,parameter_counts={k:9026 for k in heads},same_batch_schedule=True,fit_visits=256,val_visits=256,statistics_fold='fit_only C/G equal',selection='source hard Dice equally weighted C/G and four modes; tie earlier checkpoint')
    save(root/'SOURCE_D.json',result)
    from .method import load_heads
    frozen,stats=load_heads(root,selected);checks=[]
    with data(c,guard) as src:
        for arm in ('C_CONTEXT','G_CONTEXT','G_LOGIT'):
            for episode in EPISODES:
                h,corr,close=model(c,arm,guard,head=frozen['D_LOGIT' if arm=='G_LOGIT' else 'D_CONTEXT'],stats=stats)
                try:
                    for visit in range(8):
                        guard();image,label,mode,_=item(src,'val',episode,visit);z,t=h.step(image);guard.extra['image_accesses']+=1;guard.extra['head_forwards']+=1
                        if not torch.isfinite(z).all():raise ValueError('nonfinite selected-head closed-loop output')
                        checks.append(dict(condition=arm,episode=episode,visit=visit,mode=mode,correction=t['correction'],**metric(z,label)))
                finally:close();gc.collect()
    save(root/'SOURCE_CLOSED_LOOP.json',dict(status='COMPLETE',source_visits=96,rows=checks,performance_gate=False,selection_changed=False))
    return result
