"""Independent CPU consumer. Masks are never treated as real probabilities."""
import json
from pathlib import Path
import numpy as np
import torch
from ..r10_12h_core.run import read,save,sha
from ..r8_ba.journal import verify_online_complete,_digest
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader
from ..p1_analysis import evaluate
from .methods import STATIC
from .structure import topology,containment

WIDTH=65536

def metrics(mask,label,baseline=None):
    values=evaluate(torch.from_numpy(mask.copy()).float()[None],label,'fundus')
    for c,m in enumerate(values):
        m.update(TP=m['intersection'],FP=m['pred_pixels']-m['intersection'],FN=m['gt_pixels']-m['intersection'],soft_dice=None,Brier=None)
        if baseline is not None:
            gt=label.bool().numpy()[0,c];old=baseline[c];new=mask[c];changed=old!=new
            m.update(correct_to_error=int((changed&(old==gt)&(new!=gt)).sum()),error_to_correct=int((changed&(old!=gt)&(new==gt)).sum()),edited_pixels=int(changed.sum()))
    return values

def score(c,guard):
    root=Path(c['output_root']);state=read(root/'RUN_STATE.json');lock=read(root/'EXPERIMENT_LOCK.json');p=lock['payload'];manifests=[read(m['path']) for m in p['manifests']]
    if state['status']!='TARGET_MATRIX_TERMINAL' or any(v in ('RUNNING','NOT_RUN') for v in state['jobs'].values()):raise ValueError('all target jobs must be terminal before CPU labels')
    if lock['sha256']!=sha(p):raise ValueError('experiment lock changed')
    score_root=root/'score';score_root.mkdir(exist_ok=False);scalars=[];completed=[];failed=[];statics={};ds_masks={}
    baseline={}
    for r in p['reused']:
        values=read(r['values_path'])
        if sha(values)!=r['values_sha256']:raise ValueError('reference scalar identity changed')
        baseline[r['condition'],r['order']]=values
    reader=TargetReader(c['bindings']['target_root'],256*1024**2,'mask')
    try:
        # No model, optimizer or CUDA state is constructed in this independent process.
        for jid,status in state['jobs'].items():
            if jid in ('preflight','source_Z','source_D') or jid=='score':continue
            if status!='COMPLETE':failed.append(dict(job=jid,status=status));continue
            path=root/'target'/jid;on=read(path/'online_complete.json');static=jid=='STATIC_o0';order=0 if static else int(jid.split('_o')[1][0]);seed=None if static else int(jid.split('_s')[1]);arm=None if static else jid.split('_o')[0]
            rows=manifests[order];width=WIDTH*len(STATIC) if static else WIDTH
            verify_online_complete(path,jid,on['identity']['context_sha256'],rows_sha(rows),len(rows),width)
            output=score_root/(jid+'.private.jsonl')
            with (path/'predictions.bits').open('rb') as predictions,output.open('x') as dest:
                for i,m in enumerate(rows):
                    guard();raw=predictions.read(width)
                    if len(raw)!=width:raise ValueError('sealed prediction truncated')
                    masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(len(STATIC) if static else 1,2,512,512).astype(bool)
                    label=reader.read(m)
                    if (label[:,1]>label[:,0]).any():raise ValueError('target annotation containment protocol changed')
                    if static:
                        ds=masks[STATIC.index('DS')];ds_masks[m['group_id']]=ds
                        if not np.array_equal(ds,masks[STATIC.index('H025')]):raise ValueError('DS/quarter hard equivalence failed')
                    else:ds=ds_masks.get(m['group_id'])
                    for k,mask in zip(STATIC if static else (arm,),masks):
                        result=dict(condition=k,order=order,seed=seed,visit=i+1,content=m['group_id'],domain=m['domain'],subset=m['subset'],
                                    metrics=metrics(mask,label,ds),containment_violations=containment(mask),fragments_holes=topology(mask),origin='NEW_SHARED_STATIC' if static else 'NEW_STATEFUL',soft_metrics='NA_MASK_ONLY')
                        if static and k in ('C0','H025'):
                            ref=baseline[k,0][i]
                            for a,b in zip(result['metrics'],ref['metrics']):
                                if a['channel']!=b['channel'] or any(a[x]!=b[x] for x in ('intersection','pred_pixels','gt_pixels','total_pixels')):raise ValueError('new shared path mismatches frozen historical '+k)
                            result['origin']='HISTORICAL_MATCHED_WITH_NEW_SHARED_PIXEL_COUNT_PARITY'
                        scalars.append(result);dest.write(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
                        if static:statics[k,m['group_id']]=result
                if predictions.read(1):raise ValueError('unregistered prediction tail')
            receipt=dict(job=jid,visits=len(rows),principal=sum(m['subset']=='remaining_dev' for m in rows),online_sha256=sha(on),scalar_sha256=_digest(output),independent_CPU=True,real_probability_metrics=False)
            save(score_root/(jid+'.complete.json'),receipt);completed.append(receipt)
        # Static order 1 is a content-verified reordering, never a second model run.
        if statics:
            for i,m in enumerate(manifests[1]):
                for k in STATIC:
                    r=statics[k,m['group_id']]
                    if (r['domain'],r['subset'])!=(m['domain'],m['subset']):raise ValueError('static order role mismatch')
                    scalars.append(dict(r,order=1,visit=i+1,origin='REORDERED_REUSE' if k not in ('C0','H025','DS') else 'REORDERED_REUSE_EQUIVALENT_BY_CONSTRUCTION' if k=='DS' else 'REORDERED_REUSE_WITH_HISTORICAL_PARITY'))
        for o in (0,1):
            for i,m in enumerate(manifests[o]):
                ref=baseline['G',o][i]
                scalars.append(dict(ref,condition='G',order=o,seed=20260907,origin='HISTORICAL_MATCHED_INTACT_STATEFUL',soft_metrics='NA_MASK_ONLY'))
        out=score_root/'all-scalars.private.jsonl'
        with out.open('x') as f:
            for r in scalars:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    finally:reader.after_check()
    result=dict(status='COMPLETE',completed=completed,failed=failed,scalar_rows=len(scalars),scalar_sha256=_digest(out),
                source_lock_sha256=lock['sha256'],mask_reads=dict(reader.counts),soft_metrics='NA; not measured from probabilities',embargo='all registered prediction jobs terminal before label scoring')
    save(root/'SCORER_RECEIPT.json',result);return result
