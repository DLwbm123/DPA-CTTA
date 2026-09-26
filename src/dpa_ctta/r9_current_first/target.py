"""Sealed float32 probabilities enable exact soft metrics in an independent CPU phase."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..r7_target_screen.runner import TargetReader,image_records
from ..r8_ba.journal import TargetJournal,verify_online_complete,_digest
from ..r8_ba.streams import rows_sha
from .protocol import digest,SPEC_SHA,SPEC
from .storage import write_json
from .metrics import evaluate_probability

PREDICTION_BYTES=2*512*512*4


def lock_sources(receipts,lr_receipt,recipe_receipt):
    if set(receipts)!={j['id'] for j in SPEC['source_tasks']} or any(r.get('schema')!='R9_SOURCE_COMPLETE_V1' or r.get('fit_steps')!=16000 for r in receipts.values()):
        raise ValueError('all 43 completed sources required')
    if not lr_receipt.get('selected_lr') or recipe_receipt.get('schema')!='R9_RECIPE_SELECTION_V1':raise ValueError('all source selections required')
    payload=dict(spec_sha256=SPEC_SHA,sources={k:digest(v) for k,v in receipts.items()},lr=digest(lr_receipt),recipes=digest(recipe_receipt))
    return dict(schema='R9_SOURCE_LOCK_V1',payload=payload,sha256=digest(payload))


def check_lock(lock):
    if lock.get('schema')!='R9_SOURCE_LOCK_V1' or lock.get('sha256')!=digest(lock['payload']) or lock['payload']['spec_sha256']!=SPEC_SHA or len(lock['payload']['sources'])!=43:
        raise ValueError('unsealed source selection')


def online(host,rows,data_root,job_root,job_id,source_lock,guard,failure=None):
    check_lock(source_lock)
    journal=TargetJournal(job_root,host,job_id,rows_sha(rows),prediction_bytes=PREDICTION_BYTES)
    if failure:journal.recover_once(failure)
    else:journal.create()
    reader=TargetReader(data_root,256*1024**2,'image')
    try:
        for i in range(host.visits,len(rows)):
            guard();image=reader.read(image_records((rows[i],))[0]);logits,trace=host.step(image)
            p=logits.sigmoid().detach().cpu().float().contiguous()
            if p.shape!=(1,2,512,512) or not torch.isfinite(p).all():raise ValueError('R9 probability shape/finite')
            trace.update(cycle=i//1951+1,cycle_visit=i%1951+1,
                         mean_probability=p.mean((0,2,3)).tolist(),foreground_fraction=(p>=.5).float().mean((0,2,3)).tolist())
            journal.append(p.numpy().astype('<f4',copy=False).tobytes(),trace);guard()
    finally:reader.after_check()
    receipt=journal.complete(len(rows));write_json(Path(job_root)/'source_lock.json',source_lock);return receipt


def score(rows,data_root,job_root,job_id,context_sha256,source_lock,guard,failure=None):
    check_lock(source_lock);root=Path(job_root)
    if json.loads((root/'source_lock.json').read_text())!=source_lock:raise ValueError('source lock changed')
    seal=verify_online_complete(root,job_id,context_sha256,rows_sha(rows),len(rows),PREDICTION_BYTES)
    out=root/'scalars.private.jsonl';saved=root/'score_checkpoint.json';first=0
    identity=dict(online_sha256=digest(seal),source_lock_sha256=source_lock['sha256'])
    if out.exists():
        failures=root/'failures.jsonl'
        if failures.exists() and json.loads(failures.read_text().splitlines()[-1])['class']!='INFRASTRUCTURE':raise ValueError('noninfrastructure score failure')
        if not failure or failure.get('class')!='INFRASTRUCTURE' or not failure.get('evidence') or (root/'recovery.json').exists():
            raise ValueError('existing partial score needs one equivalent infrastructure recovery')
        state=json.loads(saved.read_text())
        if state['identity']!=identity or _digest(out,state['bytes'])!=state['sha256']:raise ValueError('score prefix seal')
        first=state['visits'];prefix=out.read_bytes()[:state['bytes']].splitlines()
        if len(prefix)!=first or any(json.loads(line)['visit']!=i+1 for i,line in enumerate(prefix)):raise ValueError('score visit prefix')
        write_json(root/'recovery.json',dict(schema='R9_SCORE_RECOVERY_V1',identity=identity,failure=failure,visits=first,retained_tail_bytes=out.stat().st_size-state['bytes']))
        tail=out.read_bytes()[state['bytes']:]
        if tail:(root/'score_uncommitted_tail.private.jsonl').write_bytes(tail)
        with out.open('r+b') as f:f.truncate(state['bytes'])
    else:
        if failure:raise ValueError('no scoring state to recover')
        out.touch();write_json(saved,dict(identity=identity,bytes=0,visits=0,sha256=_digest(out)))
    reader=TargetReader(data_root,256*1024**2,'mask')
    try:
        with (root/'predictions.bits').open('rb') as src,out.open('ab') as dest:
            src.seek(first*PREDICTION_BYTES)
            for i in range(first,len(rows)):
                row=rows[i];guard();raw=src.read(PREDICTION_BYTES)
                if len(raw)!=PREDICTION_BYTES:raise ValueError('sealed probability prefix missing')
                p=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(1,2,512,512)
                result=dict(visit=i+1,cycle=i//1951+1,cycle_visit=i%1951+1,domain=row['domain'],subset=row['subset'],content=row['group_id'],metrics=evaluate_probability(p,reader.read(row)))
                dest.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode())
                if (i+1)%50==0 or i+1==len(rows):
                    dest.flush()
                    import os
                    os.fsync(dest.fileno())
                    write_json(saved,dict(identity=identity,bytes=dest.tell(),visits=i+1,sha256=_digest(out)))
    finally:reader.after_check()
    receipt=dict(schema='R9_SCORE_COMPLETE_V1',visits=len(rows),principal=sum(r['subset']=='remaining_dev' for r in rows),**identity,scalar_sha256=_digest(out))
    write_json(root/'score_complete.json',receipt);return receipt


def retire_probabilities(root):
    """Only R9's temporary sealed probabilities; keep all scalar/trace/recovery evidence."""
    root=Path(root);receipt=json.loads((root/'score_complete.json').read_text())
    if receipt.get('schema')!='R9_SCORE_COMPLETE_V1' or receipt['scalar_sha256']!=_digest(root/'scalars.private.jsonl'):
        raise ValueError('cannot release storage before verified scoring')
    p=root/'predictions.bits'
    if p.exists():
        online=json.loads((root/'online_complete.json').read_text())
        if digest(online)!=receipt['online_sha256']:raise ValueError('sealed source mismatch')
        write_json(root/'probabilities_retired.json',dict(schema='R9_TEMPORARY_PREDICTIONS_RETIRED_V1',retained_prediction_sha256=online['prediction_sha256'],bytes=p.stat().st_size,score_receipt_sha256=digest(receipt)))
        p.unlink()
