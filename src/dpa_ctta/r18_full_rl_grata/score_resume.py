"""Resume only missing CPU scalar rows after the scorer's short timeout."""
import json,os,signal,time,traceback
from pathlib import Path
import numpy as np
from . import quick,score as original_score
from ..r10_12h_core.run import read,save,sha
from ..r8_ba.journal import _digest,verify_online_complete
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader
from ..r16_evidence_correction.score import metrics,WIDTH
from ..r16_evidence_correction.structure import topology,containment
from ..r9_current_first.storage import lease


def resume(c,deadline):
    root=Path(c['output_root']);state=read(root/'RUN_STATE.json');lock=read(root/'EXPERIMENT_LOCK.json');p=lock['payload'];dest=root/'score';plan=read(root/'CPU_SCORE_RESUME_PLAN.json')
    if any(state['jobs'].get(f'{a}_o{o}')!='COMPLETE' for a in quick.ARMS for o in (0,1)):raise ValueError('all six online trajectories must be complete')
    if lock['sha256']!=sha(p) or p['config_sha256']!=sha(c):raise ValueError('frozen lock identity')
    completed=[];scalars=[];new_reads=0
    for ref in p['reused']:
        values=read(ref['values_path'])
        if sha(values)!=ref['values_sha256']:raise ValueError('reference mutation')
        scalars.extend(dict(v,origin='SEALED_R17_REFERENCE',soft_metrics='NA_MASK_ONLY') for v in values)
    reader=TargetReader(c['bindings']['target_root'],256*1024**2,'mask')
    try:
        for arm in quick.ARMS:
            for order in (0,1):
                jid=f'{arm}_o{order}';path=root/'target'/jid;on=read(path/'online_complete.json');rows=read(p['manifests'][order]['path']);output=dest/(jid+'.private.jsonl');done=dest/(jid+'.complete.json')
                if _digest(output)!=plan['prefixes'][jid]['sha256']:raise ValueError('preserved scalar prefix mutation')
                prefix=[json.loads(line) for line in output.open()]
                if len(prefix)!=plan['prefixes'][jid]['rows'] or any((v['condition'],v['order'],v['visit'],v['content'],v['domain'],v['subset'])!=(arm,order,i+1,m['group_id'],m['domain'],m['subset']) for i,(v,m) in enumerate(zip(prefix,rows))):raise ValueError('scalar prefix identity')
                scalars.extend(prefix)
                if done.exists():
                    receipt=read(done)
                    if len(prefix)!=1951 or receipt['scalar_sha256']!=_digest(output) or receipt['online_sha256']!=sha(on):raise ValueError('completed score receipt identity')
                    completed.append(receipt);continue
                verify_online_complete(path,jid,on['identity']['context_sha256'],rows_sha(rows),len(rows),WIDTH)
                with (path/'predictions.bits').open('rb') as f,output.open('a') as out:
                    f.seek(len(prefix)*WIDTH)
                    for i in range(len(prefix),len(rows)):
                        if time.time()>deadline-15:raise TimeoutError('CPU close reserve')
                        m=rows[i];raw=f.read(WIDTH)
                        if len(raw)!=WIDTH:raise ValueError('prediction truncation')
                        mask=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool);label=reader.read(m)
                        if (label[:,1]>label[:,0]).any():raise ValueError('label containment')
                        row=dict(condition=arm,order=order,seed=quick.SEED,visit=i+1,content=m['group_id'],domain=m['domain'],subset=m['subset'],metrics=metrics(mask,label),containment_violations=containment(mask),fragments_holes=topology(mask),origin='NEW_STATEFUL_INTEGRATED',soft_metrics='NA_MASK_ONLY')
                        out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');scalars.append(row);new_reads+=1
                        if new_reads%64==0:out.flush();save(root/'CPU_SCORE_RESUME_PROGRESS.json',dict(job=jid,new_rows=new_reads,completed_rows=i+1,total=len(rows)))
                    if f.read(1):raise ValueError('prediction tail')
                receipt=dict(job=jid,visits=len(rows),principal=sum(m['subset']=='remaining_dev' for m in rows),online_sha256=sha(on),scalar_sha256=_digest(output),independent_CPU=True,reused_prefix_rows=len(prefix),new_rows=len(rows)-len(prefix))
                save(done,receipt);completed.append(receipt)
    finally:reader.after_check()
    f=dest/'all-scalars.private.jsonl'
    with f.open('x') as out:
        for row in scalars:out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    if len(scalars)!=19510 or new_reads!=1294 or len(completed)!=6:raise ValueError('complete scoring denominator')
    result=dict(status='COMPLETE',completed=completed,failed=[],scalar_rows=len(scalars),scalar_sha256=_digest(f),source_lock_sha256=lock['sha256'],mask_reads_resume=dict(reader.counts),embargo='all six online trajectories sealed and retired before first CPU label read',target_soft_metrics='NA: masks only',preserved_prior_rows=11706-new_reads,new_rows=new_reads,repair='CPU deadline extension within unchanged user absolute cap; no GPU rerun',resume_plan_sha256=sha(plan))
    save(root/'SCORER_RECEIPT.json',result);return result


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer only')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    c=quick.config();root=Path(c['output_root']);start=time.time();deadline=min(start+600,c['origin']['absolute_deadline_epoch']-300);failure=None;result=None
    record=dict(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=Path('/proc/self/stat').read_text().split()[21],active=True,phase='score',attempt=1,started=start,deadline=deadline,assignment=None,argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
    save(root/'processes/score_resume.json',record)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU repair absolute deadline')));signal.alarm(max(1,int(deadline-start)))
    with lease(root/'scorer_resume_control',dict(config=sha(c),attempt=1)):
        try:result=resume(c,deadline)
        except BaseException as e:failure=dict(error_type=type(e).__name__,reason=str(e));traceback.print_exc()
        finally:signal.alarm(0)
        cost=dict(gpu_seconds=0,model_forwards=0,backward_calls=0,optimizer_steps=0,vjp_calls=0)
        save(root/'attempts/score.1.json',dict(status='FAILED' if failure else 'COMPLETE',phase='score',attempt=1,started=start,ended=time.time(),wall_seconds=time.time()-start,cost=cost,failure=failure,result=result,code_sha=c['code_sha'],repair_code_sha=os.environ['REPAIR_SHA'],config_sha256=sha(c)))
        state=read(root/'RUN_STATE.json');state.update(status='PARTIAL' if failure else 'COMPLETE',ended=time.time(),delivery='PENDING_LOCAL_GITHUB',target_scores_embargoed=False);state['jobs']['score']='FAILED' if failure else 'COMPLETE';save(root/'RUN_STATE.json',state)
        quick.base.ID=quick.ID;quick.base.update_ledger(c,state);original_score.ARMS=quick.ARMS;quick.report(c,state)
    record.update(active=False,ended=time.time(),exit_code=1 if failure else 0);save(root/'processes/score_resume.json',record)
    if failure:raise SystemExit(1)
