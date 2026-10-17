"""Bounded CPU selector followed by an isolated target-label scorer."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from ..r51_same_state.run import read, save, run_child
from .score import features, proxy_mask, candidate_score, choose


def access_guard(root, source_paths):
    root=Path(root).resolve(); mount=root.parent
    sources={Path(p).resolve(strict=True) for p in source_paths}
    libraries=[Path(sys.prefix).resolve(),root/'code']
    current={'image':None}; opened=set(); rejected=[]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        protected=p.is_relative_to(root/'scorer')
        allowed=p in sources or p==current['image'] or p.is_relative_to(root) or any(p.is_relative_to(x) for x in libraries)
        asset=p.suffix.lower() in ('.png','.jpg','.jpeg','.pt','.pth','.npy','.npz','.bits')
        if protected or (p.is_relative_to(mount) or asset) and not allowed:
            rejected.append(str(p));raise PermissionError('unregistered data or target-label access')
        if p.is_relative_to(mount):opened.add(str(p))
    sys.addaudithook(audit)
    return current,opened,rejected


def guard_check(c):
    root=Path(c['output_root']);private=root/'private'
    source=private/'synthetic-source';source.write_text('fixture')
    link=private/'synthetic-escape';link.symlink_to(root.parent/'forbidden.png')
    current,_,_=access_guard(root,[source])
    assert source.read_text()=='fixture'
    for p in [root/'scorer/denial-probe',root.parent/'forbidden.png',link]:
        try:p.read_bytes()
        except PermissionError:continue
        raise AssertionError('guard failed closed')
    link.unlink();source.unlink()
    save(root/'GUARD_CHECK.json',dict(status='PASS',synthetic_only=True,source_allowlist=True,
         scorer_denied=True,unregistered_asset_denied=True,symlink_escape_denied=True))


def worker(c):
    import torch
    from ..r7_target_screen.runner import TargetReader,file_identity
    root=Path(c['output_root']);started=time.time()
    status=dict(status='QUALIFYING',formal_pairs=0,started=started)
    save(root/'RUN_STATE.json',status)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    references=read(root/'private/references.json');panel=read(root/'private/images.json')
    if len(references)!=8 or len(panel)!=48:raise ValueError('fixed input coverage')
    allowed=[r[k+'_path'] for r in references for k in ('image','mask')]
    current,opened,rejected=access_guard(root,allowed)
    for p in [root/'scorer/panel.json',Path(c['denial_mask_path'])]:
        try:p.read_bytes()
        except PermissionError:continue
        raise AssertionError('target-label denial failed')
    image_reader=TargetReader(c['data_root'],256*1024**2,'image')
    reference_reader=TargetReader(c['data_root'],256*1024**2,'mask')
    def image(row):return image_reader.read(row).numpy()[0].transpose(1,2,0).copy()
    try:
        ref_features=np.stack([features(image(r)) for r in references])
        ref_masks=np.stack([proxy_mask(reference_reader.read(r).numpy()[0].astype(bool)) for r in references])
        with (root/'private/choices.jsonl').open('x') as trace:
            for order in (0,1):
                for i,row in enumerate(panel):
                    if time.time()-started>c['selector_cap_seconds']-20:raise TimeoutError('selector budget')
                    current['image']=Path(row['image_path']).resolve(strict=True)
                    x=features(image(row));current['image']=None
                    pid=order*48+i
                    raw=(root/'private/candidates'/f'{pid:03d}.bits').read_bytes()
                    if len(raw)!=2*65536:raise ValueError('candidate mask coverage')
                    masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,2,512,512).astype(bool)
                    tick=time.time();results=[candidate_score(x,m,ref_features,ref_masks) for m in masks]
                    seconds=time.time()-tick;scores=[q[0] for q in results];selected=choose(scores)
                    if pid==0:
                        repeat=[candidate_score(x,m,ref_features,ref_masks) for m in masks]
                        if any(a[0]!=b[0] or not np.array_equal(a[1],b[1]) for a,b in zip(results,repeat)):
                            raise ValueError('real-pair repeatability failure')
                        projection=(time.time()-started)+95*seconds*1.5+120
                        if projection>c['selector_cap_seconds']-20:raise TimeoutError('full matrix not admitted')
                        save(root/'QUALIFICATION.json',dict(status='PASS',real_pair_repeat_exact=True,
                             first_pair_seconds=seconds,projected_selector_seconds=projection,
                             references=8,source_masks=8,target_label_reads=0))
                        status['status']='SELECTING';save(root/'RUN_STATE.json',status)
                    with (root/'predictions'/f'{pid:03d}.bits').open('xb') as f:
                        f.write(raw);f.write(raw[selected*65536:(selected+1)*65536])
                    trace.write(json.dumps(dict(pair=pid,selected=selected,scores=scores,seconds=seconds))+'\n');trace.flush()
                    status['formal_pairs']+=1
                    if status['formal_pairs']%8==0:save(root/'RUN_STATE.json',status)
        for path,(_,identity) in image_reader.used.items():
            if file_identity(path)!=identity:raise ValueError('image metadata changed')
        for path,(_,identity) in reference_reader.used.items():
            if file_identity(path)!=identity:raise ValueError('source mask metadata changed')
        save(root/'private/ACCESS_AUDIT.json',dict(source_images=8,source_masks=8,target_label_reads=0,
             target_image_reads=96,opened_paths=sorted(opened),rejected=rejected))
        new_bytes=sum(p.stat().st_size for name in ('private','predictions','logs') for p in (root/name).rglob('*') if p.is_file())
        if new_bytes>2*1024**3:raise ValueError('artifact budget exceeded')
        status.update(status='SELECTION_COMPLETE',new_artifact_bytes=new_bytes)
        return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e));raise
    finally:
        status.update(ended=time.time(),worker_seconds=time.time()-started);save(root/'RUN_STATE.json',status)


def score(c):
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import interval,csvout
    root=Path(c['output_root']);state=read(root/'RUN_STATE.json');retired=read(root/'processes/worker.json')
    if state['status']!='SELECTION_COMPLETE' or state['formal_pairs']!=96 or retired['active'] or retired['exit_code']!=0:
        raise ValueError('complete successful selector retirement required')
    panel=read(root/'scorer/panel.json');traces=[json.loads(s) for s in (root/'private/choices.jsonl').read_text().splitlines()]
    if len(panel)!=48 or [r['pair'] for r in traces]!=list(range(96)):raise ValueError('scoring coverage')
    reader=TargetReader(c['data_root'],256*1024**2,'mask');values=np.empty((48,2,3,2))
    for i,row in enumerate(panel):
        label=reader.read(row)
        for order in (0,1):
            raw=(root/'predictions'/f'{order*48+i:03d}.bits').read_bytes()
            if len(raw)!=3*65536:raise ValueError('selected output coverage')
            masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(3,2,512,512).astype(bool)
            for b,m in enumerate(masks):values[i,order,b]=[100*x['dice'] for x in hard_metrics(m,label)]
    for path,(_,identity) in reader.used.items():
        if file_identity(path)!=identity:raise ValueError('target mask metadata changed')
    np.save(root/'scorer/scores.private.npy',values)
    domains=np.array([r['domain'] for r in panel]);assert all(sum(domains==d)==12 for d in set(domains))
    # The oracle selects a whole candidate mask using joint OD/OC target Dice.
    truth_choice=values[:,:,:2].mean(3).argmax(2)
    oracle=np.take_along_axis(values[:,:,:2],truth_choice[:,:,None,None],axis=2)[:,:,0]
    contrasts=dict(selector_minus_CW=values[:,:,2]-values[:,:,0],selector_minus_VPTTA=values[:,:,2]-values[:,:,1],
                   selector_minus_random=values[:,:,2]-values[:,:,:2].mean(2),oracle_minus_CW=oracle-values[:,:,0],
                   oracle_minus_selector=oracle-values[:,:,2])
    summaries=[];cells=[]
    for name,delta in contrasts.items():
        summaries.append(dict(contrast=name,mean_pp=float(delta.mean()),ci95_pp=interval(delta,domains),order_pp=delta.mean((0,2)).tolist()))
        for order in (0,1):
            for domain in sorted(set(domains)):
                for ch,channel in enumerate(('OD','OC')):
                    cells.append(dict(contrast=name,order=order,domain=domain,channel=channel,delta_pp=float(delta[domains==domain,order,ch].mean())))
    main=summaries[0];primary=[r for r in cells if r['contrast']=='selector_minus_CW']
    rv=float(np.mean([r['delta_pp'] for r in primary if r['domain']=='REFUGE_Valid' and r['channel']=='OC']))
    gate=main['mean_pp']>=.1 and main['ci95_pp'][0]>0 and min(main['order_pp'])>0 and min(r['delta_pp'] for r in primary)>=-2 and rv>=1
    selections=np.array([r['selected'] for r in traces]).reshape(2,48).T
    distinguish=values[:,:,0].mean(2)!=values[:,:,1].mean(2)
    result=dict(status='COMPLETE',allocation_gate=bool(gate),contrasts=summaries,refuge_valid_OC_pp=rv,
                selected_VPTTA=int(selections.sum()),selected_CW=int((selections==0).sum()),
                target_non_tied_pairs=int(distinguish.sum()),pair_ranking_accuracy=float((selections[distinguish]==truth_choice[distinguish]).mean()) if distinguish.any() else None,
                independent_validation=False,source_references=8,source_free=False)
    out=root/'public';save(out/'RESULTS.json',result);csvout(out/'ALL_PAIRED_CELLS.csv',cells)
    csvout(out/'ALL_NEGATIVE_CELLS.csv',[r for r in cells if r['delta_pp']<0]);save(out/'QUALIFICATION.json',read(root/'QUALIFICATION.json'))
    save(out/'COMPLETION_AUDIT.json',dict(pairs=96,contents=48,orders=2,predictions=288,source_references=8,online_target_label_reads=0,selector_retired_exit=0))
    (out/'REPORT.md').write_text('# R63 RCA diagnostic\n\n'+json.dumps(result,indent=2)+'\n\nSource-reference-assisted output selection; exposed development evidence only.\n')
    state.update(status='COMPLETE',scored_predictions=288,ended=time.time());save(root/'RUN_STATE.json',state)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only run required')
    c=read(os.environ['DIAG_CONFIG']);root=Path(c['output_root']);mode=os.environ['DIAG_MODE']
    if not c['source_reference_authorized'] or c['source_references']!=8:raise ValueError('explicit limited source exception required')
    if mode=='guard_check':guard_check(c);return 0
    if mode=='worker':return worker(c)
    if mode=='score':score(c);return 0
    if mode!='controller':raise ValueError('unknown mode')
    if read(root/'GUARD_CHECK.json')['status']!='PASS':raise ValueError('guard qualification required')
    with (root/'started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    attempts=[]
    try:
        attempts.append(run_child(c,'worker',c['selector_cap_seconds'],False))
        if attempts[-1]['exit_code']==0:
            remaining=c['cpu_cap_seconds']-c['prior_cpu_seconds']-attempts[-1]['cpu_worker_seconds']
            if remaining<30:raise TimeoutError('no scoring budget')
            attempts.append(run_child(c,'score',min(900,remaining),False))
    finally:
        ledger=dict(attempts=attempts,gpu_worker_seconds=0,cpu_worker_seconds=c['prior_cpu_seconds']+sum(a['cpu_worker_seconds'] for a in attempts),
                    prior_preparation_seconds=c['prior_cpu_seconds'],all_workers_retired=True)
        save(root/'RESOURCE_LEDGER.json',ledger);save(root/'public/RESOURCE_LEDGER.json',ledger)
    return 0 if len(attempts)==2 and all(a['exit_code']==0 for a in attempts) else 1


if __name__=='__main__':raise SystemExit(main())
