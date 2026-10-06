"""Two baseline replays, causal probes, then independent CPU-only judge scoring."""
import concurrent.futures
import json
import os
from pathlib import Path
import statistics as st
import sys
import time
import numpy as np
from scipy.ndimage import label
from scipy.stats import rankdata
import torch
from .method import Host, classes
from ..r20_model_only_search import runtime as base
from ..r10_12h_core.run import read, save, sha
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader

ID='R22_CAUSAL_CUT_DIAGNOSTIC';WIDTH=65536
NAMES=('student','source','cut','confidence','entropy')


def online(c,guard,permit,job):
    root=Path(c['output_root']);rows=read(root/'private'/f'SCREEN_o{job["order"]}.json')
    assert rows_sha(rows)==c['manifests']['SCREEN'][job['order']]['sha256']
    h=Host(base.weights(c),job['candidate'],job['seed'],sha(job));base.attach(guard.meter,h)
    reader=TargetReader(c['target_root'],256*1024**2,'image')
    dest=root/'target'/job['id'];dest.mkdir()
    try:
        with (dest/'predictions.bits').open('xb') as bits,(dest/'visits.jsonl').open('x') as traces,(root/'private'/f'BASE_o{job["order"]}.bits').open('rb') as reference:
            for i,row in enumerate(rows):
                guard();x=base.image(reader,row,permit,guard);start=time.perf_counter();z,t=h.step(x);torch.cuda.synchronize()
                raw=reference.read(4*WIDTH);assert len(raw)==4*WIDTH
                assert np.packbits(z.sigmoid().numpy()>=.5).tobytes()==raw[:WIDTH], 'baseline replay changed'
                arrays=[(h.student[0]>=.5).cpu().numpy(),(h.source[0]>=.5).cpu().numpy(),*(h.outputs[k] for k in NAMES[2:]),h.stable.numpy()[0]]
                for a in arrays:bits.write(np.packbits(a).tobytes())
                t.update(visit=i+1,seconds=time.perf_counter()-start,regions=h.regions)
                traces.write(json.dumps(t,allow_nan=False)+'\n');traces.flush()
            assert not reference.read(1)
        h.check_frozen(True)
        result=dict(status='COMPLETE',arrivals=len(rows),baseline_mask_parity=True,prediction_bytes=(dest/'predictions.bits').stat().st_size,source_reads=0,label_reads=0)
        assert result['prediction_bytes']==len(rows)*6*WIDTH
        save(dest/'complete.json',result);return result
    finally:h.close()


def auc(scores,truth):
    scores=np.asarray(scores);truth=np.asarray(truth,dtype=bool);p=int(truth.sum());n=len(truth)-p
    if not p or not n:return None
    return float((rankdata(scores)[truth].sum()-p*(p+1)/2)/(p*n))


def score(c):
    root=Path(c['output_root']);assert all(not read(p)['active'] for p in (root/'processes').glob('*.json'))
    reader=TargetReader(c['target_root'],256*1024**2,'mask');images=[];regions=[]
    for job in c['jobs']:
        rows=read(root/'scorer'/f'SCREEN_o{job["order"]}.json');dest=root/'target'/job['id']
        assert read(dest/'complete.json')['arrivals']==len(rows)==192
        traces=[json.loads(x) for x in (dest/'visits.jsonl').read_text().splitlines()];assert len(traces)==len(rows)
        with (dest/'predictions.bits').open('rb') as bits:
            for row,t in zip(rows,traces):
                raw=bits.read(6*WIDTH);assert len(raw)==6*WIDTH
                masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(6,2,512,512).astype(bool)
                gt=reader.read(row).numpy()[0].astype(bool)
                student,source=masks[:2];stable=masks[5]
                rr,_=label((student.astype(int).sum(0)!=source.astype(int).sum(0)))
                oracle=student.copy()
                for r in t['regions']:
                    m=rr==r['region'];assert int(m.sum())==r['area']
                    corrected=(student[:,m]!=gt[:,m])&(source[:,m]==gt[:,m]);damaged=(student[:,m]==gt[:,m])&(source[:,m]!=gt[:,m])
                    net=int(corrected.sum()-damaged.sum())
                    if net>0:oracle[:,m]=source[:,m]
                    regions.append(dict(**r,order=job['order'],domain=row['domain'],content=row['image_sha256'],corrected=int(corrected.sum()),damaged=int(damaged.sum()),net=net,stable_corrected=int((corrected&stable[:,m]).sum()),stable_damaged=int((damaged&stable[:,m]).sum())))
                metrics={}
                for name,pred in list(zip(NAMES,masks[:5]))+[('region_pixel_oracle',oracle)]:
                    metrics[name]=dict(dice=[float((2*(pred[k]&gt[k]).sum()+1e-6)/(pred[k].sum()+gt[k].sum()+1e-6)*100) for k in (0,1)],corrected=int(((student!=gt)&(pred==gt)).sum()),damaged=int(((student==gt)&(pred!=gt)).sum()),stable_corrected=int(((student!=gt)&(pred==gt)&stable).sum()),stable_damaged=int(((student==gt)&(pred!=gt)&stable).sum()))
                images.append(dict(order=job['order'],domain=row['domain'],content=row['image_sha256'],metrics=metrics,seconds=t['seconds'],collapse_guard=t['diagnostics']['collapse_guard']))
            assert not bits.read(1)
    save(root/'scorer/images.private.json',images);save(root/'scorer/regions.private.json',regions)
    summaries={}
    for name in (*NAMES,'region_pixel_oracle'):
        cells={f'{o}/{d}/{k}':st.mean(r['metrics'][name]['dice'][k] for r in images if r['order']==o and r['domain']==d) for o in (0,1) for d in sorted({r['domain'] for r in images}) for k in (0,1)}
        summaries[name]=dict(macro_Dice_percent=st.mean(cells.values()),cells=cells,order_macro=[st.mean(v for k,v in cells.items() if k.startswith(f'{o}/')) for o in (0,1)],**{k:sum(r['metrics'][name][k] for r in images) for k in ('corrected','damaged','stable_corrected','stable_damaged')})
    judges={};eligible=[r for r in regions if r['eligible']];informative=[r for r in eligible if r['net']!=0]
    for name in NAMES[2:]:
        picked=[r for r in eligible if r['choices'][name]]
        judges[name]=dict(AUROC=auc([r['scores'][name] for r in informative],[r['net']>0 for r in informative]),selected_regions=len(picked),beneficial_regions=sum(r['net']>0 for r in picked),harmful_regions=sum(r['net']<0 for r in picked),tied_regions=sum(r['net']==0 for r in picked),order_net_pixels=[sum(r['net'] for r in picked if r['order']==o) for o in (0,1)],macro_delta_pp=summaries[name]['macro_Dice_percent']-summaries['student']['macro_Dice_percent'],order_delta_pp=[a-b for a,b in zip(summaries[name]['order_macro'],summaries['student']['order_macro'])])
    cut=judges['cut'];valid_auc=[judges[k]['AUROC'] for k in NAMES[2:]]
    promising=len(informative)>=100 and all(x is not None for x in valid_auc) and cut['AUROC']>=max(.6,*valid_auc) and min(cut['order_net_pixels'])>0 and min(cut['order_delta_pp'])>0 and summaries['cut']['stable_corrected']>summaries['cut']['stable_damaged']
    result=dict(status='COMPLETE',signal='PROMISING_JUDGE_ONLY' if promising else 'NOT_ESTABLISHED',unique_images=192,orders=2,seed=20260907,baseline_replays=2,baseline_mask_parity_arrivals=384,regions=len(regions),eligible_regions=len(eligible),informative_regions=len(informative),source_better_regions=sum(r['net']>0 for r in informative),student_better_regions=sum(r['net']<0 for r in informative),oracle_net_pixels=sum(max(0,r['net']) for r in regions),eligible_oracle_net_pixels=sum(max(0,r['net']) for r in eligible),collapse_guard_arrivals=sum(r['collapse_guard'] for r in images),summaries=summaries,judges=judges,probe_seconds_per_image=st.mean(r['seconds'] for r in images),online_label_reads=0,judge_driven_parameter_updates=0,scope='Causal diagnostic on historically exposed SEARCH. Dice describes candidate pseudo-label selection, not a trained method. Region oracle minimizes pixel errors, not global Dice. AUROC excludes true-error ties. Frozen source and adapted student share initialization.')
    save(root/'public/RESULTS.json',result);return result


def supervise():
    c=base.config();root=Path(c['output_root']);start=time.time();save(root/'RUN_STATE.json',dict(status='RUNNING',started=start))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        ff=[pool.submit(base.run_task,c,'online_'+j['id'],g,480,j) for j,g in zip(c['jobs'],c['gpu_assignments'])]
        results=[f.result() for f in ff]
    if any(r['status']!='COMPLETE' for r in results):
        save(root/'RUN_STATE.json',dict(status='INCOMPLETE',reason='fixed replay failed; no automatic retry'));return
    save(root/'stages/ALL_RETIRED.json',dict(status='ALL_WORKERS_RETIRED',at=time.time()))
    out=score(c);out.update(wall_seconds=time.time()-start,gpu_worker_seconds=sum(r['cost']['gpu_seconds'] for r in results),code_sha=c['code_sha']);save(root/'public/RESULTS.json',out)
    save(root/'RUN_STATE.json',dict(status='COMPLETE',ended=time.time(),delivery='PENDING_GITHUB'))


def main():
    base.ID=ID;base.Host=Host;base.online=online
    torch.set_num_threads(2)
    if os.environ['RUN_MODE']=='worker':
        r=base.worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    else:
        try:supervise()
        except BaseException as e:
            c=base.config();save(Path(c['output_root'])/'RUN_STATE.json',dict(status='INCOMPLETE',reason=str(e)));raise


if __name__=='__main__':main()
