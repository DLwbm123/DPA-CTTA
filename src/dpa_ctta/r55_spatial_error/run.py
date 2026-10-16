"""Fixed largest-component error attribution on retired predictions."""
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from ..r10_12h_core.run import read, save
from ..r20_model_only_search.report import csvout
from ..r7_target_screen.runner import TargetReader


def largest(mask):
    components, n = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if not n: return mask.copy()
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    return components == int(sizes.argmax())


def main():
    a=np.zeros((5,5),dtype=bool);a[0,:2]=True;a[4,4]=True
    assert largest(a).sum()==2 and not largest(np.zeros_like(a)).any()
    a=np.eye(3,dtype=bool);assert largest(a).sum()==3
    c=read(os.environ['RUN_CONFIG']);root=Path(c['output_root']);src=Path(c['r52_root']);cw=Path(c['r50_root'])
    started=time.time();state=dict(status='RUNNING',started=started,GPU_seconds=0)
    save(root/'RUN_STATE.json',state)
    try:
        assert read(src/'RUN_STATE.json')['status']=='COMPLETE'
        reader=TargetReader(c['target_root'],256*1024**2,'mask');groups=defaultdict(lambda:np.zeros(6,dtype=np.int64));count=0
        for seed in (20260907,17011,29009):
            for order in (0,1):
                manifest=read(src/'scorer'/f'FULL_o{order}.json');path=cw/'target'/f'CW_LSO_s{seed}_o{order}'
                assert read(path/'online_complete.json')['visits']==1951
                assert (path/'predictions.bits').stat().st_size==1951*65536
                with (path/'predictions.bits').open('rb') as f:
                    for m in manifest:
                        packed=f.read(65536);assert len(packed)==65536
                        if m['subset']!='remaining_dev':continue
                        assert time.time()-started<3600
                        mask=np.unpackbits(np.frombuffer(packed,dtype=np.uint8)).reshape(2,512,512).astype(bool)
                        gt=reader.read(m).numpy()[0].astype(bool)
                        for k,ch in enumerate(('OD','OC')):
                            kept=largest(mask[k]);removed=mask[k]&~kept
                            tp=int((mask[k]&gt[k]).sum());fp=int((mask[k]&~gt[k]).sum());actual=int(gt[k].sum())
                            groups[(seed,order,m['domain'],ch)]+=np.array([1,tp,fp,actual,int((removed&gt[k]).sum()),int((removed&~gt[k]).sum())])
                        count+=1
                save(root/'PROGRESS.json',dict(paired_images=count,seed=seed,order=order))
        reader.after_check();assert count==10170 and len(groups)==48
        rows=[]
        for (seed,order,domain,channel),v in sorted(groups.items()):
            n,tp,fp,actual,removed_tp,removed_fp=map(int,v)
            rows.append(dict(seed=seed,order=order,domain=domain,channel=channel,images=n,TP=tp,FP=fp,GT=actual,removed_TP=removed_tp,removed_FP=removed_fp,removed_FP_fraction=removed_fp/fp if fp else 0.,removed_TP_per_GT=removed_tp/actual if actual else 0.))
        csvout(root/'public/COMPONENT_ERRORS.csv',rows)
        focus=[r for r in rows if r['domain']=='REFUGE_Valid' and r['channel']=='OC'];assert len(focus)==6
        positive=sum(r['removed_FP_fraction']>=.5 and r['removed_TP_per_GT']<=.01 for r in focus)
        decision=dict(fragment_hypothesis_supported=positive>=5,passing_pairs=positive,total_pairs=6,independent_validation=False)
        save(root/'public/DECISION.json',decision)
        (root/'public/REPORT.md').write_text('# R55 spatial error attribution\n\n'+json.dumps(decision,indent=2)+'\n\nFixed 8-connected largest-component diagnostic; all domain/channel/order/seed cells retained. No new training and no prediction replacement. Removing fragments is not established as an efficacy improvement. All results are development evidence.\n')
        (root/'public/PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())
        state.update(status='COMPLETE',ended=time.time(),wall_seconds=time.time()-started,scored_images=count,aggregate_rows=len(rows));save(root/'RUN_STATE.json',state);save(root/'public/RUN_STATE.json',state)
    except BaseException as e:
        state.update(status='FAILED',reason=str(e),ended=time.time());save(root/'RUN_STATE.json',state);raise


if __name__=='__main__':main()
