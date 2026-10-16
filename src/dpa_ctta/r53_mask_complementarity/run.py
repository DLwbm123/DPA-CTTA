"""Two-stage CPU diagnostic; freeze all fused masks before loading any labels."""
import json
import os
import time
from pathlib import Path
import numpy as np
from ..r10_12h_core.run import read, save
from ..r20_model_only_search.report import aggregate, bootstrap_pair, csvout
from ..r24_c_context.run import hard_metrics
from ..r7_target_screen.runner import TargetReader
from ..r16_evidence_correction.structure import containment

WIDTH=65536
SEEDS=(20260907,17011,29009)


def combine(a,b):
    if a.shape!=b.shape or a.dtype!=np.uint8 or b.dtype!=np.uint8:
        raise ValueError('paired packed uint8 masks required')
    return np.bitwise_or(a,b),np.bitwise_and(a,b)


def self_check():
    a=np.array([0,255,85],dtype=np.uint8);b=np.array([255,0,51],dtype=np.uint8)
    union,inter=combine(a,b)
    assert union.tolist()==[255,255,119] and inter.tolist()==[0,0,17]
    assert np.array_equal(combine(a,a)[0],a) and np.array_equal(combine(a,a)[1],a)
    try:combine(a,b[:2])
    except ValueError:pass
    else:raise AssertionError('shape guard')


def main():
    import torch
    torch.set_num_threads(2)
    os.umask(0o077);self_check();c=read(os.environ['RUN_CONFIG']);root=Path(c['output_root'])
    started=time.time();state=dict(status='GENERATING_MASKS',started=started,GPU_seconds=0)
    save(root/'RUN_STATE.json',state)
    def guard():
        if time.time()-started>7200:raise TimeoutError('two-hour CPU wall cap')
    try:
        src=Path(c['r52_root']);cw=Path(c['r50_root'])
        assert read(src/'RUN_STATE.json')['status']=='COMPLETE'
        assert not any(read(p).get('active') for p in (src/'processes').glob('*.json'))
        for seed in SEEDS:
            for order in (0,1):
                guard();jid=f's{seed}_o{order}'
                paths=[cw/'target'/f'CW_LSO_{jid}',src/'target'/f'VPTTA_{jid}']
                for p in paths:
                    assert read(p/'online_complete.json')['visits']==1951
                    assert (p/'predictions.bits').stat().st_size==1951*WIDTH
                with (paths[0]/'predictions.bits').open('rb') as aa,(paths[1]/'predictions.bits').open('rb') as bb,\
                     (root/'predictions'/f'UNION_{jid}.bits').open('xb') as uu,(root/'predictions'/f'INTERSECTION_{jid}.bits').open('xb') as ii:
                    for visit in range(1951):
                        a=np.frombuffer(aa.read(WIDTH),dtype=np.uint8);b=np.frombuffer(bb.read(WIDTH),dtype=np.uint8)
                        assert len(a)==len(b)==WIDTH
                        union,inter=combine(a,b);uu.write(union.tobytes());ii.write(inter.tobytes())
                    assert not aa.read(1) and not bb.read(1)
        save(root/'PREDICTIONS_COMPLETE.json',dict(at=time.time(),trajectories=12,visits=23412,target_labels_read=0,postprocessing_only=True))
        state['status']='CPU_SCORING';save(root/'RUN_STATE.json',state)
        assignment=read(src/'scorer/SPLIT.private.json');reader=TargetReader(c['target_root'],256*1024**2,'mask');allrows=[]
        for seed in SEEDS:
            for order in (0,1):
                guard();jid=f's{seed}_o{order}';manifest=read(src/'scorer'/f'FULL_o{order}.json')
                for method,path in [('CW_LSO',src/'scorer/reused'/f'CW_LSO_{jid}.private.jsonl'),('VPTTA',src/'scores/final'/f'VPTTA_{jid}.private.jsonl')]:
                    rs=[json.loads(line) for line in path.read_text().splitlines()]
                    assert len(rs)==1951 and all((row['content'],row['visit'])==(m['image_sha256'],i+1) for i,(row,m) in enumerate(zip(rs,manifest)))
                    allrows.extend(rs)
                with (root/'predictions'/f'UNION_{jid}.bits').open('rb') as uu,(root/'predictions'/f'INTERSECTION_{jid}.bits').open('rb') as ii,\
                     (root/'scores'/f'{jid}.private.jsonl').open('x') as log:
                    for visit,m in enumerate(manifest,1):
                        label=reader.read(m)
                        for method,bits in [('UNION',uu),('INTERSECTION',ii)]:
                            mask=np.unpackbits(np.frombuffer(bits.read(WIDTH),dtype=np.uint8)).reshape(2,512,512).astype(bool)
                            row=dict(condition=method,seed=seed,order=order,visit=visit,content=m['image_sha256'],domain=m['domain'],subset=m['subset'],role=assignment.get(m['image_sha256'],'CONTEXT'),metrics=hard_metrics(mask,label),containment_violations=containment(mask),empty_foreground=[not x.any() for x in mask],diagnostics={},seconds=None)
                            log.write(json.dumps(row)+'\n');allrows.append(row)
                save(root/'scores'/f'{jid}.complete.json',dict(status='COMPLETE',rows=3902))
        reader.after_check();assert len(allrows)==46824
        out=root/'public';main,domains=aggregate(allrows);csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domains)
        primary=[dict(r,role='PRIMARY_ALL') for r in allrows if r['subset']=='remaining_dev'];pairs=[];intervals=[];negative=[]
        for candidate in ('UNION','INTERSECTION'):
            for baseline in ('CW_LSO','VPTTA'):
                pair,ci,neg=bootstrap_pair(primary,candidate,baseline,list(SEEDS),role='PRIMARY_ALL')
                pairs.append(pair);intervals.extend(ci);negative.extend(neg)
        csvout(out/'PAIRED_SUMMARY.csv',pairs);csvout(out/'CONTENT_BOOTSTRAP_CI.csv',intervals);csvout(out/'NEGATIVE_CELLS.csv',negative)
        p=next(p for p in pairs if p['candidate']=='UNION' and p['baseline']=='CW_LSO');v=next(p for p in pairs if p['candidate']=='UNION' and p['baseline']=='VPTTA')
        passed=p['delta_pp']>=.1 and p['CI_low_pp']>0 and min(p['order_delta_pp'])>0 and p['positive_trajectories']>=5 and p['imageweighted_delta_pp']>=0 and p['worst_seed_averaged_cell_pp']>=-2 and v['worst_seed_averaged_cell_pp']>=-14.326778243134322
        save(out/'FINAL_DECISION.json',dict(primary='UNION',development_signal=passed,independent_validation=False,training_runs=0,new_adaptive_trajectories=0))
        lines=['# R53 fixed spatial-complementarity diagnostic','','| Output | Primary macro Dice (%) |','|---|---:|']
        import statistics as st
        for method in ('CW_LSO','VPTTA','UNION','INTERSECTION'):
            values=[x['macro_Dice_percent'] for x in main if x['condition']==method and x['role']=='PRIMARY_ALL'];assert len(values)==6
            lines.append(f'| {method} | {st.mean(values):.6f} |')
        lines+=['',f'Frozen UNION development gate: {passed}.',
            'This replays fixed per-image output operators on stored predictions from two independently adapted models; it performs no new training. Labels were first read after all union/intersection masks were written. Union and intersection do not use domain labels, image labels, thresholds or fitted selectors.',
            'UNION minus CW_LSO is the sole primary inferential contrast. Other contrasts are descriptive; no best-arm promotion or independent-validation claim. Both source model execution costs would be needed in deployment. Offline CPU runtime is not end-to-end deployment latency.',
            'All content is development-exposed. Frozen predictions, individual records and labels remain private.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())
        state.update(status='COMPLETE',ended=time.time(),wall_seconds=time.time()-started,scored_rows=len(allrows));save(root/'RUN_STATE.json',state);save(out/'RUN_STATE.json',state)
    except BaseException as e:
        state.update(status='FAILED',reason=str(e),ended=time.time());save(root/'RUN_STATE.json',state);raise


if __name__=='__main__':main()
