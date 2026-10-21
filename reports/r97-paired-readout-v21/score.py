"""Join sealed label-free decisions to existing scores; no new segmentation scoring."""
import json,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def run(c,write):
    start=time.time();root=Path(c['output_root']);old=read(c['prior_config']);r91=Path(old['output_root'])
    assert read(root/'public/SEALED.json')['status']=='SEALED'
    assert read(root/'public/FORMAL_WRAPPER.json')['exit_code']==0
    rec=read(root/'private/formal_process.json');stat=Path('/proc')/str(rec['pid'])/'stat'
    assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[19]!=rec['start_ticks'],'GPU still alive'
    signals=[read(root/'private'/f'signals_{o}.json') for o in (0,1)]
    assert all(s['status']=='SEALED' and s['online_label_reads']==s['online_score_reads']==0 and len(s['rows'])==192 for s in signals)
    with np.load(r91/'scorer/values.private.npz',allow_pickle=False) as f:values=f['values'];domains=f['domains']
    assert values.shape==(192,6,6,2,2,11)
    K=values[:,:2,2];Q=values[:,:2,4];diff=Q-K
    keys=('U','entropyD','entropyFull','singleR','noAction','noHistory','Earea','Darea')
    scores={k:np.zeros((192,2)) for k in keys};z=np.zeros((192,2),bool);visits=np.zeros((192,2),int)
    for o,s in enumerate(signals):
        seen=set()
        for row in s['rows']:
            i=row['index'];assert i not in seen;seen.add(i);assert row['order']==o
            z[i,o]=row['z'];visits[i,o]=row['visit']
            for k in keys:scores[k][i,o]=row[k]
        assert seen==set(range(192)) and sorted(visits[:,o])==list(range(192))
    rules={'main':z,'entropyD':scores['entropyD']>0,'entropyFull_aux':scores['entropyFull']>0,
           'singleR':scores['singleR']>.5,'noAction':scores['noAction']>.5,'noHistory':scores['noHistory']>.5,
           'always_Q':np.ones_like(z),'always_K':np.zeros_like(z)}
    def matched(sc):
        out=np.zeros_like(z)
        for o in (0,1):
            for b in range(6):
                ids=np.flatnonzero(visits[:,o]//32==b);n=int(z[ids,o].sum())
                ranking=sorted(ids,key=lambda i:(-sc[i,o],visits[i,o]));out[ranking[:n],o]=True
        return out
    for k in ('entropyD','singleR','noAction','noHistory','Earea','Darea'):rules[k+'_matched']=matched(scores[k])
    def macro(x):
        # Equal domains; missing pairs reported separately, never silently used for Dice.
        return float(np.mean([x[domains==d].mean() for d in range(4)]))
    def summary(sel,name):
        out=[]
        for o in (0,1):
            for vi,variant in enumerate(('raw','post')):
                g=diff[:,o,vi,:,0].mean(1);assert np.isfinite(g).all()
                action=sel[:,o]
                out.append(dict(rule=name,order=o,variant=variant,G_pp=macro(action*g),
                    switch_loss_pp=macro(action*np.maximum(-g,0)),missed_gain_pp=macro((~action)*np.maximum(g,0)),
                    switches=int(action.sum()),positive_switches=int((action&(g>0)).sum()),negative_switches=int((action&(g<0)).sum()),
                    worst_selected_input_pp=float(g[action].min()) if action.any() else None,
                    best_selected_input_pp=float(g[action].max()) if action.any() else None))
        return out
    rows=[r for name,sel in rules.items() for r in summary(sel,name)]
    main=[r for r in rows if r['rule']=='main' and r['variant']=='raw'];G=float(np.mean([r['G_pp'] for r in main]))
    gains={name:float(np.mean([r['G_pp'] for r in rows if r['rule']==name and r['variant']=='raw'])) for name in rules}
    rng=np.random.default_rng(20261021);random_g=[]
    for _ in range(1000):
        zz=matched(rng.random(z.shape));random_g.append(np.mean([macro(zz[:,o]*diff[:,o,0,:,0].mean(1)) for o in (0,1)]))
    # Descriptive sample, not a calibrated p-value or independent cohort uncertainty.
    random_result=dict(repeats=1000,seed=20261021,mean_G_pp=float(np.mean(random_g)),p025=float(np.quantile(random_g,.025)),p975=float(np.quantile(random_g,.975)),matching='within-order contiguous 32-arrival blocks; exact main action count')
    metrics=('Dice','FP','FN','foreground','ASSD_px','HD95_px','vCDR_abs_error','nesting_pixel_percent','nesting_present_percent','pred_empty','gt_empty')
    cells=[];protect={'raw':True,'post':True};failed=[]
    for o in (0,1):
        for vi,variant in enumerate(('raw','post')):
            new=np.where(z[:,o,None,None],Q[:,o,vi],K[:,o,vi])
            for d in range(4):
                for h,head in enumerate(('OD','OC')):
                    for mi,metric in enumerate(metrics):
                        if metric=='vCDR_abs_error' and h==1:continue
                        a=K[domains==d,o,vi,h,mi];b=new[domains==d,h,mi];valid=np.isfinite(a)&np.isfinite(b)
                        delta=float(np.mean(b[valid]-a[valid])) if valid.any() else None
                        row=dict(order=o,variant=variant,domain=f'D{d}',head=head,metric=metric,delta=delta,n=int(len(a)),paired_finite=int(valid.sum()),NA_K=int((~np.isfinite(a)).sum()),NA_M=int((~np.isfinite(b)).sum()))
                        cells.append(row)
                        if metric in ('Dice','ASSD_px','HD95_px','vCDR_abs_error'):
                            # Lack of finite evidence fails protection; do not silently drop NA cells.
                            passed=delta is not None and row['NA_M']<=row['NA_K'] and (delta>=0 if metric=='Dice' else delta<=0)
                            if not passed:protect[variant]=False;failed.append(row)
    core=['entropyD','entropyD_matched','singleR_matched','noHistory_matched','noAction_matched','Earea_matched','Darea_matched']
    gain_gate=all(r['G_pp']>0 and r['switches']>=1 for r in main) and G>=.1
    info_gate=all(G>gains[k] for k in core) and G>random_result['p975']
    result=dict(status='DEVELOPMENT_PASS_NOT_CONFIRMATION' if gain_gate and info_gate and protect['raw'] and protect['post'] else 'FROZEN_VERSION_NOT_PROMOTED',
        gain_gate=bool(gain_gate),information_gate=bool(info_gate),raw_protection=protect['raw'],post_protection=protect['post'],
        main_macro_gain_pp=G,rule_gains_pp=gains,main_orders=main,random=random_result,failed_protection_cells=failed,
        new_label_reads=0,new_model_calls=0,old_score_rows_joined=384,independent_confirmation=False,
        decision='Prepare independent admission only' if gain_gate and info_gate and protect['raw'] and protect['post'] else 'Close this frozen signal version; no threshold/region/fusion rescue',
        CPU_body_seconds=time.time()-start)
    write('public/RULE_SUMMARY.json',rows);write('public/PROTECTION_CELLS.json',cells);write('public/RESULT.json',result)
    # Per-input action/cost vectors stay private, with no published linkage rows.
    details=[dict(index=i,order=o,selected=int(z[i,o]),gain_raw_classes=diff[i,o,0,:,0].tolist(),gain_post_classes=diff[i,o,1,:,0].tolist()) for o in (0,1) for i in range(192)]
    write('private/action_values.json',details)
    return dict(status=result['status'],main_gain_pp=G,CPU_body_seconds=time.time()-start)
