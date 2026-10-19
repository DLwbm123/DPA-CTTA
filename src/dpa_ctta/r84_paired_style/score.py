"""Terminal-only paired factorial analysis; no online metric selection."""
import json,os,time
from pathlib import Path
import numpy as np
from .method import ARMS,SEEDS,CELLS
from .run import jobs,WIDTH
from ..r51_same_state.run import read,save
from ..r79_mechanism_control.score import csvout,summaries


def decompose(values,p0,native=None):
    H=values[:,:,:,:,0]-p0[:,:,:,None,:];C=values[:,:,:,:,1]-values[:,:,:,:,0];T=H+C
    if not np.allclose(T,values[:,:,:,:,1]-p0[:,:,:,None,:],atol=1e-10):raise ValueError('H+C=T')
    if native is None:native=np.zeros((len(values),6,2,2))
    nh=native[:,:,0]-p0[:,:,0];nc=native[:,:,1]-native[:,:,0]
    terms={}
    for phase,v,n in (('HISTORY',H,nh),('CURRENT',C,nc),('TOTAL',T,nh+nc)):
        terms['VERTEX/'+phase+'/ORIGINAL']=v[:,:,0,0]
        terms['VERTEX/'+phase+'/PAIRED_STYLE']=v[:,:,0,1]
        terms['VERTEX/'+phase+'/PAIRED_STYLE_MINUS_ORIGINAL']=v[:,:,0,1]-v[:,:,0,0]
        terms['VERTEX_PAIRED_STYLE-NATIVE_ORIGINAL/'+phase]=v[:,:,0,1]-n
        terms['VERTEX_ORIGINAL-NATIVE_ORIGINAL/'+phase]=v[:,:,0,0]-n
    return terms


def self_check():
    v=np.zeros((8,6,1,2,2,2));p=np.zeros((8,6,1,2));v[:,:,:,1,0]=2;v[:,:,:,1,1]=5
    t=decompose(v,p)
    assert np.all(t['VERTEX/HISTORY/PAIRED_STYLE']==2) and np.all(t['VERTEX/CURRENT/PAIRED_STYLE']==3)
    assert np.all(t['VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL']==5)
    print('PAIRED_STYLE_FEEDBACK_SCORE_SELF_CHECK_PASS',flush=True)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only label scorer')
    root=Path(c['output_root']);start=time.time();old=Path(c['reference_root'])
    for name in ('qualify_worker0','formal_worker0','formal_worker1'):
        p=read(root/'processes'/f'{name}.json');s=read(root/'private'/name/'RUN_STATE.json')
        if p['active'] or p['exit_code']!=0 or s['status']!='GPU_COMPLETE':raise ValueError('all GPU retirement barrier')
        if s['physical_updates']!=s['completed_physical_updates']:raise ValueError('attempt coverage')
    if read(old/'public/RESULTS.json')['status']!='COMPLETE' or not read(old/'public/TERMINAL_AUDIT.json')['all_registered_pids_absent']:
        raise ValueError('reused R80 completed endpoints required')
    panel=read(root/'scorer/screen_o0.json');index={r['image_sha256']:i for i,r in enumerate(panel)}
    if len(index)!=192 or [r['image_sha256'] for r in panel]!=[r['image_sha256'] for r in read(old/'scorer/screen_o0.json')]:
        raise ValueError('exact paired reference cohort')
    for o in (0,1):
        if [r['image_sha256'] for r in read(root/'scorer'/f'screen_o{o}.json')]!=[r['image_sha256'] for r in read(old/'scorer'/f'screen_o{o}.json')]:
            raise ValueError('reused reference order mismatch')
    domains=np.array([r['domain'] for r in panel]);names=sorted(set(domains))
    reader=TargetReader(c['target_root'],256*1024**2,'mask');labels={r['image_sha256']:reader.read(r) for r in panel}
    values=np.full((192,6,1,2,2,2),np.nan);errors=np.full((*values.shape,3),np.nan)
    reused=np.load(old/'scorer/values.private.npz')
    values[:,:,:,0]=reused['values'][:,:,1:2,3];errors[:,:,:,0]=reused['errors'][:,:,1:2,3]
    p0=reused['p0'][:,:,1:2];p0errors=reused['p0errors'][:,:,1:2]
    updates=0;mechanisms=[]
    for slot in (0,1):
        for job in jobs(slot):
            folder=root/'jobs'/job['id'];receipt=read(folder/'COMPLETE.json');updates+=receipt['physical_updates']
            rows=read(root/'scorer'/f'screen_o{slot}.json');a=ARMS.index(job['arm']);j=SEEDS.index(job['seed'])*2+slot;cell=1
            if receipt['arrivals']!=192 or receipt['physical_updates']!=192:raise ValueError('incomplete job')
            with (folder/'predictions.bits').open('rb') as f:
                for row in rows:
                    raw=f.read(2*WIDTH)
                    if len(raw)!=2*WIDTH:raise ValueError('truncated predictions')
                    masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,2,512,512).astype(bool)
                    i=index[row['image_sha256']];label=labels[row['image_sha256']];truth=label.numpy()[0].astype(bool)
                    for k in (0,1):
                        values[i,j,a,cell,k]=[100*v['dice'] for v in hard_metrics(masks[k],label)]
                        for ch in (0,1):errors[i,j,a,cell,k,ch]=[(masks[k,ch]&~truth[ch]).mean(),(~masks[k,ch]&truth[ch]).mean(),masks[k,ch].mean()]
                if f.read(1):raise ValueError('extra predictions')
            ts=[json.loads(v) for v in (folder/'diagnostics.jsonl').read_text().splitlines()]
            if [t['visit'] for t in ts]!=list(range(1,193)):raise ValueError('diagnostic coverage')
            ds=[t['K'] for t in ts]
            mechanisms.append(dict(cell=job['cell'],arm=job['arm'],seed=job['seed'],order=slot,arrivals=192,
                mean_BN_gradient=float(np.mean([d['BN_gradient_norm'] for d in ds])),mean_BN_update=float(np.mean([d['BN_update_norm'] for d in ds])),
                mean_BCE=float(np.mean([d['loss_parts']['BCE'] for d in ds])),mean_LSO=float(np.mean([d['loss_parts']['LSO'] for d in ds]))))
    if updates!=1152 or not np.isfinite(values).all() or not np.isfinite(errors).all():raise ValueError('full 1152 update coverage')
    reader.after_check();np.savez(root/'scorer/values.private.npz',values=values,errors=errors,p0=p0,p0errors=p0errors)
    terms=decompose(values,p0,reused['values'][:,:,0,3]);rows,cells=summaries(terms,domains)
    scores=[];domain_errors=[];content_rows=[];error_cells=[];position_rows=[]
    for a,arm in enumerate(ARMS):
        for ci,cell in enumerate(('ORIGINAL','PAIRED_STYLE')):
            for k,output in enumerate(('PRE','K')):
                v=values[:,:,a,ci,k];ev=errors[:,:,a,ci,k]
                scores.append(dict(arm=arm,cell=cell,output=output,macro_Dice_percent=float(np.mean([v[domains==d].mean() for d in names])),
                    imageweighted_Dice_percent=float(v.mean()),OD=float(np.mean([v[domains==d,:,0].mean() for d in names])),OC=float(np.mean([v[domains==d,:,1].mean() for d in names]))))
                for d in names:
                    for ch,channel in enumerate(('OD','OC')):
                        domain_errors.append(dict(arm=arm,cell=cell,output=output,domain='D'+str(names.index(d)),channel=channel,
                            Dice_percent=float(v[domains==d,:,ch].mean()),FP_pixel_fraction=float(ev[domains==d,:,ch,0].mean()),
                            FN_pixel_fraction=float(ev[domains==d,:,ch,1].mean()),foreground_fraction=float(ev[domains==d,:,ch,2].mean())))
    order1=read(root/'scorer/screen_o1.json');positions=[{r['image_sha256']:i for i,r in enumerate(rr)} for rr in (panel,order1)]
    for name,delta in terms.items():
        for d in names:
            for ch,channel in enumerate(('OD','OC')):
                v=delta[domains==d,:,ch].mean(1)
                content_rows.append(dict(contrast=name,domain='D'+str(names.index(d)),channel=channel,n_contents=len(v),mean_delta_pp=float(v.mean()),
                    median_delta_pp=float(np.median(v)),positive_contents=int((v>0).sum()),negative_contents=int((v<0).sum())))
                early=[];late=[];switch=0
                for i in np.flatnonzero(domains==d):
                    identity=panel[i]['image_sha256'];e,l=sorted((0,1),key=lambda o:positions[o][identity])
                    if positions[e][identity]==positions[l][identity]:continue
                    early.append(float(delta[i,e::2,ch].mean()));late.append(float(delta[i,l::2,ch].mean()))
                    switch+=positions[e][identity]<96<=positions[l][identity]
                position_rows.append(dict(contrast=name,domain='D'+str(names.index(d)),channel=channel,n_paired_contents=len(early),
                    n_switching_halves=switch,earlier_delta_pp=float(np.mean(early)),later_delta_pp=float(np.mean(late)),later_minus_earlier_pp=float(np.mean(np.array(late)-early))))
    for k,label in enumerate(('FP','FN','FOREGROUND')):
        et=decompose(errors[...,k]*100,p0errors[...,k]*100,reused['errors'][:,:,0,3,...,k]*100)
        for name,delta in et.items():
            for d in names:
                for j in range(6):
                    for ch,channel in enumerate(('OD','OC')):
                        error_cells.append(dict(contrast=name,metric=label,domain='D'+str(names.index(d)),seed=SEEDS[j//2],order=j%2,
                            channel=channel,delta_pixel_pp=float(delta[domains==d,j,ch].mean())))
    out=root/'public'
    for filename,table in (('CONTRASTS',rows),('ALL_CELLS',cells),('NEGATIVE_CELLS',[r for r in cells if r['delta_pp']<0]),
        ('ARM_SCORES',scores),('DOMAIN_CHANNEL_ERRORS',domain_errors),('CONTENT_DISTRIBUTIONS',content_rows),
        ('MECHANISM',mechanisms),('ERROR_COMPONENT_CELLS',error_cells),('POSITION_DECOMPOSITION',position_rows)):
        csvout(out/(filename+'.csv'),[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in table])
    target=next(x for x in rows if x['contrast']=='VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL')
    candidate=terms['VERTEX_PAIRED_STYLE-NATIVE_ORIGINAL/TOTAL'];focal=terms['VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL']
    domain_harm=all(float(candidate[domains==d,:,ch].mean())>=-.5 for d in names for ch in (0,1))
    order_harm=all(float(candidate[domains==d,o::2,ch].mean())>=-2 for d in names for o in (0,1) for ch in (0,1))
    success=target['macro_delta_pp']>=.1 and target['ci95_pp'][0]>0 and float(focal[domains==names[1],:,1].mean())>=-.2 and domain_harm and order_harm
    save(out/'PERFORMANCE_GATE.json',dict(status='PASS' if success else 'NOT_PASSED',candidate='VERTEX_PAIRED_STYLE',macro_vs_original=target['macro_delta_pp'],macro_ci95=target['ci95_pp'],domain_harm_pass=domain_harm,order_harm_pass=order_harm,automatic_followup=False))
    save(out/'RESULTS.json',dict(status='COMPLETE',contents=192,new_continuous_trajectories=6,reused_endpoint_trajectories=12,
        new_physical_updates=1152,new_prediction_units=2304,reused_PRE_K_prediction_units=4608,
        contrasts=rows,paired_identity_verified=True,history_current_identity_verified=True,
        bootstrap_draws=2000,independent_validation=False,automatic_followup=False,cpu_body_seconds=time.time()-start))
    save(out/'COMPLETION.json',dict(all_GPU_children_retired=True,online_label_reads=0,source_image_reads=0,
        new_physical_updates=1152,new_continuous_trajectories=6,reused_endpoint_trajectories=12,new_prediction_units=2304,automatic_followup=False))
    return 0
