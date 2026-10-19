"""Paired mechanism decomposition from frozen predictions, CPU labels only."""
import csv,json,os,time
from pathlib import Path
import numpy as np
from .method import ARMS,SEEDS
from .run import jobs,WIDTH
from ..r51_same_state.run import read,save
from ..r51_same_state.score import interval


def csvout(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def decompose(main,reset,prior):
    # content, six coupled seed/order anchors, arm, [pre,U,K], channel
    p0=np.repeat(reset[:,:,:,0,:],2,axis=1);r=np.repeat(reset[:,:,:,1,:],2,axis=1)
    if not np.allclose(p0[:,:,1:],p0[:,:,:1],atol=0,rtol=0):raise ValueError('initial no-update output differs across targets')
    C=main[:,:,:,2]-main[:,:,:,0];E=r-p0;H=main[:,:,:,0]-p0;I=C-E;total=main[:,:,:,2]-p0
    if not np.allclose(H+E+I,total,atol=1e-10):raise ValueError('decomposition identity')
    terms={}
    for a,arm in enumerate(ARMS):
        for name,v in (('CURRENT',C),('EPISODIC',E),('HISTORY',H),('INTERACTION',I),('TOTAL',total)):
            terms[arm+'/'+name]=v[:,:,a]
            if a:terms[arm+'-NATIVE/'+name]=v[:,:,a]-v[:,:,0]
        terms[arm+'/CURRENT_RNG']=main[:,:,a,1]-main[:,:,a,2]
        terms[arm+'/SCHEDULE_K_MINUS_P']=main[:,:,a,2]-prior[:,:,a]
    for a,b in ((2,1),):
        for name,v in (('CURRENT',C),('EPISODIC',E),('HISTORY',H),('INTERACTION',I),('TOTAL',total)):
            terms[ARMS[a]+'-'+ARMS[b]+'/'+name]=v[:,:,a]-v[:,:,b]
    return terms


def summaries(terms,domains):
    names=sorted(set(domains));rows=[];cells=[]
    for name,delta in terms.items():
        macro=np.mean([delta[domains==d].mean((0,2)) for d in names],0)
        oc=np.mean([delta[domains==d,:,1].mean(0) for d in names],0)
        rows.append(dict(contrast=name,macro_delta_pp=float(macro.mean()),ci95_pp=interval(delta,domains),
            OD_delta_pp=float(np.mean([delta[domains==d,:,0].mean() for d in names])),OC_delta_pp=float(oc.mean()),
            OC_ci95_pp=interval(delta[:,:,1:2],domains),OC_order_delta_pp=[float(oc[o::2].mean()) for o in (0,1)],
            OC_positive_anchors=int((oc>0).sum()),mean_absolute_content_anchor_channel_pp=float(np.abs(delta).mean()),
            p90_absolute_content_anchor_channel_pp=float(np.quantile(np.abs(delta),.9))))
        for d in names:
            for j in range(6):
                for ch,channel in enumerate(('OD','OC')):
                    cells.append(dict(contrast=name,domain='D'+str(names.index(d)),seed=SEEDS[j//2],order=j%2,channel=channel,
                        n_contents=int((domains==d).sum()),delta_pp=float(delta[domains==d,j,ch].mean())))
    return rows,cells


def self_check():
    m=np.zeros((8,6,3,3,2));r=np.zeros((8,3,3,2,2));p=np.zeros((8,6,3,2))
    m[:,:,1,0]=2;m[:,:,1,2]=5;m[:,:,1,1]=4;r[:,:,1,1]=1
    terms=decompose(m,r,p)
    assert np.all(terms['VERTEX/HISTORY']==2) and np.all(terms['VERTEX/CURRENT']==3)
    assert np.all(terms['VERTEX/EPISODIC']==1) and np.all(terms['VERTEX/INTERACTION']==2)
    assert np.all(terms['VERTEX/CURRENT_RNG']==-1) and np.all(terms['VERTEX/TOTAL']==5)
    print('MECHANISM_DECOMPOSITION_SELF_CHECK_PASS',flush=True)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only label scorer')
    root=Path(c['output_root']);start=time.time()
    for name in ('qualify_worker0','main_worker0','main_worker1','reset_worker0','reset_worker1'):
        p=read(root/'processes'/f'{name}.json');s=read(root/'private'/name/'RUN_STATE.json')
        if p['active'] or p['exit_code']!=0 or s['status']!='GPU_COMPLETE':raise ValueError('all GPU retirement barrier')
    panel=read(root/'scorer/screen_o0.json');index={r['image_sha256']:i for i,r in enumerate(panel)}
    if len(index)!=192:raise ValueError('paired unique contents')
    domains=np.array([r['domain'] for r in panel]);names=sorted(set(domains))
    reader=TargetReader(c['target_root'],256*1024**2,'mask');labels={r['image_sha256']:reader.read(r) for r in panel}
    main=np.full((192,6,3,3,2),np.nan);reset=np.full((192,3,3,2,2),np.nan)
    main_errors=np.full((*main.shape,3),np.nan);reset_errors=np.full((*reset.shape,3),np.nan)
    mechanisms=[];updates=0
    for stage in ('main','reset'):
        for slot in (0,1):
            for job in jobs(stage,slot):
                folder=root/'jobs'/job['id'];receipt=read(folder/'COMPLETE.json');updates+=receipt['physical_updates']
                rows=read(root/'scorer'/f'screen_o{job["order"]}.json') if stage=='main' else panel[slot::2]
                count=3 if stage=='main' else 2;a=ARMS.index(job['arm']);j=SEEDS.index(job['seed'])
                if receipt['arrivals']!=len(rows):raise ValueError('incomplete job')
                with (folder/'predictions.bits').open('rb') as f:
                    for row in rows:
                        raw=f.read(count*WIDTH)
                        if len(raw)!=count*WIDTH:raise ValueError('truncated masks')
                        masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(count,2,512,512).astype(bool)
                        i=index[row['image_sha256']];label=labels[row['image_sha256']];truth=label.numpy()[0].astype(bool)
                        for k in range(count):
                            scores=[100*v['dice'] for v in hard_metrics(masks[k],label)]
                            dest=main if stage=='main' else reset;errs=main_errors if stage=='main' else reset_errors
                            anchor=j*2+job['order'] if stage=='main' else j
                            dest[i,anchor,a,k]=scores
                            for ch in (0,1):errs[i,anchor,a,k,ch]=[(masks[k,ch]&~truth[ch]).mean(),(~masks[k,ch]&truth[ch]).mean(),masks[k,ch].mean()]
                    if f.read(1):raise ValueError('extra masks')
                traces=[json.loads(v) for v in (folder/'diagnostics.jsonl').read_text().splitlines()]
                if [t['visit'] for t in traces]!=list(range(1,len(rows)+1)):raise ValueError('trace coverage')
                for branch in ('K','U') if stage=='main' else ('R',):
                    ds=[t[branch] for t in traces]
                    mechanisms.append(dict(stage=stage,arm=job['arm'],seed=job['seed'],order=job.get('order','NONE'),part=job.get('part','ALL'),branch=branch,
                        arrivals=len(rows),mean_BN_gradient=float(np.mean([d['BN_gradient_norm'] for d in ds])),
                        mean_BN_update=float(np.mean([d['BN_update_norm'] for d in ds])),
                        mean_BCE=float(np.mean([d['loss_parts']['BCE'] for d in ds])),mean_LSO=float(np.mean([d['loss_parts']['LSO'] for d in ds]))))
    if updates!=8640 or not np.isfinite(main).all() or not np.isfinite(reset).all():raise ValueError('complete 8640 update coverage')
    reader.after_check();np.savez(root/'scorer/values.private.npz',main=main,reset=reset,main_errors=main_errors,reset_errors=reset_errors)
    old=Path(c['reference_root']);prior_panel=read(old/'scorer/screen_o0.json')
    if [r['image_sha256'] for r in prior_panel]!=[r['image_sha256'] for r in panel]:raise ValueError('prior result pairing')
    prior=np.load(old/'scorer/values.private.npy')[:,:,(0,2,4),:]
    terms=decompose(main,reset,prior);rows,cells=summaries(terms,domains)
    scores=[];errors=[];rng=[];content_rows=[]
    for a,arm in enumerate(ARMS):
        for label,values,ev in [('PRE',main[:,:,a,0],main_errors[:,:,a,0]),('U',main[:,:,a,1],main_errors[:,:,a,1]),
            ('K',main[:,:,a,2],main_errors[:,:,a,2]),('P0',np.repeat(reset[:,:,a,0],2,axis=1),np.repeat(reset_errors[:,:,a,0],2,axis=1)),
            ('R',np.repeat(reset[:,:,a,1],2,axis=1),np.repeat(reset_errors[:,:,a,1],2,axis=1))]:
            scores.append(dict(arm=arm,output=label,macro_Dice_percent=float(np.mean([values[domains==d].mean() for d in names])),
                OD=float(np.mean([values[domains==d,:,0].mean() for d in names])),OC=float(np.mean([values[domains==d,:,1].mean() for d in names]))))
            for d in names:
                for ch,channel in enumerate(('OD','OC')):
                    errors.append(dict(arm=arm,output=label,domain='D'+str(names.index(d)),channel=channel,
                        Dice_percent=float(values[domains==d,:,ch].mean()),FP_pixel_fraction=float(ev[domains==d,:,ch,0].mean()),
                        FN_pixel_fraction=float(ev[domains==d,:,ch,1].mean()),foreground_fraction=float(ev[domains==d,:,ch,2].mean())))
        current=main[:,:,a,2]-main[:,:,a,0];alternate=main[:,:,a,1]-main[:,:,a,0]
        for d in names:
            for ch,channel in enumerate(('OD','OC')):
                x=current[domains==d,:,ch];y=alternate[domains==d,:,ch]
                rng.append(dict(arm=arm,domain='D'+str(names.index(d)),channel=channel,
                    mean_K_current_pp=float(x.mean()),mean_U_current_pp=float(y.mean()),
                    paired_content_anchor_sign_disagreements=int(((x>0)&(y<0)|(x<0)&(y>0)).sum()),
                    paired_content_anchor_count=int(x.size),mean_absolute_U_minus_K_pp=float(np.abs(y-x).mean()),
                    p90_absolute_U_minus_K_pp=float(np.quantile(np.abs(y-x),.9))))
    for name,delta in terms.items():
        for d in names:
            for ch,channel in enumerate(('OD','OC')):
                v=delta[domains==d,:,ch].mean(1)
                content_rows.append(dict(contrast=name,domain='D'+str(names.index(d)),channel=channel,n_contents=len(v),
                    mean_delta_pp=float(v.mean()),median_delta_pp=float(np.median(v)),positive_contents=int((v>0).sum()),negative_contents=int((v<0).sum())))
    # Attribute pixel errors as well as Dice, without pretending mean pixel errors reconstruct mean Dice.
    error_cells=[]
    for k,label in enumerate(('FP','FN','FOREGROUND')):
        et=decompose(main_errors[...,k]*100,reset_errors[...,k]*100,np.zeros_like(prior))
        for name,delta in et.items():
            if name.endswith('/SCHEDULE_K_MINUS_P'):continue
            for d in names:
                for j in range(6):
                    for ch,channel in enumerate(('OD','OC')):
                        error_cells.append(dict(contrast=name,metric=label,domain='D'+str(names.index(d)),seed=SEEDS[j//2],order=j%2,
                            channel=channel,delta_pixel_pp=float(delta[domains==d,j,ch].mean())))
    order1=read(root/'scorer/screen_o1.json');positions=[{r['image_sha256']:i for i,r in enumerate(rows)} for rows in (panel,order1)]
    position_rows=[]
    for name,delta in terms.items():
        for d in names:
            indices=np.flatnonzero(domains==d)
            for ch,channel in enumerate(('OD','OC')):
                early=[];late=[];switch=0
                for i in indices:
                    identity=panel[i]['image_sha256'];e,l=sorted((0,1),key=lambda o:positions[o][identity])
                    if positions[e][identity]==positions[l][identity]:continue
                    early.append(float(delta[i,e::2,ch].mean()));late.append(float(delta[i,l::2,ch].mean()))
                    switch+=positions[e][identity]<96<=positions[l][identity]
                position_rows.append(dict(contrast=name,domain='D'+str(names.index(d)),channel=channel,n_paired_contents=len(early),
                    n_switching_halves=switch,earlier_delta_pp=float(np.mean(early)),later_delta_pp=float(np.mean(late)),
                    later_minus_earlier_pp=float(np.mean(np.array(late)-early))))
    out=root/'public'
    for filename,table in (('CONTRASTS',rows),('ALL_CELLS',cells),('NEGATIVE_CELLS',[r for r in cells if r['delta_pp']<0]),
        ('ARM_SCORES',scores),('DOMAIN_CHANNEL_ERRORS',errors),('RNG_SENSITIVITY',rng),('CONTENT_DISTRIBUTIONS',content_rows),
        ('MECHANISM',mechanisms),('ERROR_COMPONENT_CELLS',error_cells),('POSITION_DECOMPOSITION',position_rows)):
        # Nested intervals remain exact JSON strings within CSV columns.
        csvout(out/(filename+'.csv'),[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in table])
    save(out/'RESULTS.json',dict(status='COMPLETE',contents=192,continuous_trajectories=18,main_committed_updates=3456,
        current_rng_diagnostic_updates=3456,reset_diagnostic_updates=1728,physical_updates=8640,prediction_units=13824,
        contrasts=rows,algebraic_identity_verified=True,initial_reference_equal_across_targets=True,
        bootstrap_draws=2000,independent_validation=False,automatic_followup=False,cpu_body_seconds=time.time()-start))
    save(out/'COMPLETION.json',dict(all_GPU_children_retired=True,online_label_reads=0,source_image_reads=0,
        physical_updates=8640,continuous_trajectories=18,reset_partition_jobs=18,prediction_units=13824,automatic_followup=False))
    return 0
