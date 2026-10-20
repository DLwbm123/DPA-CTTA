"""CPU-only final scoring, paired development inference, fixed multiplicity and promotion."""
import json,os,time
from pathlib import Path
import numpy as np
from .method import SEEDS,CELLS
from .run import jobs,WIDTH
from .process import alive
from ..r51_same_state.run import read,save
from ..r79_mechanism_control.score import csvout,summaries


def decompose(v,p):
    h=v[:,:,:,0]-p[:,:,None];current=v[:,:,:,1]-v[:,:,:,0];total=h+current
    assert np.allclose(total,v[:,:,:,1]-p[:,:,None],atol=1e-10)
    out={}
    for name,x in (('HISTORY',h),('CURRENT',current),('TOTAL',total)):
        for i,cell in enumerate(CELLS):out[cell+'/'+name]=x[:,:,i]
        for i,cell in enumerate(CELLS[1:],1):out[cell+'-VV/'+name]=x[:,:,i]-x[:,:,0]
        out['FACTORIAL/'+name]=x[:,:,0]-x[:,:,1]-x[:,:,2]+x[:,:,3]
    return out


def interval(delta,domains,alpha=.05):
    rng=np.random.default_rng(20260907);ids=[np.flatnonzero(domains==d) for d in sorted(set(domains))];draws=[]
    for _ in range(2000):draws.append(np.mean([delta[rng.choice(i,len(i),replace=True)].mean() for i in ids]))
    return np.quantile(draws,(alpha/2,1-alpha/2)).tolist()


def self_check():
    v=np.zeros((8,6,4,2,2));p=np.zeros((8,6,2));v[:,:,1,0]=2;v[:,:,1,1]=5
    t=decompose(v,p);assert np.all(t['VB/HISTORY']==2) and np.all(t['VB/CURRENT']==3)
    assert np.all(t['FACTORIAL/TOTAL']==-5)
    print('LOSS_FACTORIAL_SCORE_CPU_PASS',flush=True)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer only')
    root=Path(c['output_root']);start=time.time();out=root/'public'
    states=[]
    for name in ('qualify_worker0','source_worker0','replay_worker0','replay_worker1','candidate_worker0','candidate_worker1'):
        p=read(root/'processes'/f'{name}.json');s=read(root/'private'/name/'RUN_STATE.json');states.append(s)
        if p.get('active') or p.get('exit_code')!=0 or alive(p) or s['status']!='GPU_COMPLETE':raise ValueError('actual all GPU retirement barrier')
    expected=dict(physical_updates=3616,completed_physical_updates=3616,physical_forwards=0,diagnostic_backwards=400,training_backwards=3616)
    if sum(s['physical_updates'] for s in states)!=3616 or sum(s['diagnostic_backwards'] for s in states)!=400 or sum(s['physical_forwards'] for s in states)!=36204:raise ValueError('total physical coverage')
    panel=read(root/'scorer/screen_o0.json');index={r['image_sha256']:i for i,r in enumerate(panel)}
    old=Path(c['reference_root']);prior_panel=read(old/'scorer/screen_o0.json')
    if len(index)!=192 or list(index)!=[r['image_sha256'] for r in prior_panel]:raise ValueError('exact existing content pairing')
    domains=np.array([r['domain'] for r in panel]);names=sorted(set(domains));reused=np.load(old/'scorer/values.private.npz')
    values=np.full((192,6,4,2,2),np.nan);errors=np.full((*values.shape,3),np.nan)
    for ci,mi in ((0,1),(3,2)):
        values[:,:,ci]=reused['values'][:,:,mi,3];errors[:,:,ci]=reused['errors'][:,:,mi,3]
    p0=reused['p0'][:,:,1];p0errors=reused['p0errors'][:,:,1];native=reused['values'][:,:,0,3];nativeerrors=reused['errors'][:,:,0,3]
    reader=TargetReader(c['target_root'],256*1024**2,'mask');labels={r['image_sha256']:reader.read(r) for r in panel};mechanisms=[];probe_rows=[];probe_scores=[]
    for slot in (0,1):
        rows=read(root/'scorer'/f'screen_o{slot}.json')
        for stage in ('replay','candidate'):
            for job in jobs(stage,slot):
                folder=root/'jobs'/job['id'];receipt=read(folder/'COMPLETE.json');ci=CELLS.index(job['cell']);j=SEEDS.index(job['seed'])*2+slot
                if receipt['arrivals']!=192:raise ValueError('job arrival coverage')
                with (folder/'predictions.bits').open('rb') as f:
                    for visit,row in enumerate(rows):
                        raw=f.read(2*WIDTH)
                        if len(raw)!=2*WIDTH:raise ValueError('prediction byte coverage')
                        masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,2,512,512).astype(bool)
                        i=index[row['image_sha256']];label=labels[row['image_sha256']];truth=label.numpy()[0].astype(bool)
                        for k in (0,1):
                            scored=[100*m['dice'] for m in hard_metrics(masks[k],label)]
                            if stage=='replay' and not np.allclose(values[i,j,ci,k],scored,rtol=0,atol=1e-10):raise ValueError('reused endpoint score parity')
                            values[i,j,ci,k]=scored
                            for ch in (0,1):errors[i,j,ci,k,ch]=[(masks[k,ch]&~truth[ch]).mean(),(~masks[k,ch]&truth[ch]).mean(),masks[k,ch].mean()]
                        pf=folder/f'probe_{visit}.bits'
                        if pf.exists():
                            pm=np.unpackbits(np.frombuffer(pf.read_bytes(),dtype=np.uint8)).reshape(4,2,512,512).astype(bool)
                            for cell,mask in zip(CELLS,pm):
                                for ch,m in enumerate(hard_metrics(mask,label)):
                                    probe_scores.append(dict(seed=job['seed'],order=slot,probe_number=sorted(int(p.stem.split('_')[1]) for p in folder.glob('probe_*.bits')).index(visit)+1,domain='D'+str(names.index(row['domain'])),cell=cell,channel=('OD','OC')[ch],K_minus_PRE_pp=100*m['dice']-values[i,j,0,0,ch]))
                    if f.read(1):raise ValueError('extra predictions')
                traces=[json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines()]
                if [r['visit'] for r in traces]!=list(range(1,193)):raise ValueError('full diagnostics')
                ds=[r['K'] for r in traces];mechanisms.append(dict(cell=job['cell'],seed=job['seed'],order=slot,mean_BN_gradient=float(np.mean([d['BN_gradient_norm'] for d in ds])),mean_BN_update=float(np.mean([d['BN_update_norm'] for d in ds])),mean_BCE=float(np.mean([d['loss_parts']['BCE'] for d in ds])),mean_LSO=float(np.mean([d['loss_parts']['LSO'] for d in ds]))))
                pn=0
                for t in traces:
                    if t['probe']:
                        pn+=1;meta=dict(seed=job['seed'],order=slot,probe_number=pn)
                        for cell,d in t['probe']['cells'].items():probe_rows.append(dict(**meta,cell=cell,**d,gradient_factorial_max_abs=t['probe']['gradient_factorial_max_abs'],OD_OC_cosine=t['probe']['OD_OC_cosines'][cell],component_norms=json.dumps(t['probe']['component_norms'])))
    if not np.isfinite(values).all() or not np.isfinite(errors).all() or len(probe_rows)!=192:raise ValueError('full factorial/probe coverage')
    reader.after_check();np.savez(root/'scorer/values.private.npz',values=values,errors=errors,p0=p0,p0errors=p0errors)
    source=np.full((192,2),np.nan)
    with (root/'jobs/source_only/predictions.bits').open('rb') as f:
        for i,row in enumerate(panel):
            raw=f.read(WIDTH)
            if len(raw)!=WIDTH:raise ValueError('source-only complete prediction coverage')
            mask=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool)
            source[i]=[100*m['dice'] for m in hard_metrics(mask,labels[row['image_sha256']])]
        if f.read(1):raise ValueError('extra source-only predictions')
    terms=decompose(values,p0)
    terms['P0-SOURCE_ONLY']=p0-source[:,None]
    for ci,cell in enumerate(CELLS):terms[cell+'-SOURCE_ONLY']=values[:,:,ci,1]-source[:,None]
    for cell,ci in zip(CELLS,range(4)):
        for phase,k in (('TOTAL',1),('HISTORY',0)):
            terms[cell+'-NATIVE/'+phase]=values[:,:,ci,k]-native[:,:,k]
        terms[cell+'-NATIVE/CURRENT']=(values[:,:,ci,1]-values[:,:,ci,0])-(native[:,:,1]-native[:,:,0])
    summary,cells=summaries(terms,domains);scores=[];errrows=[];contentrows=[];errorcells=[];positions=[];concentration=[];gates=[]
    for ci,cell in enumerate(CELLS):
        for k,output in enumerate(('PRE','K')):
            v=values[:,:,ci,k];ev=errors[:,:,ci,k]
            scores.append(dict(cell=cell,output=output,macro_Dice_percent=float(np.mean([v[domains==d].mean() for d in names])),imageweighted_Dice_percent=float(v.mean()),OD=float(np.mean([v[domains==d,:,0].mean() for d in names])),OC=float(np.mean([v[domains==d,:,1].mean() for d in names]))))
            for di,d in enumerate(names):
                for ch,channel in enumerate(('OD','OC')):errrows.append(dict(cell=cell,output=output,domain='D'+str(di),channel=channel,Dice_percent=float(v[domains==d,:,ch].mean()),FP_pixel_fraction=float(ev[domains==d,:,ch,0].mean()),FN_pixel_fraction=float(ev[domains==d,:,ch,1].mean()),foreground_fraction=float(ev[domains==d,:,ch,2].mean())))
    order1=read(root/'scorer/screen_o1.json');pos=[{r['image_sha256']:i for i,r in enumerate(a)} for a in (panel,order1)]
    for name,delta in terms.items():
        for di,d in enumerate(names):
            for ch,channel in enumerate(('OD','OC')):
                a=delta[domains==d,:,ch].mean(1);contentrows.append(dict(contrast=name,domain='D'+str(di),channel=channel,n_contents=len(a),mean_delta_pp=float(a.mean()),median_delta_pp=float(np.median(a)),positive_contents=int((a>0).sum()),negative_contents=int((a<0).sum())))
                early=[];late=[]
                for i in np.flatnonzero(domains==d):
                    identity=panel[i]['image_sha256'];e,l=sorted((0,1),key=lambda o:pos[o][identity])
                    if pos[e][identity]!=pos[l][identity]:early.append(delta[i,e::2,ch].mean());late.append(delta[i,l::2,ch].mean())
                positions.append(dict(contrast=name,domain='D'+str(di),channel=channel,n_contents=len(early),earlier_delta_pp=float(np.mean(early)),later_delta_pp=float(np.mean(late)),later_minus_earlier_pp=float(np.mean(late)-np.mean(early))))
    for k,label in enumerate(('FP','FN','FOREGROUND')):
        for name,delta in decompose(errors[...,k]*100,p0errors[...,k]*100).items():
            for di,d in enumerate(names):
                for j in range(6):
                    for ch,channel in enumerate(('OD','OC')):errorcells.append(dict(contrast=name,metric=label,domain='D'+str(di),seed=SEEDS[j//2],order=j%2,channel=channel,delta_pixel_pp=float(delta[domains==d,j,ch].mean())))
    for cell in ('VB','BV'):
        delta=terms[cell+'-VV/TOTAL'];harm=terms[cell+'-NATIVE/TOTAL'];macro=np.mean([delta[domains==d].mean((0,2)) for d in names],0);ci=interval(delta,domains,.025)
        gh=all(harm[domains==d,:,ch].mean()>=-.5 for d in names for ch in (0,1));oh=all(harm[domains==d,o::2,ch].mean()>=-2 for d in names for o in (0,1) for ch in (0,1))
        robust=all(macro[o::2].mean()>0 for o in (0,1)) and (macro>0).sum()>=5 and all(macro[s*2:s*2+2].mean()>0 for s in range(3))
        passed=macro.mean()>=.1 and ci[0]>0 and delta[domains==names[1],:,1].mean()>=-.2 and gh and oh and robust
        gates.append(dict(cell=cell,status='PASS' if passed else 'NOT_PASSED',macro_delta_pp=float(macro.mean()),familywise_ci97_5_pp=ci,legacy_ci95_pp=interval(delta,domains),D1_OC_delta_pp=float(delta[domains==names[1],:,1].mean()),domain_harm_pass=bool(gh),order_harm_pass=bool(oh),robustness_pass=bool(robust),positive_anchors=int((macro>0).sum()),anchor_macro_delta_pp=macro.tolist()))
        dmean=[float(delta[domains==d].mean()) for d in names]
        for di in range(4):concentration.append(dict(cell=cell,domain='D'+str(di),domain_macro_delta_pp=dmean[di],contribution_to_four_domain_macro_pp=dmean[di]/4,leave_this_domain_out_macro_pp=float(np.mean(dmean[:di]+dmean[di+1:]))))
    promoted=next((g['cell'] for g in gates if g['status']=='PASS'),None)
    tables={'CONTRASTS':summary,'ALL_CELLS':cells,'NEGATIVE_CELLS':[r for r in cells if r['delta_pp']<0],'ARM_SCORES':scores,'DOMAIN_CHANNEL_ERRORS':errrows,'CONTENT_DISTRIBUTIONS':contentrows,'ERROR_COMPONENT_CELLS':errorcells,'POSITION_DECOMPOSITION':positions,'GAIN_CONCENTRATION':concentration,'MECHANISM':mechanisms,'SAME_STATE_PROBES':probe_rows,'SAME_STATE_SCORES':probe_scores}
    for filename,table in tables.items():csvout(out/(filename+'.csv'),[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in table])
    csvout(out/'SOURCE_ONLY_SCORES.csv',[dict(output=label,macro_Dice_percent=float(np.mean([v[domains==d].mean() for d in names])),OD=float(np.mean([v[domains==d,...,0].mean() for d in names])),OC=float(np.mean([v[domains==d,...,1].mean() for d in names]))) for label,v in (('SOURCE_ONLY',source),('P0_CURRENT_STAT',p0))])
    save(out/'PERFORMANCE_GATE.json',dict(gates=gates,promoted=promoted,multiplicity='Bonferroni two candidate comparisons; two-sided 97.5 percent content bootstrap intervals',independent_validation=False))
    costs={key:sum(s[key] for s in states) for key in ('physical_updates','completed_physical_updates','physical_forwards','diagnostic_backwards','training_backwards')}
    save(out/'RESULTS.json',dict(status='COMPLETE',**costs,reference_replay_updates=1152,discarded_same_state_updates=144,new_candidate_updates=2304,qualification_updates=16,new_continuous_candidate_trajectories=12,replayed_reference_trajectories=6,same_state_probes=48,source_only_new_prediction_units=192,new_PRE_K_prediction_units=6912,probe_prediction_units=192,reused_endpoint_trajectories=18,reused_PRE_K_prediction_units=6912,contents=192,independent_validation=False,contrasts=summary,cpu_body_seconds=time.time()-start))
    save(out/'COMPLETION.json',dict(status='COMPLETE',all_GPU_registered_pids_absent=True,online_label_reads=0,source_image_reads=0,**costs));return 0
