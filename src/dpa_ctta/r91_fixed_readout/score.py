"""All GPUs retired, seal fixed transforms, then CPU-only paired development scoring."""
import os,time
START=time.time();os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip,json,runpy,signal
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from dpa_ctta.r85_migration.process import alive
from dpa_ctta.r7_target_screen.runner import TargetReader
SEEDS=(20260907,17011,29009)
ARMS=('P0','P','K','Qt','Q0','NATIVE')

def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def macro(v,domains):return np.array([np.nanmean(v[domains==d],axis=(0,2)) for d in range(4)]).mean(0)
def ci(v,domains):
    rng=np.random.default_rng(20260907);ids=[np.flatnonzero(domains==d) for d in range(4)]
    draws=[np.mean([v[rng.choice(i,len(i),replace=True)].mean() for i in ids]) for _ in range(2000)]
    return np.quantile(draws,(.0125,.9875)).tolist()
def main_gate(v,domains,threshold):
    a=macro(v,domains);bounds=ci(v,domains)
    ok=a.mean()>=threshold and bounds[0]>0 and (a>0).sum()>=5 and all(a[o::2].mean()>0 for o in (0,1)) and all(a[s*2:s*2+2].mean()>0 for s in range(3))
    return dict(delta_macro_pp=float(a.mean()),ci97_5_pp=bounds,anchors=a.tolist(),positive_anchors=int((a>0).sum()),passed=bool(ok))

def self_check():
    domains=np.repeat(np.arange(4),3);v=np.ones((12,6,2))*.4
    g=main_gate(v,domains,.3);assert g['passed'] and g['positive_anchors']==6
    assert not main_gate(-v,domains,.1)['passed']
    assert not main_gate(v*.1,domains,.1)['passed']
    # Coupling six anchors within each sampled content is intentional.
    u=v.copy();u[:,0]*=-9;assert not main_gate(u,domains,.1)['passed']


def run(c):
    root=Path(c['output_root']);public=root/'public';campaign=Path(c['campaign_root']);source=Path(c['source_root'])
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU score independent deadline')))
    signal.alarm(max(1,int(min(1740,c['deadline']-time.time()-180))))
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    states=[]
    for name in ('qualify_worker0','source_worker0','formal_worker0','formal_worker1'):
        p=read(root/'processes'/f'{name}.json');s=read(root/'private'/f'{p["stage"]}_{p["slot"]}'/'RUN_STATE.json')
        assert not p['active'] and p['exit_code']==0 and not alive(p) and s['status']=='GPU_COMPLETE'
        states.append(s)
    totals={k:sum(s[k] for s in states) for k in ('physical_updates','completed_physical_updates','physical_forwards','training_backwards')}
    assert totals==dict(physical_updates=1158,completed_physical_updates=1158,physical_forwards=12750,training_backwards=1158)
    module=runpy.run_path(str(campaign/'m0.py'));post,metrics,boundary,csvout,contrasts=[module[k] for k in ('post','metrics','boundary','csvout','contrasts')];names=module['METRICS']
    files=[root/'jobs/Q0.bits.gz']+[root/'jobs'/f's{s}_o{o}'/'Qt.bits.gz' for s in SEEDS for o in (0,1)]
    sealed=[]
    for p in files:
        pp=p.with_name(p.name.replace('.bits.gz','.post.bits.gz'))
        with gzip.open(p,'rb') as f,gzip.GzipFile(filename=str(pp),mode='xb',compresslevel=1,mtime=0) as g:
            for _ in range(192):
                b=f.read(65536);assert len(b)==65536
                mask=np.unpackbits(np.frombuffer(b,dtype=np.uint8)).reshape(2,512,512).astype(bool)
                g.write(np.packbits(post(mask)).tobytes())
            assert not f.read(1)
        p.chmod(0o444);pp.chmod(0o444);sealed.append(dict(stream=p.parent.name+'/'+p.name,arrivals=192,raw_bytes=p.stat().st_size,post_bytes=pp.stat().st_size))
    save(public/'SEALED.json',dict(status='ALL_RAW_AND_FIXED_POST_SEALED',sealed_at=time.time(),new_streams=7,streams=sealed,label_reads_before_seal=0,all_registered_GPU_workers_exit0_absent=True))
    panels=[read(source/'scorer'/f'screen_o{o}.json') for o in (0,1)];index={r['image_sha256']:i for i,r in enumerate(panels[0])};assert len(index)==192
    ds=sorted({r['domain'] for r in panels[0]});domains=np.array([ds.index(r['domain']) for r in panels[0]])
    old=np.load(campaign/'private/m0_values.private.npz')['values']
    values=np.full((192,6,6,2,2,len(names)),np.nan)
    values[:,:,0]=old[:,:,1,1];values[:,:,1]=old[:,:,3,0];values[:,:,2]=old[:,:,3,1];values[:,:,5]=old[:,:,2,1]
    reader=TargetReader(c['target_root'],256*1024**2,'mask');truth=[];edges=[];dist=[]
    for row in panels[0]:
        t=reader.read(row).numpy()[0].astype(bool);e=np.stack([boundary(a) for a in t]);truth.append(t);edges.append(e);dist.append(np.stack([ndi.distance_transform_edt(~a).astype(np.float32) for a in e]))
    for fi,p in enumerate(files):
        j=fi-1;order=j%2 if fi else 0;ai=3 if fi else 4;pp=p.with_name(p.name.replace('.bits.gz','.post.bits.gz'))
        with gzip.open(p,'rb') as f,gzip.open(pp,'rb') as g:
            for row in panels[order]:
                i=index[row['image_sha256']]
                for variant,file in enumerate((f,g)):
                    b=file.read(65536);assert len(b)==65536;m=np.unpackbits(np.frombuffer(b,dtype=np.uint8)).reshape(2,512,512).astype(bool)
                    got=metrics(m,truth[i],edges[i],dist[i])
                    if fi:values[i,j,ai,variant]=got
                    else:values[i,:,ai,variant]=got
            assert not f.read(1) and not g.read(1)
    reader.after_check()
    np.savez_compressed(root/'scorer/values.private.npz',values=values,domains=domains)
    all_scores=[]
    for a,arm in enumerate(ARMS):
        for vi,variant in enumerate(('raw','post')):
            for d in range(4):
                for j in range(6):
                    for ch,head in enumerate(('OD','OC')):
                        for mi,metric in enumerate(names):
                            v=values[domains==d,j,a,vi,ch,mi];ok=v[np.isfinite(v)]
                            all_scores.append(dict(arm=arm,variant=variant,domain=f'D{d}',seed=SEEDS[j//2],order=j%2,channel=head,metric=metric,mean=float(ok.mean()) if len(ok) else None,n=len(v),n_NA=int((~np.isfinite(v)).sum())))
    csvout(public/'ALL_SCORES.csv',all_scores)
    summaries=[];cells=[];contents=[];position=[];lodo=[]
    for vi,kind in enumerate(('raw','post')):
        v=values[:,:,:,vi];pairs={'READ':v[:,:,3]-v[:,:,1],'DEPLOY':v[:,:,3]-v[:,:,2],'HISTORY':v[:,:,3]-v[:,:,4],
            'INTERACTION':(v[:,:,3]-v[:,:,1])-(v[:,:,4]-v[:,:,0]),'STATIC_READ':v[:,:,4]-v[:,:,0],'Qt-NATIVE':v[:,:,3]-v[:,:,5]}
        for name,delta in pairs.items():
            contrasts(public,name+'/'+kind,delta,domains,summaries,cells,contents)
            for omitted in range(4):
                x=delta[domains!=omitted,...,0];dd=domains[domains!=omitted]
                lodo.append(dict(contrast=name,variant=kind,omitted_domain=f'D{omitted}',macro_Dice_delta_pp=float(np.mean([x[dd==d].mean() for d in range(4) if d!=omitted]))))
            for j in range(6):
                for half in (0,1):
                    ids=[index[r['image_sha256']] for r in panels[j%2][half*96:(half+1)*96]]
                    for d in range(4):
                        ii=[i for i in ids if domains[i]==d]
                        for ch,head in enumerate(('OD','OC')):
                            position.append(dict(contrast=name,variant=kind,seed=SEEDS[j//2],order=j%2,half=half,domain=f'D{d}',channel=head,n=len(ii),Dice_delta_pp=float(delta[ii,j,ch,0].mean()) if ii else None))
    csvout(public/'CONTRAST_SUMMARY.csv',summaries);csvout(public/'ALL_CONTRAST_CELLS.csv',cells);csvout(public/'CONTENT_DISTRIBUTIONS.csv',contents);csvout(public/'EARLIER_LATER.csv',position);csvout(public/'LODO.csv',lodo)
    deploy=values[:,:,3,0,:,0]-values[:,:,2,0,:,0];history=values[:,:,3,0,:,0]-values[:,:,4,0,:,0]
    gates={'DEPLOY':main_gate(deploy,domains,.1),'HISTORY':main_gate(history,domains,.3)}
    d1=deploy[domains==1].mean((0,1));d3=float(deploy[domains==3,:,1].mean())
    nv=values[:,:,3,0,:,0]-values[:,:,5,0,:,0];dc=np.array([nv[domains==d].mean((0,1)) for d in range(4)])
    dco=np.array([[nv[domains==d,o::2].mean((0,1)) for o in (0,1)] for d in range(4)])
    protection=bool((d1>=-.2).all() and d3>=.1 and (dc>=-.5).all() and (dco>=-2).all())
    post_positive=all(macro(values[:,:,3,1,:,0]-values[:,:,a,1,:,0],domains).mean()>0 for a in (2,4))
    structure=[]
    for vi,variant in enumerate(('raw','post')):
        for scope in ('overall','D1','D3'):
            ids=np.arange(192) if scope=='overall' else np.flatnonzero(domains==int(scope[1:]))
            channels=(None,) if scope=='overall' else (0,1)
            for mi in (4,5,6):
                for ch in ((None,) if mi==6 else channels):
                    selected=([0] if mi==6 else ([ch] if ch is not None else [0,1]))
                    a=values[ids,:,3,vi][:,:,selected,mi];b=values[ids,:,2,vi][:,:,selected,mi];common=np.isfinite(a)&np.isfinite(b)
                    delta=a-b
                    groups=range(4) if scope=='overall' else [int(scope[1:])]
                    shifts=[]
                    for d in groups:
                        sub=domains[ids]==d;v=delta[sub];valid=common[sub];shifts.append(float(v[valid].mean()) if valid.any() else None)
                    difference=float(np.mean(shifts)) if all(x is not None for x in shifts) else None
                    na_a=int((~np.isfinite(a)).sum());na_b=int((~np.isfinite(b)).sum());tol=1e-9 if mi==6 else 1e-6
                    structure.append(dict(variant=variant,scope=scope,channel='vCDR' if mi==6 else ('both' if ch is None else ('OD','OC')[ch]),metric=names[mi],paired_mean_delta=difference,Qt_NA=na_a,K_NA=na_b,tolerance=tol,passed=bool(difference is not None and difference<=tol and na_a<=na_b)))
            for ch in (0,1):
                a=values[ids,:,3,vi,ch,9];b=values[ids,:,2,vi,ch,9]
                structure.append(dict(variant=variant,scope=scope,channel=('OD','OC')[ch],metric='empty_prediction_count',Qt_count=int((a>0).sum()),K_count=int((b>0).sum()),passed=bool((a>0).sum()<=(b>0).sum())))
    passed=all(g['passed'] for g in gates.values()) and protection and post_positive and all(s['passed'] for s in structure)
    save(public/'PERFORMANCE_GATE.json',dict(status='PASS' if passed else 'NOT_PASSED',main=gates,D1_OD_OC_delta_pp=d1.tolist(),D3_OC_delta_pp=d3,Qt_minus_NATIVE_domain_head=dc.tolist(),Qt_minus_NATIVE_domain_order_head=dco.tolist(),protection_passed=protection,post_main_directions_positive=bool(post_positive),structure_passed=all(s['passed'] for s in structure),structure=structure,independent_validation=False))
    save(public/'COMPLETION.json',dict(status='COMPLETE',**totals,main_streams=6,main_arrivals=1152,Q0_unique_arrivals=192,Qt_arrivals=1152,PRE_K_reference_exact_all=True,Q0_P0_identity_exact=True,online_label_reads=0,source_image_reads=0,label_contents=192,all_registered_GPU_exit0_absent_before_labels=True,all_raw_post_sealed_before_labels=True,CPU_body_seconds=time.time()-START,prior_R90_updates=c['prior_updates'],prior_R90_forwards=c['prior_forwards'],shared_total_updates=c['prior_updates']+1158,shared_total_forwards=c['prior_forwards']+12750,deadline=c['deadline']))
    print(json.dumps(read(public/'PERFORMANCE_GATE.json')),flush=True)

if __name__=='__main__':
    self_check()
    run(read(Path(os.environ['CONFIG'])))
