"""Sealed CPU scoring; one primary development contrast, no winner selection."""
import csv,gzip,json,os,runpy,time
from pathlib import Path
import numpy as np
SEEDS=(20260907,17011,29009)
ARMS=('E_C2','E_VV','C_C2','C_VV')
def read(p):return json.loads(Path(p).read_text())
def macro(x,domains):return np.array([x[domains==d].mean(axis=(0,2)) for d in range(4)]).mean(0)
def interval(x,domains):
    # Same sampled contents jointly carry all six anchors and both heads.
    rng=np.random.default_rng(20260907);ids=[np.flatnonzero(domains==d) for d in range(4)]
    v=x.mean(axis=(1,2))
    draws=np.mean([v[rng.choice(i,(2000,len(i)),replace=True)].mean(1) for i in ids],axis=0)
    return np.quantile(draws,(.025,.975)).tolist()
def assess(x,domains):
    assert np.isfinite(x).all();a=macro(x,domains);ci=interval(x,domains)
    stable=int((a>0).sum())>=5 and all(a[o::2].mean()>0 for o in (0,1)) and all(a[s*2:s*2+2].mean()>0 for s in range(3))
    return dict(mean_pp=float(a.mean()),CI95_pp=ci,anchors_pp=a.tolist(),positive_anchors=int((a>0).sum()),stability_pass=bool(stable),performance_pass=bool(a.mean()>=.2 and ci[0]>0 and stable),below_minimum_supported=bool(ci[1]<.2))
def self_check():
    ds=np.repeat(np.arange(4),3);x=np.full((12,6,2),.3)
    assert assess(x,ds)['performance_pass']
    assert not assess(x*.5,ds)['performance_pass'] and assess(x*.5,ds)['below_minimum_supported']
    assert not assess(-x,ds)['performance_pass']
    x[:,0]=-3;assert not assess(x,ds)['performance_pass']
def run(c,write):
    from scipy import ndimage as ndi
    from dpa_ctta.r7_target_screen.runner import TargetReader
    start=time.time();root=Path(c['output_root']);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    assert read(root/'public/RAW_SEALED.json')['jobs']==12
    assert read(root/'public/FORMAL_WRAPPER.json')['exit_code']==0
    pr=read(root/'private/formal_process.json');p=Path('/proc')/str(pr['pid'])/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[19]!=pr['start_ticks']
    self_check();module=runpy.run_path('/tmp/n90/m0.py');post,metrics,boundary=[module[k] for k in ('post','metrics','boundary')];names=module['METRICS'];module['self_check']()
    def guard():
        if time.time()>c['deadline']-120:raise TimeoutError('score deadline')
    jobs=[(s,o,a,root/'jobs'/f'{a}_s{s}_o{o}') for s in SEEDS for o in (0,1) for a in ('E_C2','E_VV')]
    for s,o,a,f in jobs:
        guard();assert read(f/'COMPLETE.json')['arrivals']==192
        assert (f/'K.bits.gz').exists()
        with gzip.open(f/'K.bits.gz','rb') as src,gzip.GzipFile(filename=str(f/'K.post.bits.gz'),mode='xb',compresslevel=1,mtime=0) as dst:
            for i in range(192):
                guard();mask=np.unpackbits(np.frombuffer(src.read(65536),dtype=np.uint8)).reshape(2,512,512).astype(bool);dst.write(np.packbits(post(mask)).tobytes())
            assert not src.read(1)
        (f/'K.bits.gz').chmod(0o444);(f/'K.post.bits.gz').chmod(0o444)
    write('public/SEALED.json',dict(status='ALL_RAW_POST_SEALED',streams=12,labels_read_before_seal=0,at=time.time()))
    panels=[read(root/'scorer'/f'screen_o{o}.json') for o in (0,1)];idx={r['image_sha256']:i for i,r in enumerate(panels[0])};ds=sorted({r['domain'] for r in panels[0]})
    domains=np.array([ds.index(r['domain']) for r in panels[0]])
    with np.load('/tmp/n91/scorer/values.private.npz',allow_pickle=False) as f:old=f['values'];assert np.array_equal(domains,f['domains'])
    assert old.shape==(192,6,6,2,2,11)
    values=np.full((192,6,4,2,2,11),np.nan);values[:,:,3]=old[:,:,2]
    with np.load('/tmp/n102/scorer/values.private.npz',allow_pickle=False) as f:
        assert np.array_equal(domains,f['domains']);values[:,:,2]=f['values'][:,:,1]
    assert abs(float(macro(values[:,:,3,0,:,0],domains).mean())-76.822122648)<1e-6
    assert abs(float(macro(values[:,:,2,0,:,0],domains).mean())-76.724873023)<1e-6
    reader=TargetReader(c['target_root'],256*1024**2,'mask');truth=[];edges=[];dist=[]
    for row in panels[0]:
        guard();t=reader.read(row).numpy()[0].astype(bool);e=np.stack([boundary(x) for x in t]);truth.append(t);edges.append(e);dist.append(np.stack([ndi.distance_transform_edt(~x).astype(np.float32) for x in e]))
    for s,o,a,folder in jobs:
        j=SEEDS.index(s)*2+o;ai=ARMS.index(a)
        with gzip.open(folder/'K.bits.gz','rb') as f,gzip.open(folder/'K.post.bits.gz','rb') as g:
            for row in panels[o]:
                guard();i=idx[row['image_sha256']]
                for vi,src in enumerate((f,g)):
                    raw=src.read(65536);assert len(raw)==65536
                    m=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool)
                    values[i,j,ai,vi]=metrics(m,truth[i],edges[i],dist[i])
            assert not f.read(1) and not g.read(1)
        write('public/SCORE_PROGRESS.json',dict(completed_arm=a,seed=s,order=o,at=time.time()))
    reader.after_check();np.savez_compressed(root/'scorer/values.private.npz',values=values,domains=domains)
    rows=[]
    for ai,a in enumerate(ARMS):
        for vi,variant in enumerate(('raw','post')):
            for d in range(4):
                for j in range(6):
                    for h,head in enumerate(('OD','OC')):
                        for mi,name in enumerate(names):
                            v=values[domains==d,j,ai,vi,h,mi];ok=np.isfinite(v)
                            rows.append(dict(arm=a,variant=variant,domain=f'D{d}',seed=SEEDS[j//2],order=j%2,head=head,metric=name,mean=float(v[ok].mean()) if ok.any() else 'NA',n=len(v),n_NA=int((~ok).sum())))
    with (root/'public/ALL_CELLS.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    summaries={};contrast_arrays={}
    for vi,variant in enumerate(('raw','post')):
        v=values[:,:,:,vi,:,0]
        pairs={'history_C2':v[:,:,2]-v[:,:,0],'history_VV':v[:,:,3]-v[:,:,1],'interaction':(v[:,:,3]-v[:,:,2])-(v[:,:,1]-v[:,:,0])}
        for name,delta in pairs.items():
            summaries[name+'/'+variant]=assess(delta,domains);contrast_arrays[name+'/'+variant]=delta
    delta=values[:,:,2]-values[:,:,0];local=[];structure=[]
    for vi,variant in enumerate(('raw','post')):
        for d in range(4):
            for h,head in enumerate(('OD','OC')):
                v=delta[domains==d,:,vi,h,0];dm=float(v.mean());order=[float(v[:,o::2].mean()) for o in (0,1)]
                local.append(dict(variant=variant,domain=f'D{d}',head=head,delta_pp=dm,order_pp=order,passed=bool(dm>=-.5 and min(order)>=-2)))
                for mi,name in enumerate(names):
                    aa=values[domains==d,:,2,vi,h,mi];bb=values[domains==d,:,0,vi,h,mi]
                    if mi in (4,5,6,9):
                        if mi==6 and h==1:continue
                        na_a=int((~np.isfinite(aa)).sum());na_b=int((~np.isfinite(bb)).sum())
                        if mi==9:
                            va=int((aa>0).sum());vb=int((bb>0).sum());passed=va<=vb and na_a<=na_b
                        else:va=na_a;vb=na_b;passed=na_a<=na_b
                        structure.append(dict(variant=variant,scope=f'D{d}/{head}',metric=name+'_missing_or_empty',C_C2=va,E_C2=vb,passed=bool(passed)))
        for mi in (4,5,6):
            selected=[0] if mi==6 else [0,1];x=delta[:,:,vi][:,:,selected,mi];tol=1e-9 if mi==6 else 1e-6
            finite=bool(np.isfinite(x).all());mean=float(macro(x,domains).mean()) if finite else None
            structure.append(dict(variant=variant,scope='overall',metric=names[mi],delta=mean,tolerance=tol,passed=bool(finite and mean<=tol)))
    main=summaries['history_C2/raw'];post_positive=summaries['history_C2/post']['mean_pp']>0
    safe=all(x['passed'] for x in structure+local)
    passed=main['performance_pass']
    decision='HISTORY_VALUE_WITH_LOCAL_TRADEOFF' if passed and not safe else 'HISTORY_VALUE_SUPPORTED' if passed else 'BELOW_REGISTERED_INCREMENT' if main['below_minimum_supported'] else 'INSUFFICIENT_EVIDENCE'
    write('public/RESULT.json',dict(status=decision,primary=main,post_positive=post_positive,local_pass=all(x['passed'] for x in local),structure_pass=all(x['passed'] for x in structure),contrasts=summaries,local=local,structure=structure,independent_confirmation=False,threshold_pp=.2,no_automatic_followup=True))
    baseline_rows=[]
    for ai,a in enumerate(ARMS):
        baseline_rows.append(dict(arm=a,raw=float(macro(values[:,:,ai,0,:,0],domains).mean()),post=float(macro(values[:,:,ai,1,:,0],domains).mean())))
    write('public/ABSOLUTE.json',baseline_rows)
    # P0/Q0 descriptive contrasts retain the static references and known adverse cells.
    static=[]
    for ai,a in enumerate(ARMS):
        for ri,rname in ((0,'P0'),(4,'Q0')):
            for vi,variant in enumerate(('raw','post')):
                x=values[:,:,ai,vi,:,0]-old[:,:,ri,vi,:,0]
                static.append(dict(arm=a,reference=rname,variant=variant,macro_pp=float(macro(x,domains).mean()),D3_OC_pp=float(x[domains==3,:,1].mean()),without_D1_pp=float(np.mean([x[domains==d].mean() for d in (0,2,3)]))))
    write('public/STATIC_DESCRIPTIVE.json',static)
    totals={k:sum(read(root/'private'/f'{mode}_state.json')[k] for mode in ('qualify','formal')) for k in ('physical_updates','completed_updates','physical_forwards','training_backwards')}
    assert totals==dict(physical_updates=2336,completed_updates=2336,physical_forwards=21048,training_backwards=2336)
    write('public/COMPLETION.json',dict(status='COMPLETE',**totals,new_label_contents=192,scored_streams=12,all_GPU_retired_before_scoring=True,raw_post_sealed_before_labels=True,CPU_body_seconds=time.time()-start))
    return dict(status=decision,primary_gain=main['mean_pp'])
