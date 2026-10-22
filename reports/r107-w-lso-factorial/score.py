"""Sealed CPU scoring; two frozen primary contrasts and fixed candidate priority."""
import csv,gzip,json,os,runpy,time,lzma,io
from codec import PackedWriter,put,xor,unpack
from pathlib import Path
import numpy as np
SEEDS=(20260907,17011,29009)
ARMS=('W_ONLY','LSO_ONLY','C2','VV')
def read(p):return json.loads(Path(p).read_text())
def macro(x,domains):return np.array([x[domains==d].mean(axis=(0,2)) for d in range(4)]).mean(0)
def interval(x,domains,adjusted=False):
    # Same sampled contents jointly carry all six anchors and both heads.
    rng=np.random.default_rng(20260907);ids=[np.flatnonzero(domains==d) for d in range(4)]
    v=x.mean(axis=(1,2))
    draws=np.mean([v[rng.choice(i,(2000,len(i)),replace=True)].mean(1) for i in ids],axis=0)
    return np.quantile(draws,(.0125,.9875) if adjusted else (.025,.975)).tolist()
def assess(x,domains,adjusted=False):
    assert np.isfinite(x).all();a=macro(x,domains);ci=interval(x,domains,adjusted)
    stable=int((a>0).sum())>=5 and all(a[o::2].mean()>0 for o in (0,1)) and all(a[s*2:s*2+2].mean()>0 for s in range(3))
    return dict(mean_pp=float(a.mean()),CI_pp=ci,confidence=.975 if adjusted else .95,anchors_pp=a.tolist(),positive_anchors=int((a>0).sum()),stability_pass=bool(stable),performance_pass=bool(a.mean()>=.2 and ci[0]>0 and stable),below_minimum_supported=bool(ci[1]<.2))
def self_check():
    __import__('codec').self_check()
    ds=np.repeat(np.arange(4),3);x=np.full((12,6,2),.3)
    assert assess(x,ds)['performance_pass']
    assert not assess(x*.5,ds)['performance_pass'] and assess(x*.5,ds)['below_minimum_supported']
    assert not assess(-x,ds)['performance_pass']
    x[:,0]=-3;assert not assess(x,ds)['performance_pass']
    assert assess(x,ds,True)['confidence']==.975
    lo=interval(x,ds);hi=interval(x,ds,True);assert hi[0]<=lo[0] and hi[1]>=lo[1]
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
    jobs=[(s,o,a,root/'jobs'/f'{a}_s{s}_o{o}') for s in SEEDS for o in (0,1) for a in ('W_ONLY','LSO_ONLY')]
    for s,o,a,f in jobs:
        guard();assert read(f/'COMPLETE.json')['arrivals']==192
        assert (f/'K.bits.xz').exists()
        with lzma.open(f/'K.bits.xz','rb') as src,PackedWriter(f/'K.post.xor.xz',c) as dst:
            for i in range(192):
                guard();raw=src.read(65536);mask=unpack(raw);dst.write(xor(raw,np.packbits(post(mask)).tobytes()))
            assert not src.read(1)
        (f/'K.bits.xz').chmod(0o444);(f/'K.post.xor.xz').chmod(0o444)
    write('public/SEALED.json',dict(status='ALL_RAW_POST_SEALED',streams=12,labels_read_before_seal=0,at=time.time(),encoding='post_bits = raw_bits XOR decoded_post_xor; fixed post unchanged'))
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
        with lzma.open(folder/'K.bits.xz','rb') as f,lzma.open(folder/'K.post.xor.xz','rb') as g:
            for row in panels[o]:
                guard();i=idx[row['image_sha256']];raw=f.read(65536);delta=g.read(65536)
                for vi,data in enumerate((raw,xor(raw,delta))):
                    values[i,j,ai,vi]=metrics(unpack(data),truth[i],edges[i],dist[i])
            assert not f.read(1) and not g.read(1)
        write('public/SCORE_PROGRESS.json',dict(completed_arm=a,seed=s,order=o,at=time.time()))
    reader.after_check();buf=io.BytesIO();np.savez_compressed(buf,values=values,domains=domains);put(c,root/'scorer/values.private.npz',buf.getvalue())
    rows=[]
    for ai,a in enumerate(ARMS):
        for vi,variant in enumerate(('raw','post')):
            for d in range(4):
                for j in range(6):
                    for h,head in enumerate(('OD','OC')):
                        for mi,name in enumerate(names):
                            v=values[domains==d,j,ai,vi,h,mi];ok=np.isfinite(v)
                            rows.append(dict(arm=a,variant=variant,domain=f'D{d}',seed=SEEDS[j//2],order=j%2,head=head,metric=name,mean=float(v[ok].mean()) if ok.any() else 'NA',n=len(v),n_NA=int((~ok).sum())))
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);put(c,root/'public/ALL_CELLS.csv',buf.getvalue().encode())
    summaries={};functional_rows=[]
    for vi,variant in enumerate(('raw','post')):
        v=values[:,:,:,vi,:,0]
        pairs={'W_at_LSO0':v[:,:,0]-v[:,:,2], 'W_at_LSO1':v[:,:,3]-v[:,:,1],
               'LSO_at_W0':v[:,:,1]-v[:,:,2], 'LSO_at_W1':v[:,:,3]-v[:,:,0],
               'interaction':v[:,:,3]-v[:,:,0]-v[:,:,1]+v[:,:,2]}
        for name,delta in pairs.items():
            summaries[name+'/'+variant]=assess(delta,domains)
            for d in range(4):
                for j in range(6):
                    for h,head in enumerate(('OD','OC')):
                        functional_rows.append(dict(contrast=name,variant=variant,domain=f'D{d}',seed=SEEDS[j//2],order=j%2,head=head,delta_Dice_pp=float(delta[domains==d,j,h].mean())))
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(functional_rows[0]),lineterminator='\n');w.writeheader();w.writerows(functional_rows);put(c,root/'public/FUNCTIONAL_CELLS.csv',buf.getvalue().encode())
    results=[]
    for ai,arm in enumerate(ARMS[:2]):
        delta=values[:,:,ai]-values[:,:,3];local=[];structure=[]
        main=assess(delta[:,:,0,:,0],domains,True);post_test=assess(delta[:,:,1,:,0],domains,True)
        for vi,variant in enumerate(('raw','post')):
            for d in range(4):
                for h,head in enumerate(('OD','OC')):
                    v=delta[domains==d,:,vi,h,0];dm=float(v.mean());order=[float(v[:,o::2].mean()) for o in (0,1)]
                    local.append(dict(variant=variant,domain=f'D{d}',head=head,delta_pp=dm,order_pp=order,passed=bool(dm>=-.5 and min(order)>=-2)))
                    for mi in (4,5,6,9):
                        if mi==6 and h==1:continue
                        aa=values[domains==d,:,ai,vi,h,mi];bb=values[domains==d,:,3,vi,h,mi]
                        na_a=int((~np.isfinite(aa)).sum());na_b=int((~np.isfinite(bb)).sum())
                        va=int((aa>0).sum()) if mi==9 else na_a;vb=int((bb>0).sum()) if mi==9 else na_b
                        structure.append(dict(variant=variant,scope=f'D{d}/{head}',metric=names[mi]+'_missing_or_empty',candidate=va,VV=vb,passed=bool(va<=vb and na_a<=na_b)))
            for mi in (4,5,6):
                selected=[0] if mi==6 else [0,1];x=delta[:,:,vi][:,:,selected,mi];tol=1e-9 if mi==6 else 1e-6
                finite=bool(np.isfinite(x).all());mean=float(macro(x,domains).mean()) if finite else None
                structure.append(dict(variant=variant,scope='overall',metric=names[mi],delta=mean,tolerance=tol,passed=bool(finite and mean<=tol)))
        local_pass=all(x['passed'] for x in local);structure_pass=all(x['passed'] for x in structure);post_positive=post_test['mean_pp']>0
        promoted=main['performance_pass'] and local_pass and structure_pass and post_positive
        status='DEVELOPMENT_CANDIDATE_PASS' if promoted else 'TRADEOFF_NOT_PROMOTED' if main['performance_pass'] else 'BELOW_REGISTERED_INCREMENT' if main['below_minimum_supported'] else 'INSUFFICIENT_EVIDENCE'
        results.append(dict(arm=arm,status=status,promoted=promoted,primary_vs_VV=main,post_vs_VV=post_test,vs_C2_raw=assess(values[:,:,ai,0,:,0]-values[:,:,2,0,:,0],domains),vs_C2_post=assess(values[:,:,ai,1,:,0]-values[:,:,2,1,:,0],domains),local_pass=local_pass,structure_pass=structure_pass,post_positive=post_positive,local=local,structure=structure))
    passing=[r['arm'] for r in results if r['promoted']]
    write('public/RESULT.json',dict(status='COMPLETE',candidates=results,passing_candidates=passing,selected_candidate=passing[0] if passing else None,selection_priority=['W_ONLY','LSO_ONLY'],functional_contrasts=summaries,independent_confirmation=False,threshold_pp=.2,primary_family=2,primary_confidence=.975,auxiliary_intervals_descriptive=True,no_automatic_R108=True))
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
    assert totals==dict(physical_updates=2340,completed_updates=2340,physical_forwards=21084,training_backwards=2340)
    write('public/COMPLETION.json',dict(status='COMPLETE',**totals,new_label_contents=192,scored_streams=12,all_GPU_retired_before_scoring=True,raw_post_sealed_before_labels=True,CPU_body_seconds=time.time()-start))
    return dict(status='COMPLETE',passing_candidates=passing)
