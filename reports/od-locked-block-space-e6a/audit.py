"""OD_LOCK exact label oracle and nonadditive size ablations. CPU only."""
from pathlib import Path
import importlib.util
import json,os,sys,time
import numpy as np
from scipy.ndimage import label,binary_erosion,distance_transform_edt
from PIL import Image

spec=importlib.util.spec_from_file_location('endpoint',Path(__file__).parent/'endpoint_audit.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SCENARIOS=('ALL_OC_ONLY','ONLY_1','ONLY_2_4','ONLY_GE5','NO_1','NO_2_4','NO_GE5')
ARMS=('SOURCE','TOPO_SMOOTH')+SCENARIOS
read,save,score,metric,summary,finite=base.read,base.save,base.score,base.metric,base.summary,base.finite
BLOCKS=base.BLOCKS

def block_data(s,b):
    groups,n=label((s!=b).any(0),structure=np.ones((3,3),int))
    sizes=np.bincount(groups.ravel(),minlength=n+1)[1:].astype(np.int64)
    changes=[np.bincount(groups.ravel(),weights=(s[ch]!=b[ch]).ravel(),minlength=n+1)[1:].astype(np.int64) for ch in range(2)]
    od,oc=changes;allowed=(od==0)&(oc>0)
    kinds=np.where(allowed,'OC_ONLY',np.where((od>0)&(oc==0),'OD_ONLY','MIXED'))
    assert int(((od>0)|(oc>0)).sum())==n
    return groups,sizes,od,oc,allowed,kinds
def solve(s,b,y,groups,allowed):
    n=len(allowed);d=s[1].astype(np.int64)-b[1].astype(np.int64)
    a=np.bincount(groups.ravel(),weights=d.ravel(),minlength=n+1)[1:].astype(np.int64)
    t=np.bincount(groups.ravel(),weights=(d*y[1]).ravel(),minlength=n+1)[1:].astype(np.int64)
    vals=tuple((int(x),int(z)) for x,z in zip(a[allowed],t[allowed]))
    value,chosen,it=base.formula.solve(int(b[1].sum()),int(y[1].sum()),int((b[1]&y[1]).sum()),vals)
    use=np.zeros(n+1,dtype=bool);use[1:][allowed]=chosen
    p=np.where(use[groups][None],s,b)
    assert np.array_equal(p[0],b[0]) and score(p[1],y[1])==value and value>=score(b[1],y[1])
    if not (s[1]&~s[0]).any() and not (b[1]&~b[0]).any():assert not (p[1]&~p[0]).any()
    return p,dict(withdrawn=np.flatnonzero(use[1:]).tolist(),iterations=it)
def checks():
    result=base.checks()
    s=np.zeros((2,5,5),bool);b=s.copy();y=s.copy()
    b[0,1,1:3]=True;b[1,1,2]=True # nested, joint 8-connected mixed block
    y[1,1,2]=True
    g,sz,od,oc,ok,k=block_data(s,b);assert list(k)==['MIXED'] and not ok.any()
    p,_=solve(s,b,y,g,ok);assert np.array_equal(p,b)
    b=s.copy();b[0]=True;b[1,2,2]=True;y=s.copy() # mixed whole union: OD forced B
    g,sz,od,oc,ok,k=block_data(s,b);p,_=solve(s,b,y,g,ok);assert np.array_equal(p[0],b[0])
    g,sz,od,oc,ok,k=block_data(s,s);p,_=solve(s,s,s,g,ok);assert np.array_equal(p,s)
    assert all(sum([x==1,2<=x<=4,x>=5])==1 for x in range(1,20))
    s=np.zeros((2,5,5),bool);s[0]=True;b=s.copy();b[1,0,0]=True;b[1,3,3:5]=True;y=s.copy();y[1,0,0]=True
    g,sz,od,oc,ok,k=block_data(s,b)
    p,_=solve(s,b,y,g,ok&(sz==1));assert score(p[1],y[1])==score(b[1],y[1])
    p,_=solve(s,b,y,g,ok&(sz>=2));assert score(p[1],y[1])==1
    result.update(od_lock_interface_checks=5)
    return result
def run(source,prior,out):
    started=time.time();source,prior,out=map(Path,(source,prior,out));assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for mode in ('worker','score'):
        v=read(source/f'processes/{mode}.json');assert not v['active'] and v['exit_code']==0
        q=Path('/proc')/str(v['pid'])/'stat';assert not q.exists() or int(q.read_text().rsplit(')',1)[1].split()[19])!=v['start_ticks']
    assert read(prior/'public/FULL_REPEATABILITY.json')['all_32_masks_exact'] and read(prior/'public/RESOURCE_LEDGER.json')['all_workers_retired']
    rows=read(source/'private/predictions.json');cached={(v['case'],v['condition'],v['arm']):v for v in read(source/'scorer/individual.private.json')};panel=read(source/'scorer/panel.json');assert len(panel)==8
    masks={};metrics={};regions={};records=[];verified=parity=oc_same=0
    for v in rows:
        if v['arm'] not in ARMS[:2]:continue
        raw=(source/'predictions'/f"{v['index']:04d}.bits").read_bytes();assert len(raw)==65536
        masks[v['case'],v['condition'],v['arm']]=np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(2,512,512).astype(bool)
    assert set(masks)=={(i,k,a) for i in range(8) for k in range(4) for a in ARMS[:2]}
    for i,item in enumerate(panel):
        with Image.open(item['mask_path']) as im:
            assert list(im.size)==item['image_size'];gray=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        y=np.stack((gray<255,gray==0))
        for k in range(4):
            s,b=(masks[i,k,a] for a in ARMS[:2]);groups,sizes,od,oc,allowed,kinds=block_data(s,b)
            original,original_diag=base.assemble(s,b,y)
            raw=(prior/'private'/f'{4*i+k:02d}.bits').read_bytes();assert len(raw)==65536 and np.array_equal(np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(2,512,512).astype(bool),original);parity+=1
            for a in ARMS[:2]:
                v=cached[i,k,a];metrics[i,k,a]=dict(metrics=v['metrics'],vcdr_abs_error=v['ratios']['vcdr_abs_error'])
                for ch in range(2):assert abs(100*float(score(masks[i,k,a][ch],y[ch]))-v['metrics'][ch]['dice_percent'])<=1e-12;verified+=1
            edits=(s[1]!=b[1]);surface=(s[1]^binary_erosion(s[1],border_value=0))|(b[1]^binary_erosion(b[1],border_value=0))
            dist=distance_transform_edt(~surface) if surface.any() else np.zeros_like(surface,dtype=float)
            distance_sum=np.bincount(groups.ravel(),weights=(dist*edits).ravel(),minlength=len(sizes)+1)[1:]
            expand=np.bincount(groups.ravel(),weights=(b[1]&~s[1]).ravel(),minlength=len(sizes)+1)[1:].astype(int)
            shrink=np.bincount(groups.ravel(),weights=(s[1]&~b[1]).ravel(),minlength=len(sizes)+1)[1:].astype(int)
            rec=dict(case=i,condition=k,sizes=sizes.tolist(),od_changes=od.tolist(),oc_changes=oc.tolist(),kinds=kinds.tolist(),oc_expansion=expand.tolist(),oc_shrinkage=shrink.tolist(),predicted_boundary_mean_distance=[float(d/n) if n else None for d,n in zip(distance_sum,oc)],selections={})
            size_masks=(np.ones_like(allowed),sizes==1,(sizes>=2)&(sizes<=4),sizes>=5,sizes!=1,~((sizes>=2)&(sizes<=4)),sizes<5)
            for a,sz in zip(SCENARIOS,size_masks):
                p,diag=solve(s,b,y,groups,allowed&sz);masks[i,k,a]=p;metrics[i,k,a]=metric(p,y);rec['selections'][a]=diag
                np.packbits(p).tofile(out/'private'/f'{a}_{4*i+k:02d}.bits')
                assert score(p[1],y[1])<=score(original[1],y[1])
                if a=='ALL_OC_ONLY':oc_same+=int(np.array_equal(p[1],original[1]))
            records.append(rec)
            p=masks[i,k,'ALL_OC_ONLY']
            for ch,name in enumerate(('OD','OC')):
                gy=y[ch];gs=gy^binary_erosion(gy,border_value=0);bd=distance_transform_edt(~gs)<=3 if gs.any() else np.zeros_like(gy)
                for region,r in [('all',np.ones_like(gy)),('foreground',gy),('background',~gy),('boundary3px',bd)]:
                    assert r.sum()>0
                    for a in ARMS[:2]:
                        bb=masks[i,k,a][ch];c=int(((bb!=gy)&(p[ch]==gy)&r).sum());h=int(((bb==gy)&(p[ch]!=gy)&r).sum());d=int(((bb!=p[ch])&r).sum());assert c+h==d
                        regions[i,k,a,name,region]=dict(corrected=c,spoiled=h,disagreement=d,pixels=int(r.sum()))
    cubes={};absolute=[]
    for a in ARMS:
        cube=np.array([[[metrics[i,k,a]['metrics'][ch]['dice_percent'] for ch in range(2)] for k in BLOCKS] for i in range(8)]);cubes[a]=cube
        rr=[metrics[i,k,a] for i in range(8) for k in BLOCKS]
        absolute.append(dict(arm=a,all_dice=float(cube.mean()),od_dice=float(cube[:,:,0].mean()),oc_dice=float(cube[:,:,1].mean()),equal_four_condition_oc=float(np.mean([metrics[i,k,a]['metrics'][1]['dice_percent'] for i in range(8) for k in range(4)])),assd=finite([h['assd_pixel'] for v in rr for h in v['metrics']]),hd95=finite([h['hd95_pixel'] for v in rr for h in v['metrics']]),vcdr=finite([v['vcdr_abs_error'] for v in rr]),empty_heads=sum(h['pred_empty'] for v in rr for h in v['metrics']),nesting_pixels=sum(int((masks[i,k,a][1]&~masks[i,k,a][0]).sum()) for i in range(8) for k in BLOCKS)))
    contrasts=[];region_rows=[];size_space=[]
    for a in ARMS[:2]:
        d=cubes['ALL_OC_ONLY']-cubes[a]
        for name,ch in [('ALL',None),('OD',0),('OC',1)]:contrasts.append(summary('OD_LOCK-'+a+':'+name,d if ch is None else d[:,:,ch]))
        contrasts.append(summary('OD_LOCK-'+a+':ORIGINAL_CONDITION_STATIC',d[:,0]))
        for name in ('OD','OC'):
            for region in ('all','foreground','background','boundary3px'):
                rs=[regions[i,k,a,name,region] for i in range(8) for k in range(4)]
                rates=np.array([[100*(regions[i,k,a,name,region]['corrected']-regions[i,k,a,name,region]['spoiled'])/regions[i,k,a,name,region]['pixels'] for k in BLOCKS] for i in range(8)])
                region_rows.append(dict(base=a,channel=name,region=region,**{key:sum(v[key] for v in rs) for key in rs[0]},net=summary('NET',rates)))
    for a in SCENARIOS:
        size_space.append(dict(arm=a,vs_B=summary('SPACE',cubes[a][:,:,1]-cubes['TOPO_SMOOTH'][:,:,1]),vs_full=summary('DELTA_FROM_FULL',cubes[a][:,:,1]-cubes['ALL_OC_ONLY'][:,:,1])))
    metadata=[]
    for kind in ('OD_ONLY','OC_ONLY','MIXED'):
        for bucket in ('1','2_4','GE5'):
            rr=[(v,j) for v in records for j,k in enumerate(v['kinds']) if k==kind and (v['sizes'][j]==1 if bucket=='1' else 2<=v['sizes'][j]<=4 if bucket=='2_4' else v['sizes'][j]>=5)]
            metadata.append(dict(kind=kind,size_bucket=bucket,blocks=len(rr),union_pixels=sum(v['sizes'][j] for v,j in rr),od_pixels=sum(v['od_changes'][j] for v,j in rr),oc_pixels=sum(v['oc_changes'][j] for v,j in rr),expand_pixels=sum(v['oc_expansion'][j] for v,j in rr),shrink_pixels=sum(v['oc_shrinkage'][j] for v,j in rr),pred_boundary_distance=finite([v['predicted_boundary_mean_distance'][j] for v,j in rr])))
    primary=next(v for v in contrasts if v['comparison']=='OD_LOCK-TOPO_SMOOTH:OC')
    result=dict(posthoc_label_visible=True,primary=primary,od_lock_space_pass=primary['delta_pp']>=.2 and primary['ci95_pp'][0]>0,signal_qualified=False,selector_applied=False,continual_benefit_tested=False,absolute=absolute,contrasts=contrasts,regions=region_rows,size_space=size_space,block_metadata=metadata,e5_oracle_parity_inputs=parity,od_lock_oc_mask_equal_e5_inputs=oc_same)
    save(out/'private/matrices.json',dict(blocks=records,regions=[dict(case=i,condition=k,base=a,channel=ch,region=r,**v) for (i,k,a,ch,r),v in regions.items()]))
    save(out/'public/RESULTS.json',result)
    save(out/'public/ACCESS_COVERAGE.json',dict(status='PASS',unique_inputs=32,registered_labels=8,endpoint_bits=64,e5_oracle_bits=32,cached_endpoint_items_used=64,cached_container_items=len(cached),dice_heads_verified=verified,oracle_masks=224,absolute_rows=9,region_rows=16,primary_contrasts=8,size_scenarios=7,od_bit_identity_checks=224,e5_oracle_parity_inputs=parity,rgb_reads=0,weight_reads=0,model_forwards=0,updates=0,gpu_seconds=0,body_seconds=time.time()-started))
if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='check':print(json.dumps(checks(),indent=2))
    else:run(sys.argv[1],sys.argv[2],sys.argv[3])
