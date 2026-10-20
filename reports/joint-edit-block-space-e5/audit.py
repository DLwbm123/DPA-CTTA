"""Label-visible joint block oracle; no deployable decision rule or model calls."""
from pathlib import Path
import importlib.util
import json, os, sys, time
from fractions import Fraction
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_erosion, distance_transform_edt

spec=importlib.util.spec_from_file_location('qualified_formula',Path(__file__).parent/'formula_check.py')
formula=importlib.util.module_from_spec(spec);spec.loader.exec_module(formula)
BLOCKS=[0,1,2,3,0]
ARMS=('SOURCE','TOPO_SMOOTH','JOINT_BLOCK_ORACLE')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def score(p,y):
    d=int(p.sum()+y.sum())
    return Fraction(2*int((p&y).sum()),d) if d else Fraction(1)
def assemble(source,candidate,truth):
    groups,n=label((source!=candidate).any(0),structure=np.ones((3,3),dtype=int))
    area=np.bincount(groups.ravel(),minlength=n+1).astype(np.int64)[1:]
    diff=candidate[1].astype(np.int64)-source[1].astype(np.int64)
    a=np.bincount(groups.ravel(),weights=diff.ravel(),minlength=n+1)[1:].astype(np.int64)
    t=np.bincount(groups.ravel(),weights=(diff*truth[1]).ravel(),minlength=n+1)[1:].astype(np.int64)
    # Bincount integer sums are exact at this fixed <=512^2 range.
    val,chosen,it=formula.solve(int(source[1].sum()),int(truth[1].sum()),int((source[1]&truth[1]).sum()),tuple((int(x),int(y)) for x,y in zip(a,t)))
    take=np.array((False,)+chosen,dtype=bool)[groups]
    pred=np.where(take[None],candidate,source)
    all_take=np.where((groups>0)[None],candidate,source)
    assert np.array_equal(all_take,candidate) and np.array_equal(np.where(np.zeros_like(take)[None],candidate,source),source)
    assert score(pred[1],truth[1])==val and val>=max(score(source[1],truth[1]),score(candidate[1],truth[1]))
    if not (source[1]&~source[0]).any() and not (candidate[1]&~candidate[0]).any():assert not (pred[1]&~pred[0]).any()
    oc_changes=np.bincount(groups.ravel(),weights=(source[1]!=candidate[1]).ravel(),minlength=n+1)[1:].astype(np.int64)
    od_only=(oc_changes==0);assert not (np.array(chosen,dtype=bool)&od_only).any()
    return pred,dict(sizes=area.tolist(),a=a.tolist(),t=t.tolist(),chosen=list(chosen),iterations=it,od_only=int(od_only.sum()),oc_active=int((~od_only).sum()),selected=int(sum(chosen)),blocks=n)
def checks():
    result=formula.check()
    s=np.zeros((2,4,4),dtype=bool);y=s.copy();c=s.copy()
    c[:,0,0]=True;c[:,1,1]=True;y[:,0,0]=True
    p,a=assemble(s,c,y);assert a['blocks']==1 # diagonal contact is 8-connected
    p,a=assemble(s,s,y);assert a['blocks']==0 and np.array_equal(p,s)
    c=s.copy();c[0,0,0]=True;p,a=assemble(s,c,y);assert a['od_only']==1 and a['selected']==0
    result.update(component_interface_checks=3,scope='exact solver and joint 8-connected installation on synthetic masks')
    return result
def summary(name,v):
    assert v.shape[0]==8 and np.isfinite(v).all()
    ids=np.random.default_rng(20261018).integers(0,8,(2000,8));bs=v[ids].mean(tuple(range(1,v.ndim+1)))
    return dict(comparison=name,delta_pp=float(v.mean()),ci95_pp=np.quantile(bs,[.025,.975]).tolist(),worst_cell_pp=float(v.min()),negative_cells=int((v<0).sum()))
def metric(pred,truth):
    heads=[]
    for p,y in zip(pred,truth):
        ps=p^binary_erosion(p,border_value=0);ys=y^binary_erosion(y,border_value=0)
        ds=np.concatenate((distance_transform_edt(~ys)[ps],distance_transform_edt(~ps)[ys])) if p.any() and y.any() else None
        heads.append(dict(dice_percent=float(100*score(p,y)),assd_pixel=float(ds.mean()) if ds is not None else None,hd95_pixel=float(np.percentile(ds,95)) if ds is not None else None,pred_empty=not bool(p.any())))
    def ratio(p):
        heights=[]
        for h in p:
            ys=np.nonzero(h)[0];heights.append(int(ys.max()-ys.min()+1) if len(ys) else 0)
        return heights[1]/heights[0] if heights[0] else None
    a,b=ratio(pred),ratio(truth)
    return dict(metrics=heads,vcdr_abs_error=abs(a-b) if a is not None and b is not None else None)
def finite(v):
    a=[x for x in v if x is not None and np.isfinite(x)]
    return dict(mean=float(np.mean(a)) if a else None,defined=len(a),total=len(v))
def run(source,out):
    started=time.time();source,out=Path(source),Path(out);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for mode in ('worker','score'):
        x=read(source/f'processes/{mode}.json');assert not x['active'] and x['exit_code']==0
        q=Path('/proc')/str(x['pid'])/'stat';assert not q.exists() or int(q.read_text().rsplit(')',1)[1].split()[19])!=x['start_ticks']
    assert read(source/'FULL_REPEATABILITY.json')['all_arms_exact'] and read(source/'STATIC_PARITY.json')['bit_arrays']==128
    rows=read(source/'private/predictions.json');cached={(r['case'],r['condition'],r['arm']):r for r in read(source/'scorer/individual.private.json')}
    panel=read(source/'scorer/panel.json');assert len(panel)==8
    masks={};metrics={};verified=0
    for r in rows:
        if r['arm'] not in ARMS[:2]:continue
        raw=(source/'predictions'/f"{r['index']:04d}.bits").read_bytes();assert len(raw)==65536
        masks[r['case'],r['condition'],r['arm']]=np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(2,512,512).astype(bool)
    assert set(masks)=={(i,k,a) for i in range(8) for k in range(4) for a in ARMS[:2]}
    regions={};records=[];all_sizes=[];selected_sizes=[];budget=[]
    for i,item in enumerate(panel):
        with Image.open(item['mask_path']) as im:
            assert list(im.size)==item['image_size'];g=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((g<255,g==0))
        for k in range(4):
            s,c=(masks[i,k,a] for a in ARMS[:2]);p,diag=assemble(s,c,gt);masks[i,k,ARMS[2]]=p
            np.packbits(p).tofile(out/'private'/f'{4*i+k:02d}.bits')
            for a in ARMS[:2]:
                v=cached[i,k,a];metrics[i,k,a]=dict(metrics=v['metrics'],vcdr_abs_error=v['ratios']['vcdr_abs_error'])
                for ch in range(2):assert abs(100*float(score(masks[i,k,a][ch],gt[ch]))-v['metrics'][ch]['dice_percent'])<=1e-12;verified+=1
            metrics[i,k,ARMS[2]]=metric(p,gt)
            records.append(dict(case=i,condition=k,**diag));all_sizes.extend(diag['sizes']);selected_sizes.extend(v for v,keep in zip(diag['sizes'],diag['chosen']) if keep)
            budget.append(dict(added=[int((a&~b).sum()) for a,b in zip(p,s)],deleted=[int((~a&b).sum()) for a,b in zip(p,s)]))
            for ch,name in enumerate(('OD','OC')):
                y=gt[ch];surface=y^binary_erosion(y,border_value=0);bd=distance_transform_edt(~surface)<=3 if surface.any() else np.zeros_like(y)
                for region,r in [('all',np.ones_like(y)),('foreground',y),('background',~y),('boundary3px',bd)]:
                    assert r.sum()>0
                    for base in ARMS[:2]:
                        b=masks[i,k,base][ch];corr=int(((b!=y)&(p[ch]==y)&r).sum());harm=int(((b==y)&(p[ch]!=y)&r).sum());d=int(((b!=p[ch])&r).sum());assert corr+harm==d
                        regions[i,k,base,name,region]=dict(corrected=corr,spoiled=harm,disagreement=d,pixels=int(r.sum()))
    cubes={};absolute=[]
    for a in ARMS:
        cube=np.array([[[metrics[i,k,a]['metrics'][ch]['dice_percent'] for ch in range(2)] for k in BLOCKS] for i in range(8)]);cubes[a]=cube
        rr=[metrics[i,k,a] for i in range(8) for k in BLOCKS]
        absolute.append(dict(arm=a,all_dice=float(cube.mean()),od_dice=float(cube[:,:,0].mean()),oc_dice=float(cube[:,:,1].mean()),equal_four_condition_oc=float(np.mean([metrics[i,k,a]['metrics'][1]['dice_percent'] for i in range(8) for k in range(4)])),assd=finite([h['assd_pixel'] for v in rr for h in v['metrics']]),hd95=finite([h['hd95_pixel'] for v in rr for h in v['metrics']]),vcdr=finite([v['vcdr_abs_error'] for v in rr]),empty_heads=sum(h['pred_empty'] for v in rr for h in v['metrics']),nesting_pixels=sum(int((masks[i,k,a][1]&~masks[i,k,a][0]).sum()) for i in range(8) for k in BLOCKS)))
    contrasts=[];region_rows=[]
    for base in ARMS[:2]:
        d=cubes[ARMS[2]]-cubes[base]
        for name,ch in [('ALL',None),('OD',0),('OC',1)]:contrasts.append(summary('ORACLE-'+base+':'+name,d if ch is None else d[:,:,ch]))
        contrasts.append(summary('ORACLE-'+base+':ORIGINAL_CONDITION_STATIC',d[:,0]))
        for name in ('OD','OC'):
            for region in ('all','foreground','background','boundary3px'):
                rs=[regions[i,k,base,name,region] for i in range(8) for k in range(4)]
                rates=np.array([[100*(regions[i,k,base,name,region]['corrected']-regions[i,k,base,name,region]['spoiled'])/regions[i,k,base,name,region]['pixels'] for k in BLOCKS] for i in range(8)])
                region_rows.append(dict(base=base,channel=name,region=region,**{key:sum(v[key] for v in rs) for key in rs[0]},net=summary('NET',rates)))
    def size_summary(v):return dict(count=len(v),quantiles=[float(x) for x in np.quantile(v,[0,.25,.5,.75,.9,.99,1])] if v else None,singleton_count=sum(x==1 for x in v),sum_pixels=sum(v))
    primary=next(v for v in contrasts if v['comparison']=='ORACLE-TOPO_SMOOTH:OC')
    result=dict(posthoc_label_visible=True,absolute=absolute,contrasts=contrasts,regions=region_rows,primary=primary,block_space_worth_further_information_review=primary['delta_pp']>=.2 and primary['ci95_pp'][0]>0,unlabeled_signal_qualified=False,advance_to_feature_protocol=False,continual_benefit_tested=False,blocks=dict(all=size_summary(all_sizes),selected=size_summary(selected_sizes),od_only=sum(v['od_only'] for v in records),oc_active=sum(v['oc_active'] for v in records),per_input_min=min(v['blocks'] for v in records),per_input_max=max(v['blocks'] for v in records),max_iterations=max(v['iterations'] for v in records)),budget=dict(added_by_head=[sum(v['added'][ch] for v in budget) for ch in range(2)],deleted_by_head=[sum(v['deleted'][ch] for v in budget) for ch in range(2)]))
    save(out/'private/matrices.json',dict(blocks=records,regions=[dict(case=i,condition=k,base=b,channel=ch,region=r,**v) for (i,k,b,ch,r),v in regions.items()]))
    save(out/'public/RESULTS.json',result)
    save(out/'public/ACCESS_COVERAGE.json',dict(status='PASS',unique_inputs=32,registered_labels=8,endpoint_bit_arrays=64,cached_endpoint_metric_items=64,dice_heads_verified=verified,oracle_masks=32,absolute_rows=3,region_rows=16,contrasts=8,all_none_install_parity_arrays=64,rgb_reads=0,weight_reads=0,model_forwards=0,updates=0,gpu_seconds=0,body_seconds=time.time()-started))
if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='check':print(json.dumps(checks(),indent=2))
    else:run(sys.argv[1],sys.argv[2])
