"""E4 post-hoc CPU audit; never an online selector. Private matrices stay private."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label

BASE=('SOURCE','FILL_SOURCE','SMOOTH','TOPO_SMOOTH','UNIFORM_LOGIT')
EXT=BASE+('EDGE_LOGIT',)
ALL=EXT+('ROLL_MATCH','UNIFORM_MATCH')
BLOCKS=[0,1,2,3,0]

def read(p): return json.loads(Path(p).read_text())
def save(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def dice(p,y):
    n=int(p.sum()+y.sum())
    return float(200*(p&y).sum()/n) if n else 100.
def counts(p,y,b,r):
    c=int(((b!=y)&(p==y)&r).sum()); h=int(((b==y)&(p!=y)&r).sum())
    d=int(((b!=p)&r).sum()); assert c+h==d
    return dict(corrected=c,spoiled=h,disagreement=d,pixels=int(r.sum()))
def pick(d): return np.argmax(d,axis=2)
def interval(v):
    ids=np.random.default_rng(20261018).integers(0,8,(2000,8))
    return np.quantile(v[ids].mean(axis=tuple(range(1,v.ndim+1))),[.025,.975]).tolist()
def oracle_space(d):
    # d: content, bound condition, arm, head. OC picks ONE arm for BOTH heads.
    s=pick(d[:,:,:,1]); chosen=np.take_along_axis(d,s[:,:,None,None],axis=2)[:,:,0,:]
    return s,chosen,float(chosen[:,:,1].mean()-d[:,:,:,1].mean((0,1)).max())
def checks():
    d=np.zeros((8,4,2,2)); d[:,:,0,0]=99; d[:,:,1,0]=2
    s,c,g=oracle_space(d); assert (s==0).all() and (c[:,:,0]==99).all() and g==0
    d[:4,:,0,1]=1; d[4:,:,1,1]=1
    s,c,g=oracle_space(d); assert g==.5 and (c[:4,:,0]==99).all() and (c[4:,:,0]==2).all()
    assert oracle_space(d[:,:,:1])[2]==0
    y=np.array([True,False,True,False]); b=~y
    assert counts(y,y,b,np.ones(4,bool))['corrected']==4
    assert dice(np.zeros(4,bool),np.zeros(4,bool))==100
    return dict(status='PASS',tests=5,scope='oracle tie, shared heads, complementarity, singleton, correction and empty Dice')

def run(source,out):
    started=time.time(); assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source,out=Path(source),Path(out)
    for n in ('worker','score'):
        p=read(source/f'processes/{n}.json'); assert not p['active'] and p['exit_code']==0
        q=Path('/proc')/str(p['pid'])/'stat'
        assert not q.exists() or int(q.read_text().rsplit(')',1)[1].split()[19])!=p['start_ticks']
    assert read(source/'SCORE_COMPLETE.json')['unique_metric_evaluations']==256
    assert read(source/'FULL_REPEATABILITY.json')['all_arms_exact']
    assert read(source/'STATIC_PARITY.json')['bit_arrays']==128
    rows=read(source/'private/predictions.json'); mm=read(source/'scorer/individual.private.json')
    panel=read(source/'scorer/panel.json'); assert len(panel)==8 and len(rows)==len(mm)==256
    lookup={(r['case'],r['condition'],r['arm']):r for r in mm}
    assert set(lookup)=={(i,k,a) for i in range(8) for k in range(4) for a in ALL}
    masks={}
    for r in rows:
        raw=(source/'predictions'/f"{r['index']:04d}.bits").read_bytes(); assert len(raw)==65536
        masks[r['case'],r['condition'],r['arm']]=np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(2,512,512).astype(bool)
    ys=[]
    for r in panel:
        with Image.open(r['mask_path']) as im:
            assert list(im.size)==r['image_size']
            gray=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        ys.append(np.stack((gray<255,gray==0)))
    d=np.empty((8,4,8,2)); verified=0
    for (i,k,a),r in lookup.items():
        for ch in range(2):
            v=dice(masks[i,k,a][ch],ys[i][ch]); assert abs(v-r['metrics'][ch]['dice_percent'])<=1e-12
            d[i,k,ALL.index(a),ch]=v; verified+=1
    region_rows=[]; private_counts=[]; edits=[]
    for base in ('TOPO_SMOOTH','FILL_SOURCE'):
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','background','boundary3px'):
                cs=[]; nets=np.empty((8,4))
                for i in range(8):
                    y=ys[i][ch]; surf=y^binary_erosion(y,border_value=0)
                    bd=distance_transform_edt(~surf)<=3 if surf.any() else np.zeros_like(y)
                    r={'all':np.ones_like(y),'foreground':y,'background':~y,'boundary3px':bd}[region]
                    for k in range(4):
                        p=masks[i,k,'EDGE_LOGIT'][ch]; b=masks[i,k,base][ch]; src=masks[i,k,'SOURCE'][ch]
                        v=counts(p,y,b,r); assert v['pixels']>0
                        v.update(unique_source_corrections=int(((src!=y)&(p==y)&(b!=y)&r).sum()),shared_source_corrections=int(((src!=y)&(p==y)&(b==y)&r).sum()))
                        cs.append(v); nets[i,k]=100*(v['corrected']-v['spoiled'])/v['pixels']
                        private_counts.append(dict(case=i,condition=k,base=base,channel=name,region=region,**v))
                region_rows.append(dict(base=base,channel=name,region=region,**{n:sum(v[n] for v in cs) for n in cs[0]},macro_net_pp=float(nets[:,BLOCKS].mean()),ci95_pp=interval(nets[:,BLOCKS])))
    for base in ('SOURCE','TOPO_SMOOTH','FILL_SOURCE'):
        for ch,name in enumerate(('OD','OC')):
            vals=[]
            for i in range(8):
                for k in range(4):
                    p=masks[i,k,'EDGE_LOGIT'][ch]; b=masks[i,k,base][ch]; add=p&~b; delete=b&~p
                    vals.append([int(add.sum()),int(delete.sum()),label(add,structure=np.ones((3,3)))[1],label(delete,structure=np.ones((3,3)))[1]])
            edits.append(dict(base=base,channel=name,unique_inputs=32,added_pixels=sum(v[0] for v in vals),deleted_pixels=sum(v[1] for v in vals),addition_components=sum(v[2] for v in vals),deletion_components=sum(v[3] for v in vals)))
    outputs=[]; selections={}; boot=np.random.default_rng(20261018).integers(0,8,(2000,8))
    for names in (BASE,EXT,ALL):
        a=d[:,:,[ALL.index(n) for n in names]]; s,chosen,g=oracle_space(a[:,BLOCKS]); selections[str(len(names))]=s.tolist()
        fixed=a[:,BLOCKS,:,-1].mean((0,1)); best=int(fixed.argmax())
        bs=np.array([oracle_space(a[ids][:,BLOCKS])[2] for ids in boot])
        metrics=[]; vc=[]; empty=nest=0
        for i in range(8):
            for j,k in enumerate(BLOCKS):
                arm=names[s[i,j]]; r=lookup[i,k,arm]; metrics.extend(r['metrics']); vc.append(r['ratios']['vcdr_abs_error']); p=masks[i,k,arm]
                nest+=int((p[1]&~p[0]).sum()); empty+=sum(not x.any() for x in p)
        def finite(v):
            valid=[x for x in v if x is not None and np.isfinite(x)]
            return dict(mean=float(np.mean(valid)) if valid else None,defined=len(valid),total=len(v))
        outputs.append(dict(candidates=list(names),best_fixed=names[best],best_fixed_oc=float(fixed[best]),oracle_oc=float(chosen[:,:,1].mean()),oracle_od=float(chosen[:,:,0].mean()),oracle_all=float(chosen.mean()),selection_gap_pp=g,selection_gap_ci95_pp=np.quantile(bs,[.025,.975]).tolist(),equal_four_condition_gap_pp=oracle_space(a)[2],weighted_selection_counts={n:int((s==j).sum()) for j,n in enumerate(names)},assd=finite([m['assd_pixel'] for m in metrics]),hd95=finite([m['hd95_pixel'] for m in metrics]),vcdr=finite(vc),empty_heads=empty,nesting_pixels=nest))
    base=d[:,:,[ALL.index(n) for n in BASE],1].max(2); ext=d[:,:,[ALL.index(n) for n in EXT],1].max(2)
    gain=ext-base; unique=dict(delta_pp=float(gain[:,BLOCKS].mean()),ci95_pp=interval(gain[:,BLOCKS]),positive_unique_inputs=int((gain>0).sum()),equal_four_condition_delta_pp=float(gain.mean()))
    prior=read(source/'private/config.json'); parent=source.parent
    pc=read(parent/'topology-projection-p2/private/config.json')
    alignment=dict(same_registered_checkpoint_path_e3_p2=prior['checkpoint_path']==pc['checkpoint_path'],checkpoint_reloaded=False,source_bn='eval, frozen parameters and running buffers per E3 mechanism receipt',threshold='independent heads source logit>=0; sealed candidate bits',input_grid=[512,512],label_resize='nearest',source_label_encoding='OD gray<255; OC gray==0',blocks=BLOCKS,conditions=['identity','gamma .7','rgb 1.15/1/.85','blur 1'],static_parity_arrays=128,patient_independence_verified=False,other_historical_panel_equivalence_verified=False,native_cw_source_bn_equivalent=False)
    save(out/'private/matrices.json',dict(regions=private_counts,oracle_selections=selections))
    results=dict(posthoc_label_visible=True,deployable_selector=False,continual_benefit_tested=False,regions=region_rows,edit_census=edits,oracles=outputs,e3_unique=unique,whole_candidate_selector_space_at_least_point2=outputs[0]['selection_gap_pp']>=.2,alignment=alignment)
    save(out/'public/RESULTS.json',results)
    save(out/'public/ACCESS_COVERAGE.json',dict(status='PASS',unique_inputs=32,bit_arrays=256,label_files=8,cached_metric_items=256,dice_heads_verified=verified,region_rows=len(region_rows),oracle_sets=3,rgb_reads=0,checkpoint_reads=0,model_forwards=0,updates=0,gpu_seconds=0,labels_posthoc_only=True,private_matrices_not_public=True,cpu_body_seconds=time.time()-started))

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='check': print(json.dumps(checks()))
    else: run(sys.argv[1],sys.argv[2])
