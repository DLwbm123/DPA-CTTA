"""Frozen CPU support qualification. No model or new candidate generation."""
import os,sys,json,io,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import readout

CONTROLS=('AMPLITUDE','ENERGY','GRADIENT','SHIFT','LOGIT_Q','RAW_MARGIN','GEOMETRY_ONLY','PERIMETER')
CACHE_CONTROLS=('LOGIT_Q','RAW_MARGIN','GEOMETRY_ONLY')
ARMS=('PHASE',)+CONTROLS
FULL=np.ones((3,3),bool)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def unpack(p):return np.unpackbits(np.frombuffer(Path(p).read_bytes(),np.uint8)).reshape(2,512,512).astype(bool)
def components(s,b):
    g,n=ndi.label((s!=b).any(0),FULL)
    return g,n

def access(c,panel):
    root=Path(c['root']).resolve();old=Path(c['sealed_root']).resolve();data=Path(c['source_root']).resolve();models=Path(c['model_root']).resolve()
    allowed={str(Path(p['image_path']).resolve()) for p in panel}
    bits={str(old/'inputs'/f'{i}_{k}_{arm}.bits') for i in range(8) for k in range(4) for arm in ('SOURCE','TOPO_SMOOTH')}
    def audit(event,args):
        if event.startswith('socket.'):raise PermissionError('network forbidden')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p)
        if p.is_relative_to(models) or (p.is_relative_to(data) and s not in allowed):raise PermissionError('unregistered data or model')
        if p.is_relative_to(old) and s not in bits:raise PermissionError('old metadata/results denied to measurement')
        if p.is_relative_to(root/'scorer'):raise PermissionError('scorer area forbidden')
        if p.is_relative_to(root) and ('score' in p.name.lower() and p.parent!=root/'code'):raise PermissionError('old score area forbidden')
        flags=args[2] if len(args)>2 else 0
        if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) and not p.is_relative_to(root):raise PermissionError('write outside current root')
    sys.addaudithook(audit)
    checks=0
    for p in (old/'scorer/panel.json',models/'forbidden.pth',data/'unregistered.png'):
        try:p.open('rb')
        except PermissionError:checks+=1
        else:raise AssertionError('negative access test failed')
    import socket
    try:socket.socket()
    except PermissionError:checks+=1
    else:raise AssertionError('network negative check failed')
    return checks

def legal(oc,od):
    return bool(oc.any() and not (oc&~od).any() and ndi.label(oc,FULL)[1]==1 and np.array_equal(oc,ndi.binary_fill_holes(oc,structure=readout.CROSS)))

def joint(b,s,g,ids):
    out=b.copy();m=np.isin(g,list(ids));out[1,m]=s[1,m]
    return out if legal(out[1],b[0]) else b.copy()

def quota(r):return (r['size'],r['direction'],r['distance_bucket'])

def decisions(b,s,g,rows,q,support):
    individually=[]
    for j,r in enumerate(rows):
        if support[j]:
            m=b[1].copy();sel=g==r['block_id'];m[sel]=s[1,sel]
            if legal(m,b[0]):individually.append(j)
    selected=[j for j in individually if q[j,0]>1e-12]
    masks=[joint(b,s,g,[rows[j]['block_id'] for j in selected])]
    # Disjoint blocks allow actual post-fallback selections to be recovered exactly.
    actual=[j for j,r in enumerate(rows) if (masks[0][1][g==r['block_id']]!=b[1][g==r['block_id']]).any()]
    target=Counter(quota(rows[j]) for j in actual);matched=[]
    for col in range(1,len(ARMS)):
        chosen=[]
        for key,count in sorted(target.items()):
            candidates=[j for j in individually if quota(rows[j])==key]
            candidates.sort(key=lambda j:(-q[j,col],rows[j]['block_id']))
            assert len(candidates)>=count;chosen+=candidates[:count]
        m=joint(b,s,g,[rows[j]['block_id'] for j in chosen]);masks.append(m)
        used=[j for j,r in enumerate(rows) if (m[1][g==r['block_id']]!=b[1][g==r['block_id']]).any()]
        matched.append(Counter(quota(rows[j]) for j in used)==target)
    return np.stack(masks),matched,len(actual)

def check(c):
    from oracle import solve
    from itertools import product
    assert readout.checks()==9
    # Fresh caller check: B->S signs and joint ratio, not additive single edits.
    changes=((3,1),(-2,-1),(1,1));base=Fraction(8,17)
    best,sel,_=solve(10,7,4,changes)
    exact=max(Fraction(2*(4+sum(t for take,(a,t) in zip(bits,changes) if take)),17+sum(a for take,(a,t) in zip(bits,changes) if take)) for bits in product((False,True),repeat=3))
    assert best==exact and best>=base
    root=Path(c['root']);old=Path(c['sealed_root'])
    panel=read(old/'private/panel.json');rows=read(old/'private/scores.json');assert len(panel)==8 and len(rows)==32
    clean=[]
    for frame in rows:
        rr=[]
        for r in frame['blocks']:
            out={k:r[k] for k in ('block_id','query','size','oc_pixels','direction','distance_bucket','common')}
            out['controls']={k:r['scores'][k] for k in CACHE_CONTROLS}
            if r['common']:assert all(x is not None and np.isfinite(x) for x in out['controls'].values())
            rr.append(out)
        clean.append(dict(case=frame['case'],condition=frame['condition'],blocks=rr))
    assert sum(len(r['blocks']) for r in clean)==3028
    assert sum(b['common'] for r in clean for b in r['blocks'])==2881
    assert sum(b['oc_pixels'] for r in clean for b in r['blocks'] if b['common'])==6997
    write(root/'private/panel.json',panel);write(root/'private/cache.json',clean)
    write(root/'public/CHECK.json',dict(status='PASS',joint_oracle_caller_bruteforce=True,registered_contents=8,inputs=32,all_oc_only_blocks=3028,common_blocks=2881,common_edit_pixels=6997,controls=list(CONTROLS),new_encoder_calls=0,gt_reads=0,readout_checks=9))

def appearances(x):
    import torch
    import torch.nn.functional as F
    axis=torch.arange(-3,4,dtype=x.dtype);k=torch.exp(-axis.square()/2);k/=k.sum();kernel=(k[:,None]*k[None,:]).expand(3,1,7,7)
    return (x.clone(),x.pow(.7).clamp(0,1),(x*x.new_tensor((1.15,1.,.85)).view(1,3,1,1)).clamp(0,1),F.conv2d(F.pad(x,(3,)*4,mode='reflect'),kernel,groups=3).clamp(0,1))

def worker(c):
    import torch
    root=Path(c['root']);old=Path(c['sealed_root']);mode=os.environ['AUDIT_MODE']
    panel=read(root/'private/panel.json');frames=read(root/'private/cache.json');bykey={(r['case'],r['condition']):r['blocks'] for r in frames}
    negative=access(c,panel);assert not torch.cuda.is_initialized()
    supported=changes=union=scorechanges=scoreunion=actions=actionchanges=0;matched=[True]*len(CONTROLS);no_phase=0
    npzroot=root/'private/measurement';npzroot.mkdir(exist_ok=True)
    for i,item in enumerate(panel):
        with Image.open(item['image_path']) as im:
            assert list(im.size)==item['image_size'];a=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),np.float32).transpose(2,0,1).copy()/255
        x=torch.from_numpy(a).unsqueeze(0)
        for k,view in enumerate(appearances(x)):
            lo=view.amin((1,2,3),keepdim=True);span=view.amax((1,2,3),keepdim=True)-lo;assert (span>0).all()
            green=((view-lo)/span)[0,1].numpy().astype(np.float64)*255
            # Field extraction receives no mask, candidate, appearance label or GT.
            ev=readout.fields(green)
            b=unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits');s=unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits');g,n=components(s,b);allrows=bykey[i,k]
            flat=g.ravel();od=np.bincount(flat,weights=(s[0]!=b[0]).ravel(),minlength=n+1);oc=np.bincount(flat,weights=(s[1]!=b[1]).ravel(),minlength=n+1)
            expected=set(np.flatnonzero((od==0)&(oc>0)).tolist());assert {r['block_id'] for r in allrows}==expected
            for r in allrows:assert np.array_equal(np.argwhere(g==r['block_id']),np.asarray(r['query'])) and r['size']==r['oc_pixels']==len(r['query'])
            rows=[r for r in allrows if r['common']];q=np.zeros((len(rows),len(ARMS)));u=np.zeros(len(rows),bool);bb=readout.outer(b[1])
            for j,r in enumerate(rows):
                candidate=b[1].copy();region=g==r['block_id'];candidate[region]=s[1,region]
                values,u[j]=readout.score(ev,b[1],candidate,bb)
                if not u[j]:continue
                q[j,:5]=values[:5];q[j,8]=values[5]
                for col,name in enumerate(CACHE_CONTROLS,5):q[j,col]=r['controls'][name]
                assert np.isfinite(q[j]).all()
            masks,match,selected=decisions(b,s,g,rows,q,u)
            payload=dict(ev);payload.update(green=green,q=q,support=u,masks=np.packbits(masks),matched=np.asarray(match),selected=np.array(selected))
            path=npzroot/f'{i}_{k}.npz'
            if mode=='worker1':np.savez_compressed(path,**payload)
            else:
                with np.load(path) as oldpass:
                    assert set(oldpass.files)==set(payload)
                    for name,v in payload.items():assert np.array_equal(oldpass[name],v),name
            supported+=int(u.sum());actions+=selected;no_phase+=not bool((ev['PHASE']>0).any())
            nonzero=(ev['PHASE']>0)|(ev['SHIFT']>0);union+=int(nonzero.sum());changes+=int(np.count_nonzero((ev['PHASE']!=ev['SHIFT'])&nonzero))
            nonzeroq=(np.abs(q[:,0])>1e-12)|(np.abs(q[:,4])>1e-12);scoreunion+=int(nonzeroq.sum());scorechanges+=int(np.count_nonzero((np.abs(q[:,0]-q[:,4])>1e-12)&nonzeroq))
            actionchanges+=int(not np.array_equal(masks[0],masks[4]));matched=[a and b for a,b in zip(matched,match)]
            write(root/'RUN_STATE.json',dict(stage=mode,completed_inputs=i*4+k+1))
    result=dict(status='PASS',source_sha=c['source_sha'],registered_rgb_decodes=8,measured_inputs=32,common_blocks=2881,supported_blocks=supported,no_phase_inputs=no_phase,shift_nonzero_union_positions=union,shift_changed_positions=changes,shift_nonzero_union_scores=scoreunion,shift_changed_scores=scorechanges,sealed_selected_blocks=actions,shift_final_masks_changed=actionchanges,matched_control_quotas=matched,independent_exact_replay=mode=='worker2',label_reads=0,model_weight_reads=0,model_calls=0,cuda_initialized=torch.cuda.is_initialized(),access_negative_checks=negative)
    write(root/'public'/f'{mode.upper()}.json',result)

def aggregate(a):
    a=np.asarray(a,float);contents=[]
    for row in a:
        ok=np.isfinite(row);contents.append(float(np.average(row[ok],weights=np.array([2,1,1,1])[ok])) if ok.any() else None)
    valid=[v for v in contents if v is not None];ci=None
    if len(valid)==8:
        ix=np.random.default_rng(20261018).integers(0,8,(2000,8));ci=np.quantile(np.asarray(contents)[ix].mean(1),[.025,.975]).tolist()
    return dict(mean=float(np.mean(valid)) if valid else None,ci95=ci,defined_inputs=int(np.isfinite(a).sum()),defined_contents=len(valid))

def auc(q,u,w=None):
    pos=u>0;neg=u<0
    if not pos.any() or not neg.any():return np.nan
    w=np.ones(len(u)) if w is None else w;pairs=w[pos,None]*w[None,neg];p=q[pos,None];n=q[None,neg]
    return float((((p>n)+.5*(p==n))*pairs).sum()/pairs.sum())

def score(c):
    from oracle import solve
    from runtime import identity
    root=Path(c['root']);old=Path(c['sealed_root'])
    for name in ('worker1','worker2'):
        rec=read(root/'processes'/f'{name}.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    generation=read(root/'public/WORKER2.json');assert generation['independent_exact_replay'] and generation['model_calls']==0
    panel=read(old/'scorer/panel.json');registered={r['mask_path']:r['expected'] for r in read(old/'scorer/mask_identity.private.json')};assert len(panel)==8 and {p['mask_path'] for p in panel}==set(registered)
    frames=read(root/'private/cache.json');bykey={(r['case'],r['condition']):[b for b in r['blocks'] if b['common']] for r in frames}
    supportspace=np.zeros((8,4));fullspace=np.zeros((8,4));base=np.zeros((8,4));data=[]
    for i,p in enumerate(panel):
        raw=Path(p['mask_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==registered[p['mask_path']]
        with Image.open(io.BytesIO(raw)) as im:
            assert list(im.size)==p['image_size'];y0=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((y0<255,y0==0));y=gt[1]
        for k in range(4):
            b=unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits');s=unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits');g,n=components(s,b);rows=bykey[i,k]
            with np.load(root/'private/measurement'/f'{i}_{k}.npz') as z:measured={name:z[name].copy() for name in ('q','support','masks','matched')}
            d=s[1].astype(np.int64)-b[1].astype(np.int64);area=np.bincount(g.ravel(),weights=d.ravel(),minlength=n+1).astype(np.int64);tp=np.bincount(g.ravel(),weights=(d*y).ravel(),minlength=n+1).astype(np.int64)
            P,G,T=int(b[1].sum()),int(y.sum()),int((b[1]&y).sum());Db=Fraction(2*T,P+G) if P+G else Fraction(1);base[i,k]=100*float(Db)
            for supported,out in ((False,fullspace),(True,supportspace)):
                rr=[r for j,r in enumerate(rows) if not supported or measured['support'][j]];changes=tuple((int(area[r['block_id']]),int(tp[r['block_id']])) for r in rr)
                best,selected,_=solve(P,G,T,changes);oc=b[1].copy()
                for r,take in zip(rr,selected):
                    if take:region=g==r['block_id'];oc[region]=s[1,region]
                den=int(oc.sum())+G;actual=Fraction(2*int((oc&y).sum()),den) if den else Fraction(1)
                assert best==actual and best>=Db
                out[i,k]=100*float(best-Db)
            data.append((i,k,b,s,g,rows,area,tp,P,G,T,Db,measured,gt))
    full=aggregate(fullspace);space=aggregate(supportspace)
    prior=read(old/'public/RESULTS.json');assert abs(full['mean']-prior['common_oracle_space']['delta_pp'])<=1e-12
    assert abs(aggregate(base)['mean']-87.8052924668)<1e-8
    result=dict(status='COMPLETE',source_sha=c['source_sha'],common_blocks=2881,common_edit_pixels=6997,all_oc_only_blocks=3028,registered_gt_files=8,inputs=32,base_oc_dice=aggregate(base),full_common_relaxed_upper_bound_pp=full,support_restricted_relaxed_upper_bound_pp=space,oracle_exact_mask_ratio_checks=64,topology_constraints_relaxed=True,space_gate=space['mean']>=.30,signal_status='NOT_RUN_SPACE_GATE',joint_status='NOT_RUN_SPACE_GATE',qualified=False)
    # The gate precedes construction of utility labels, AUC, or decision metrics.
    if not result['space_gate']:
        write(root/'public/RESULTS.json',result);return
    values={(name,weight):np.full((8,4),np.nan) for name in ARMS for weight in ('BLOCK','UTILITY')};zero=pos=neg=0
    for i,k,b,s,g,rows,area,tp,P,G,T,Db,z,gt in data:
        uu=[]
        for r in rows:
            j=r['block_id'];den=P+G+int(area[j]);after=Fraction(2*(T+int(tp[j])),den) if den else Fraction(1);uu.append(float(after-Db))
        uu=np.asarray(uu);zero+=int((uu==0).sum());pos+=int((uu>0).sum());neg+=int((uu<0).sum())
        for col,name in enumerate(ARMS):
            for weight,w in (('BLOCK',None),('UTILITY',np.abs(uu))):values[name,weight][i,k]=auc(z['q'][:,col],uu,w)
    av={name:aggregate(values[name,'BLOCK']) for name in ARMS};contrasts={name:aggregate(values['PHASE','BLOCK']-values[name,'BLOCK']) for name in CONTROLS};utility=aggregate(values['PHASE','UTILITY']-values['AMPLITUDE','UTILITY'])
    good=lambda a:a['mean'] is not None and a['ci95'] is not None and a['ci95'][0]>0
    main=contrasts['AMPLITUDE'];gen=generation
    gates=dict(auc=av['PHASE']['mean'] is not None and av['PHASE']['mean']>=.60,primary=good(main) and main['mean']>=.05,strong_controls=all(good(contrasts[name]) for name in CONTROLS[1:]),coverage=av['PHASE']['defined_inputs']>=24 and av['PHASE']['defined_contents']==8,utility_nonreverse=utility['mean'] is not None and utility['mean']>=0,shift_positions=gen['shift_nonzero_union_positions']>0 and gen['shift_changed_positions']/gen['shift_nonzero_union_positions']>=.5,shift_scores=gen['shift_nonzero_union_scores']>0 and gen['shift_changed_scores']/gen['shift_nonzero_union_scores']>=.5)
    result.update(signal_status='PASS' if all(gates.values()) else 'FAILED',signal_gates=gates,auc=av,contrasts=contrasts,utility_primary=utility,utility_labels=dict(positive=pos,negative=neg,zero=zero),joint_status='NOT_RUN_SIGNAL_GATE')
    if all(gates.values()):evaluate_joint(result,data)
    write(root/'public/RESULTS.json',result)

def evaluate_joint(result,data):
    from metric import metric
    metrics={};coverage={name:dict(empty_heads=0,nesting_pixels=0,surface_defined=0,vcdr_defined=0) for name in ('SOURCE','B')+ARMS};worst=0.;matched=True;odlock=True;new_empty=False;new_nesting=False
    def store(name,key,i,k,v):metrics.setdefault((name,key),np.full((8,4),np.nan))[i,k]=np.nan if v is None else v
    for i,k,b,s,g,rows,area,tp,P,G,T,Db,z,gt in data:
        masks=np.unpackbits(z['masks']).reshape(len(ARMS),2,512,512).astype(bool);matched=matched and bool(z['matched'].all())
        for name,pred in [('SOURCE',s),('B',b)]+list(zip(ARMS,masks)):
            heads,v=metric(pred,gt,b);cov=coverage[name]
            cov['nesting_pixels']+=int((pred[1]&~pred[0]).sum());cov['empty_heads']+=sum(h['pred_empty'] for h in heads);cov['surface_defined']+=sum(h['assd_pixel'] is not None for h in heads);cov['vcdr_defined']+=v['vcdr_abs_error'] is not None
            for h,hname in zip(heads,('OD','OC')):
                for key in ('dice_percent','assd_pixel','hd95_pixel'):store(name,hname+'_'+key,i,k,h[key])
            store(name,'vCDR_error',i,k,v['vcdr_abs_error'])
        odlock=odlock and np.array_equal(masks[0,0],b[0])
        new_empty=new_empty or any(not p.any() and baseline.any() for p,baseline in zip(masks[0],b))
        new_nesting=new_nesting or bool(((masks[0,1]&~masks[0,0])&~(b[1]&~b[0])).any())
    key='OC_dice_percent';benefit=aggregate(metrics['PHASE',key]-metrics['B',key]);comparisons={n:aggregate(metrics['PHASE',key]-metrics[n,key]) for n in ('SOURCE',)+CONTROLS}
    worst=max(float(np.nanmax(metrics['B',head+'_dice_percent']-metrics['PHASE',head+'_dice_percent'])) for head in ('OD','OC'))
    # Resample eight bound contents. Ratios use weighted content means first.
    ix=np.random.default_rng(20261018).integers(0,8,(2000,8));ratios={}
    for key in ('OC_assd_pixel','OC_hd95_pixel'):
        a=metrics['PHASE',key]@np.array([2,1,1,1])/5;b=metrics['B',key]@np.array([2,1,1,1])/5
        ratios[key]=float(np.quantile(a[ix].mean(1)/b[ix].mean(1),.975)) if np.isfinite(a).all() and np.isfinite(b).all() and (b>0).all() else None
    vcdr=aggregate(metrics['PHASE','vCDR_error']-metrics['B','vCDR_error']);good=lambda x:x['ci95'] is not None and x['ci95'][0]>0
    gates=dict(benefit=benefit['mean']>=.30 and good(benefit),od_locked=odlock,no_new_empty=not new_empty,no_new_nesting=not new_nesting,defined=coverage['PHASE']['surface_defined']==64 and coverage['PHASE']['vcdr_defined']==32,worst_dice_drop=worst<=1.,surfaces=all(v is not None and v<=1.05 for v in ratios.values()),vcdr=vcdr['ci95'] is not None and vcdr['ci95'][1]<=.005,matched_controls=matched and all(good(comparisons[n]) for n in CONTROLS))
    result.update(joint_status='PASS' if all(gates.values()) else 'FAILED',joint_gates=gates,joint_benefit_pp=benefit,joint_comparisons_pp=comparisons,joint_absolute=[dict(arm=a,metric=k,**aggregate(v)) for (a,k),v in metrics.items()],joint_coverage=coverage,worst_dice_drop_pp=worst,surface_ratio_upper=ratios,vcdr_increment=vcdr,qualified=all(gates.values()))

def run(c):
    mode=os.environ['AUDIT_MODE'];assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if mode=='check':check(c)
    elif mode in ('worker1','worker2'):worker(c)
    elif mode=='score':score(c)
    else:raise ValueError(mode)
