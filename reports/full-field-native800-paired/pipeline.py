"""Full-field paired observation; two fresh decode/model processes before GT scoring."""
import os
import sys
import io
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
ARMS=('SOURCE','B','N','R')
SEED=20261018
INDEX=np.random.default_rng(SEED).integers(0,8,(10000,8))

def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def unpack(p):
    b=Path(p).read_bytes();assert len(b)==65536
    return np.unpackbits(np.frombuffer(b,np.uint8)).reshape(2,512,512).astype(bool)

def resize(x,size):
    import torch.nn.functional as F
    return F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False,antialias=False)

def check(c):
    import torch
    torch.set_num_threads(2)
    a=torch.arange(800,dtype=torch.float64);plane=torch.stack((a[:,None].expand(800,800),a[None,:].expand(800,800)))[None]
    out=resize(plane,512);q=(torch.arange(512,dtype=torch.float64)+.5)*800/512-.5
    err=max(float((out[0,0]-q[:,None]).abs().max()),float((out[0,1]-q[None,:]).abs().max()))
    const=float((resize(torch.full((1,3,800,800),.37,dtype=torch.float64),512)-.37).abs().max())
    lo=torch.arange(512,dtype=torch.float64)[None,None,:,None].expand(1,1,512,512)
    rt=resize(resize(lo,800),512);inner=float((rt-lo)[:,:,2:-2,2:-2].abs().max());edge=float((rt-lo).abs().max())
    step=torch.zeros(1,1,800,800,dtype=torch.float64);step[:,:,:,400:]=1
    small=resize(step,512);assert small.min()>=0 and small.max()<=1 and (small[:,:,:,1:]-small[:,:,:,:-1]>=0).all()
    assert torch.equal(out,resize(plane,512)) and err<1e-9 and inner<1e-9 and const<1e-12
    assert not torch.cuda.is_initialized()
    save(Path(c['root'])/'public/CHECK.json',dict(status='PASS',coordinate_error_max=err,constant_error_max=const,
        roundtrip_interior_error_max=inner,roundtrip_full_edge_difference_max=edge,excluded_border_pixels=2,
        synthetic_step_range_and_monotonicity=True,model_calls=0,label_reads=0,antialias=False))

def digest(t):
    a=t.detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str((str(a.dtype),list(a.shape))).encode());h.update(a.numpy().tobytes());return h.hexdigest()

def worker(c,passno):
    import torch
    root,old=Path(c['root']),Path(c['sealed_root']);panel=read(old/'private/panel.json')
    assert len(panel)==8 and all('mask_path' not in p for p in panel)
    images={Path(p['image_path']).resolve() for p in panel};sources={(old/'inputs'/f'{i}_0_SOURCE.bits').resolve() for i in range(8)}
    checkpoint=Path(c['checkpoint_path']).resolve();opened=set();denied=[]
    def audit(event,args):
        if event=='socket.connect':raise PermissionError('network denied')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        forbid=p.is_relative_to(old) and not p.is_relative_to(old/'code') and p not in sources and p!=old/'private/panel.json'
        forbid|=p.is_relative_to(Path(c['source_root'])) and p not in images
        forbid|=p.is_relative_to(Path(c['model_root'])) and p!=checkpoint
        forbid|=p.name in ('evaluate.py','score.py','e8_score.py','signal_score.py')
        if forbid:denied.append(str(p));raise PermissionError('outside generation read set')
        if p in images or p in sources or p==checkpoint:opened.add(str(p))
    sys.addaudithook(audit)
    for p in (old/'scorer/panel.json',old/'private/scores.json'):
        try:p.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('access gate')
    sys.path.insert(0,str(old/'code/src'))
    from dpa_ctta.integrations.ctta_suite import build_reference_model
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=build_reference_model('fundus');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.cuda().eval().requires_grad_(False)
    norms=[]
    for name,m in model.named_modules():
        if isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.instancenorm._InstanceNorm,torch.nn.GroupNorm,torch.nn.LayerNorm)):
            norms.append(dict(name=name,type=type(m).__name__,training=m.training,track_running_stats=getattr(m,'track_running_stats',None),eps=m.eps,
                momentum=getattr(m,'momentum',None),affine=getattr(m,'affine',None),running_mean_none=getattr(m,'running_mean',None) is None))
    assert norms and all(not m['training'] for m in norms)
    save(root/'private'/f'NORM{passno}.json',norms)
    if passno==2:assert norms==read(root/'private/NORM1.json')
    before={n:t.cpu().clone() for n,t in model.state_dict().items()};replay=[];clip={n:0 for n in ('SOURCE','N','R')};diagnostics=[];nf=nh=parity=0
    previous=read(root/'private/REPLAY1.json') if passno==2 else None
    with torch.no_grad():
        for i,p in enumerate(panel):
            raw=Path(p['image_path']).read_bytes();raw_sha=hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as im:
                assert list(im.size)==p['image_size']==[800,800] and im.mode=='RGB' and im.format=='PNG'
                native=torch.from_numpy(np.asarray(im,dtype=np.float32).transpose(2,0,1).copy()/255)[None]
                low=torch.from_numpy(np.asarray(im.resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32).transpose(2,0,1).copy()/255)[None]
            lo=low.amin();span=low.amax()-lo;assert span>0
            inputs=[(low-lo)/span,(native-lo)/span,(resize(low,800)-lo)/span]
            s=unpack(old/'inputs'/f'{i}_0_SOURCE.bits')
            for name,v in zip(('SOURCE','N','R'),inputs):
                clip[name]+=int(((v<0)|(v>1)).sum());v=v.clamp(0,1)
                assert torch.isfinite(v).all() and v.dtype==torch.float32
                out=model(v.cuda());z,f=out[0],out[2];nf+=1
                size=512 if name=='SOURCE' else 800
                assert tuple(z.shape)==(1,2,size,size) and tuple(f.shape)==(1,32,size,size) and torch.isfinite(z).all() and torch.isfinite(f).all()
                assert torch.equal(model.seg_head(f),z);nh+=1
                zz=z.cpu();mapped=zz if name=='SOURCE' else resize(zz,512)
                pred=mapped[0].numpy()>=0
                if name=='SOURCE':assert np.array_equal(pred,s);parity+=1
                row=dict(case=i,arm=name,raw_sha=raw_sha,input_digest=digest(v),logits_digest=digest(z),features_digest=digest(f))
                mask=root/'private/predictions'/f'{i}_{name}.bits';packed=np.packbits(pred.reshape(-1)).tobytes()
                if passno==1:mask.write_bytes(packed)
                else:assert row==previous[len(replay)] and mask.read_bytes()==packed
                replay.append(row)
                diagnostics.append(dict(arm=name,native_positive_fraction=[float((zz[0,j]>=0).float().mean()) for j in range(2)],
                    mapped_positive_fraction=[float(pred[j].mean()) for j in range(2)],native_empty=[not bool((zz[0,j]>=0).any()) for j in range(2)],
                    mapped_empty=[not bool(pred[j].any()) for j in range(2)],logit_min=float(zz.min()),logit_max=float(zz.max()),logit_mean=float(zz.mean())))
                if nf==1:
                    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
                    own=[line for line in lines if line.split(',')[0].strip()==str(os.getpid())]
                    assert len(own)==1 and c['gpu_uuid'] in own[0] and not any(t in own[0].lower() for t in ('wangbomin','ctta','native'))
                    save(root/'private'/f'GPU{passno}.json',dict(status='PASS',line=own[0]))
            save(root/'RUN_STATE.json',dict(pass_number=passno,contents_completed=i+1,model_forwards=nf))
    assert nf==nh==24 and parity==8 and len(replay)==24
    assert all(torch.equal(t.cpu(),before[n]) for n,t in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
    assert images|sources|{checkpoint}=={Path(p) for p in opened}
    save(root/'private'/f'REPLAY{passno}.json',replay);save(root/'private'/f'DIAGNOSTICS{passno}.json',diagnostics)
    if passno==2:assert diagnostics==read(root/'private/DIAGNOSTICS1.json')
    grouped={}
    for arm in ('SOURCE','N','R'):
        d=[r for r in diagnostics if r['arm']==arm]
        grouped[arm]=dict(native_positive_fraction_mean=np.mean([r['native_positive_fraction'] for r in d],axis=0).tolist(),mapped_positive_fraction_mean=np.mean([r['mapped_positive_fraction'] for r in d],axis=0).tolist(),
           native_empty_heads=np.sum([r['native_empty'] for r in d],axis=0).astype(int).tolist(),mapped_empty_heads=np.sum([r['mapped_empty'] for r in d],axis=0).astype(int).tolist(),
           logit_min=min(r['logit_min'] for r in d),logit_max=max(r['logit_max'] for r in d),logit_mean=float(np.mean([r['logit_mean'] for r in d])))
    save(root/'public'/f'GENERATION{passno}.json',dict(status='COMPLETE',source_sha=c['source_sha'],pass_number=passno,model_forwards=nf,head_checks=nh,source_bits_checks=parity,
        registered_rgb_files=8,new_decode=True,label_reads=0,utility_or_prior_scores_reads=0,access_negative_checks=len(denied),model_state_immutable=True,no_gradients=True,updates=0,
        raw_input_logit_feature_digests=24,independent_process_exact_replays=24 if passno==2 else 0,exact_masks_compared=24 if passno==2 else 0,
        common_clamp=True,clipped_channel_elements=clip,elements_per_native_branch=8*3*800*800,physical_gpu_index=1,physical_gpu_uuid_verified=True,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),norm_layer_count=len(norms),norm_classes=sorted({v['type'] for v in norms}),
        norm_training_true=sum(v['training'] for v in norms),norm_track_running_stats_false=sum(v['track_running_stats'] is False for v in norms),
        norm_track_running_stats_true=sum(v['track_running_stats'] is True for v in norms),diagnostics=grouped))

def summary(v):
    a=np.array([np.nan if x is None else x for x in v],float);good=np.isfinite(a)
    result=dict(total_contents=8,defined_contents=int(good.sum()),mean=float(a[good].mean()) if good.any() else None,ci95=None)
    if good.all():result['ci95']=[float(x) for x in np.percentile(a[INDEX].mean(1),[2.5,97.5])]
    return result

def contrast(a,b):
    v=[None if x is None or y is None else x-y for x,y in zip(a,b)];r=summary(v)
    if r['defined_contents']==8:
        z=np.array(v);loo=(z.sum()-z)/7
        r.update(wins=int((z>0).sum()),ties=int((z==0).sum()),losses=int((z<0).sum()),worst_difference=float(z.min()),leave_one_out_mean_range=[float(loo.min()),float(loo.max())])
    return r

def ratio(a,b):
    if any(v is None for v in list(a)+list(b)):return dict(defined_contents=sum(x is not None and y is not None for x,y in zip(a,b)),total_contents=8,mean_ratio=None,ci95=None)
    a,b=np.array(a),np.array(b);den=b[INDEX].mean(1)
    if b.mean()<=0 or (den<=0).any():return dict(total_contents=8,defined_contents=8,mean_ratio=None,ci95=None)
    return dict(total_contents=8,defined_contents=8,mean_ratio=float(a.mean()/b.mean()),ci95=np.percentile(a[INDEX].mean(1)/den,[2.5,97.5]).tolist())

def score(c):
    from runtime import identity
    from metric import metric
    root,old=Path(c['root']),Path(c['sealed_root'])
    for name in ('worker1','worker2'):
        rec=read(root/'processes'/f'{name}.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    gens=[read(root/'public'/f'GENERATION{j}.json') for j in (1,2)];assert sum(g['model_forwards'] for g in gens)==48 and gens[1]['independent_process_exact_replays']==24
    assert read(root/'public/CHECK.json')['status']=='PASS'
    panel=read(old/'scorer/panel.json');assert len(panel)==8
    registered={r['mask_path']:r['expected'] for r in read(old/'scorer/mask_identity.private.json')};assert set(registered)=={p['mask_path'] for p in panel}
    vals={};detail=[];coverage={a:dict(evaluated_masks=0,surface_defined_heads=0,vcdr_defined=0,empty_heads=0,nesting_pixels=0) for a in ARMS};structure={a:[] for a in ARMS}
    for i,p in enumerate(panel):
        raw=Path(p['mask_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==registered[p['mask_path']]
        with Image.open(io.BytesIO(raw)) as im:
            assert list(im.size)==p['image_size'];yy=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((yy<255,yy==0));source=unpack(root/'private/predictions'/f'{i}_SOURCE.bits')
        for arm in ARMS:
            pred=unpack(old/'inputs'/f'{i}_0_TOPO_SMOOTH.bits') if arm=='B' else unpack(root/'private/predictions'/f'{i}_{arm}.bits')
            m,v=metric(pred,gt,source);row={}
            for j,name in enumerate(('OD','OC')):
                row[name+'_dice']=m[j]['dice_percent'];row[name+'_ASSD']=m[j]['assd_pixel'];row[name+'_HD95']=m[j]['hd95_pixel']
            row['vCDR_error']=v['vcdr_abs_error']
            for key,x in row.items():vals.setdefault((arm,key),[]).append(x)
            st=dict(empty=[bool(x['pred_empty']) for x in m],nesting=int((pred[1]&~pred[0]).sum()));structure[arm].append(st)
            cov=coverage[arm];cov['evaluated_masks']+=1;cov['surface_defined_heads']+=sum(x['assd_pixel'] is not None for x in m);cov['vcdr_defined']+=v['vcdr_abs_error'] is not None;cov['empty_heads']+=sum(st['empty']);cov['nesting_pixels']+=st['nesting']
            detail.append(dict(case=i,arm=arm,metrics=row,structure=st))
    comparisons=[('N','R'),('N','SOURCE'),('N','B'),('R','SOURCE'),('R','B')]
    contrasts=[dict(comparison=a+'-'+b,metric=k,**contrast(vals[a,k],vals[b,k])) for a,b in comparisons for k in ('OC_dice','OD_dice','OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR_error')]
    lookup={(r['comparison'],r['metric']):r for r in contrasts}
    obs=lookup['N-R','OC_dice'];ns=lookup['N-SOURCE','OC_dice'];nb=lookup['N-B','OC_dice']
    pos=lambda r:r['ci95'] is not None and r['ci95'][0]>0
    boundary={k:ratio(vals['N',k],vals['B',k]) for k in ('OD_ASSD','OC_ASSD','OD_HD95','OC_HD95')}
    safety={}
    safety['complete_boundaries_vcdr']=coverage['N']['surface_defined_heads']==16 and coverage['N']['vcdr_defined']==8
    safety['no_new_empty_or_nesting']=all(not any(n and not b for n,b in zip(structure['N'][i]['empty'],structure[a][i]['empty'])) and structure['N'][i]['nesting']<=structure[a][i]['nesting'] for a in ('SOURCE','B') for i in range(8))
    safety['od_mean_noninferiority']=all(lookup['N-'+a,'OD_dice']['ci95'] is not None and lookup['N-'+a,'OD_dice']['ci95'][0]>=-.20 for a in ('SOURCE','B'))
    safety['worst_head_content_loss_le1pp']=all(lookup['N-B',k]['worst_difference']>=-1 for k in ('OD_dice','OC_dice'))
    safety['boundary_ratio_ci_upper_le1_05']=all(v['ci95'] is not None and v['ci95'][1]<=1.05 for v in boundary.values())
    vcdr=lookup['N-B','vCDR_error'];safety['vcdr_delta_ci_upper_le_005']=vcdr['ci95'] is not None and vcdr['ci95'][1]<=.005
    gates=dict(G0_engineering=True,G1_observation=pos(obs),G2_competitiveness=pos(ns) and pos(nb) and nb['mean']>=.30,G3_damage_constraints=all(safety.values()))
    save(root/'private/PER_CONTENT_METRICS.json',detail)
    save(root/'public/RESULTS.json',dict(status='COMPLETE',source_sha=c['source_sha'],primary='N-R:OC_dice',contents=8,conditions=['identity'],seed=SEED,bootstrap=10000,
        absolute=[dict(arm=a,metric=k,**summary(v)) for (a,k),v in vals.items()],contrasts=contrasts,coverage=coverage,boundary_ratios=boundary,damage_subgates=safety,gates=gates,
        fixed_static_candidate_qualified=all(gates.values()),observation_signal_retained=gates['G1_observation'],independent_patient_or_clinical_evidence=False))
    save(root/'public/EVALUATION_COMPLETE.json',dict(status='COMPLETE',registered_gt_files=8,gt_digest_checks=8,joint_masks=32,both_workers_exit0_and_identity_retired=True,
        model_forwards_total=48,independent_process_replay_views=24,gt_feedback_to_generation=False))

def run(c):
    mode=os.environ['AUDIT_MODE']
    if mode.startswith('worker'):worker(c,int(mode[-1]))
    else:globals()[mode](c)
