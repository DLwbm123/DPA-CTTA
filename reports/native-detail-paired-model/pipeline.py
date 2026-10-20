"""Frozen observation-to-decision test; generation and GT scoring are separate processes."""
import os
import sys
import io
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

ARMS=('SOURCE','B','N','R','L')
SEED=20261018

def save(p,v):
    Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def read(p):
    return json.loads(Path(p).read_text())

def unpack(p):
    b=Path(p).read_bytes();assert len(b)==65536
    return np.unpackbits(np.frombuffer(b,np.uint8)).reshape(2,512,512).astype(bool)

def start(v):
    t=int(np.floor((v.min()+v.max()+1)/2))-128
    return int(np.clip(16*np.floor((np.clip(t,0,256)+8)/16),0,256))

def stitch(s,local,y,x):
    assert s.shape==(2,512,512) and local.shape==(2,256,256)
    p=s.copy();p[:,y:y+256,x:x+256]=local
    return p

def check(c):
    from observation import check as geometry
    geometry(c)
    s=np.zeros((2,512,512),bool)
    for y,x in ((0,0),(16,128),(256,256)):
        p=stitch(s,np.ones((2,256,256),bool),y,x)
        assert int(p.sum())==2*256**2
        assert p[:,y:y+256,x:x+256].all()
    assert start(np.array([0,15]))==0 and start(np.array([496,511]))==256
    # Zero uses the same >=0 head threshold as the sealed SOURCE.
    assert np.array_equal(np.array([-1.,0.,1.])>=0,[False,True,True])
    save(Path(c['root'])/'public/CHECK.json',dict(status='PASS',stitch_support_checks=3,source_threshold_zero_is_positive=True))

def worker(c):
    import torch
    import torch.nn.functional as F
    from observation import views
    root,old=Path(c['root']),Path(c['sealed_root'])
    panel=read(old/'private/panel.json');assert len(panel)==8 and all('mask_path' not in p for p in panel)
    images={Path(p['image_path']).resolve() for p in panel}
    sources={(old/'inputs'/f'{i}_0_SOURCE.bits').resolve() for i in range(8)}
    checkpoint=Path(c['checkpoint_path']).resolve();opened=set();denied=[]
    def audit(event,args):
        if event=='socket.connect':raise PermissionError('network denied')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        forbid=(p.is_relative_to(old) and not p.is_relative_to(old/'code') and p not in sources and p!=old/'private/panel.json')
        forbid|=p.is_relative_to(Path(c['source_root'])) and p not in images
        forbid|=p.is_relative_to(Path(c['model_root'])) and p!=checkpoint
        forbid|=p.name in ('evaluate.py','score.py','e8_score.py','signal_score.py')
        if forbid:
            denied.append(str(p));raise PermissionError('outside generation read set')
        if p in images or p in sources or p==checkpoint:opened.add(str(p))
    sys.addaudithook(audit)
    for p in (old/'scorer/panel.json',old/'private/scores.json'):
        try:p.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('access gate')
    sys.path.insert(0,str(old/'code/src'))
    from dpa_ctta.integrations.ctta_suite import build_reference_model
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.manual_seed(SEED);np.random.seed(SEED)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=build_reference_model('fundus');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.cuda().eval().requires_grad_(False)
    before={n:t.cpu().clone() for n,t in model.state_dict().items()}
    nf=nh=parity=replay=0;clip={n:0 for n in ('N','R','L')};details=[]
    with torch.no_grad():
        for i,p in enumerate(panel):
            with Image.open(p['image_path']) as im:
                assert im.size==(800,800) and im.mode=='RGB'
                native=torch.from_numpy(np.asarray(im,dtype=np.float32).transpose(2,0,1).copy()/255)[None]
                low=torch.from_numpy(np.asarray(im.resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32).transpose(2,0,1).copy()/255)[None]
            s=unpack(old/'inputs'/f'{i}_0_SOURCE.bits')
            yy,xx=np.where(s[0]);assert len(yy)>0
            y,x=start(yy),start(xx)
            lo=low.amin();span=low.amax()-lo;assert span>0
            branch=[(v-lo)/span for v in views(native,low,y,x)]
            for name,v in zip(('N','R','L'),branch):clip[name]+=int(((v<0)|(v>1)).sum())
            inputs=[(low-lo)/span]+[v.clamp(0,1) for v in branch]
            assert all(torch.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in inputs)
            first=[]
            for repeat in (False,True):
                for j,(name,v) in enumerate(zip(('SOURCE','N','R','L'),inputs)):
                    out=model(v.cuda());z,f=out[0],out[2];nf+=1
                    assert torch.equal(model.seg_head(f),z);nh+=1
                    zz,ff=z.cpu(),f.cpu()
                    if repeat:
                        assert torch.equal(zz,first[j][0]) and torch.equal(ff,first[j][1]);replay+=1
                    else:first.append((zz.clone(),ff.clone()))
                    if name=='SOURCE':
                        assert np.array_equal(zz[0].numpy()>=0,s);parity+=1
                        pred=s
                    else:
                        local=F.interpolate(zz,size=(256,256),mode='bilinear',align_corners=False)[0].numpy()>=0
                        pred=stitch(s,local,y,x)
                    out=root/'private/predictions'/f'{i}_{name}.bits'
                    packed=np.packbits(pred.reshape(-1)).tobytes()
                    if repeat:assert out.read_bytes()==packed
                    else:out.write_bytes(packed)
                    if nf==1:
                        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
                        own=[line for line in lines if line.split(',')[0].strip()==str(os.getpid())]
                        assert len(own)==1 and c['gpu_uuid'] in own[0] and not any(t in own[0].lower() for t in ('wangbomin','ctta','native'))
                        save(root/'private/GPU_VISIBLE.json',dict(status='PASS',line=own[0]))
            details.append(dict(case=i,start_private=[y,x]))
            save(root/'RUN_STATE.json',dict(status='GENERATING',completed_contents=i+1,model_forwards=nf))
    assert nf==nh==64 and replay==32 and parity==16
    assert all(torch.equal(t.cpu(),before[n]) for n,t in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert images|sources|{checkpoint}=={Path(p) for p in opened}
    save(root/'private/ROI.json',details)
    save(root/'public/GENERATION.json',dict(status='COMPLETE',source_sha=c['source_sha'],contents=8,conditions=['identity'],model_forwards=nf,
         head_reconstruction_checks=nh,source_bits_checks=parity,exact_logit_feature_replays=replay,exact_mask_replays=32,
         model_state_immutable=True,no_gradients=True,updates=0,registered_rgb_files=8,label_reads=0,prior_utility_or_score_reads=0,
         checkpoint_files_read=1,access_negative_checks=len(denied),common_clamp=True,clipped_channel_elements=clip,
         elements_per_branch=8*3*512*512,physical_gpu_index=1,physical_gpu_uuid_verified=True,
         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),native_sensor_resolution_or_benefit_established=False))

def summary(v):
    a=np.array([np.nan if x is None else x for x in v],float);good=np.isfinite(a)
    result=dict(total_contents=8,defined_contents=int(good.sum()),mean=float(a[good].mean()) if good.any() else None,ci95=None)
    if good.all():
        ix=np.random.default_rng(SEED).integers(0,8,(10000,8))
        result['ci95']=[float(x) for x in np.percentile(a[ix].mean(1),[2.5,97.5])]
    return result

def score(c):
    from runtime import identity
    from metric import metric
    root,old=Path(c['root']),Path(c['sealed_root'])
    rec=read(root/'processes/worker.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    assert read(root/'public/GENERATION.json')['model_forwards']==64
    panel=read(old/'scorer/panel.json');assert len(panel)==8
    registered={r['mask_path']:r['expected'] for r in read(old/'scorer/mask_identity.private.json')}
    assert {p['mask_path'] for p in panel}==set(registered)
    values={};coverage={a:dict(evaluated_masks=0,surface_defined_heads=0,vcdr_defined=0,empty_heads=0,nesting_pixels=0) for a in ARMS}
    private=[]
    for i,p in enumerate(panel):
        raw=Path(p['mask_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==registered[p['mask_path']]
        with Image.open(io.BytesIO(raw)) as im:
            assert list(im.size)==p['image_size']
            yy=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((yy<255,yy==0));source=unpack(root/'private/predictions'/f'{i}_SOURCE.bits')
        for arm in ARMS:
            pred=unpack(old/'inputs'/f'{i}_0_TOPO_SMOOTH.bits') if arm=='B' else unpack(root/'private/predictions'/f'{i}_{arm}.bits')
            m,v=metric(pred,gt,source)
            row={}
            for j,name in enumerate(('OD','OC')):
                row[name+'_dice']=m[j]['dice_percent'];row[name+'_ASSD']=m[j]['assd_pixel'];row[name+'_HD95']=m[j]['hd95_pixel']
            row['vCDR_error']=v['vcdr_abs_error']
            for key,x in row.items():values.setdefault((arm,key),[]).append(x)
            cov=coverage[arm];cov['evaluated_masks']+=1;cov['surface_defined_heads']+=sum(x['assd_pixel'] is not None for x in m)
            cov['vcdr_defined']+=v['vcdr_abs_error'] is not None;cov['empty_heads']+=sum(x['pred_empty'] for x in m);cov['nesting_pixels']+=int((pred[1]&~pred[0]).sum())
            private.append(dict(case=i,arm=arm,metrics=row))
    absolute=[dict(arm=a,metric=k,**summary(v)) for (a,k),v in values.items()]
    contrasts=[]
    for control in ('R','L','SOURCE','B'):
        for key in ('OC_dice','OD_dice','OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR_error'):
            d=[None if a is None or b is None else a-b for a,b in zip(values['N',key],values[control,key])]
            contrasts.append(dict(comparison='N-'+control,metric=key,**summary(d)))
    gates={r['comparison']:r['ci95'] is not None and r['ci95'][0]>0 for r in contrasts if r['metric']=='OC_dice'}
    save(root/'private/PER_CONTENT_METRICS.json',private)
    save(root/'public/RESULTS.json',dict(status='COMPLETE',source_sha=c['source_sha'],primary='N-R:OC_dice',bootstrap=10000,seed=SEED,
        contents=8,conditions=['identity'],absolute=absolute,contrasts=contrasts,coverage=coverage,
        gates=gates,primary_passed=gates['N-R'],observation_use_qualified=all(gates.values()),independent_patient_or_clinical_evidence=False))
    save(root/'public/EVALUATION_COMPLETE.json',dict(status='COMPLETE',registered_gt_files=8,gt_manifest_digest_checks=8,joint_masks=40,
         worker_exit0_and_exact_identity_retired=True,gt_feedback_to_worker=False,static_control_masks=8))

def run(c):
    globals()[os.environ['AUDIT_MODE']](c)
