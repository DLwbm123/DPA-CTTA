"""Fixed factorial diagnosis; prediction generation cannot read labels."""
import os,sys,io,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from helpers import read,write,unpack,appearances,aggregate,audit
ARMS=('PERSIST','RESET')
DYNAMIC=('PERSIST_PRE','PERSIST_POST','RESET_PRE','RESET_POST')
READOUTS=('SOURCE',)+DYNAMIC+('B',)

def check(c):
    from engine import check as operator_check
    operator_check()
    root=Path(c['root']);panel=read(Path(c['sealed_root'])/'private/panel.json')
    registered={r['image_relative'] for r in c['rgb_identity']}
    assert len(panel)==8 and all(sum(str(p['image_path']).endswith(r) for r in registered)==1 for p in panel)
    write(root/'private/panel.json',panel)
    write(root/'public/CHECK.json',dict(status='PASS',registered_contents=8,model_calls=0,gt_reads=0,buffer_restore=True))

def worker(c):
    import torch
    from engine import Arm,clone_cpu,equal_tree,seeded,uncertainty
    root=Path(c['root']);old=Path(c['sealed_root']);prior=Path(c['prior_root']);mode=os.environ['AUDIT_MODE']
    panel=read(root/'private/panel.json')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    negative=audit(c,panel)
    sys.path.insert(0,str(old/'code/src'))
    from dpa_ctta.integrations.ctta_suite import build_reference_model
    seeded(0);source,_=build_reference_model('fundus')
    source.load_state_dict(torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True),strict=True)
    torch.cuda.set_per_process_memory_fraction(min(.8,10*1024**3/torch.cuda.get_device_properties(0).total_memory))
    source.cuda().eval().requires_grad_(False);original=clone_cpu(source.state_dict())
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True)
    mine=[v for v in gpu.splitlines() if v.split(',')[0].strip()==str(os.getpid())]
    assert len(mine)==1 and c['gpu_uuid'] in mine[0] and 'wangbomin' not in mine[0]
    out=root/'private/measurement';out.mkdir(exist_ok=True)
    meter=dict(forward=0,backward=0,updates=0);completed=changed=total=0;begin=time.perf_counter()
    try:
        for order,sequence in enumerate((list(range(8)),list(reversed(range(8))))):
            arms=[Arm(source,'BN_ONLY',meter) for n in ARMS]
            for pos,i in enumerate(sequence):
                item=panel[i]
                with Image.open(io.BytesIO(Path(item['image_path']).read_bytes())) as im:
                    assert list(im.size)==item['image_size']
                    rgb=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),np.float32)
                    image=torch.from_numpy(rgb.transpose(2,0,1).copy()/255)[None]
                for k,view in enumerate(appearances(image)):
                    input_begin=time.perf_counter();assert view.max()>view.min()
                    x=((view-view.min())/(view.max()-view.min())).cuda()
                    features=[];handle=source.up5.register_forward_hook(lambda mod,args,out:features.append(out))
                    try:
                        with torch.no_grad():meter['forward']+=1;z=source(x)[0]
                    finally:handle.remove()
                    source_logits=z.detach().cpu().numpy()[0]
                    assert np.array_equal(source_logits>=0,unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits'))
                    predictions=[];losses=[]
                    with np.load(prior/'private/measurement'/f'{order}_{i}_{k}.npz') as previous:
                        prior_bn=previous['logits'][4:6]
                    for index,arm in enumerate(arms):
                        if index==1:arm.model.load_state_dict(source.state_dict(),strict=True)
                        pre,post,loss,parity=arm.run(x,z.detach(),features[0].detach(),order*1000+pos*4+k)
                        pair=[pre.cpu().numpy()[0],post.cpu().numpy()[0]]
                        if index==0:assert np.array_equal(np.stack(pair),prior_bn),'A04 BN replay differs'
                        else:assert np.array_equal(pair[0],source_logits),'reset pre differs from source'
                        assert all(torch.equal(a,b) for a,b in zip(arm.params,source.parameters()))
                        assert all(a.grad is None for a in arm.params)
                        predictions.extend(pair);losses.append(loss)
                    if pos==k==0:assert np.array_equal(predictions[1],predictions[3])
                    logits=np.stack(predictions)
                    assert np.isfinite(logits).all() and np.isfinite(losses).all()
                    changed+=int((np.abs(logits[1]-logits[3])>1e-3).sum());total+=logits[1].size
                    payload=dict(logits=logits,masks=np.packbits(logits>=0),losses=np.asarray(losses))
                    path=out/f'{order}_{i}_{k}.npz'
                    if mode=='worker1':np.savez_compressed(path,**payload)
                    else:
                        with np.load(path) as previous:
                            assert set(previous.files)==set(payload)
                            for name,value in payload.items():assert np.array_equal(previous[name],value),name
                    completed+=1
                    write(root/'RUN_STATE.json',dict(stage=mode,completed_inputs=completed,counts=meter))
                    if mode=='worker1' and completed==1:
                        seconds=time.perf_counter()-input_begin
                        elapsed=time.clock_gettime(time.CLOCK_BOOTTIME)-c['origin_boottime']
                        estimate=seconds*127*1.5+elapsed+600
                        write(root/'public/COST_ADMISSION.json',dict(first_input_seconds=seconds,projected_total_seconds=estimate,passed=estimate<=3600))
                        assert estimate<=3600,'full controls/replay exceed budget'
                    del x,z,features,pre,post,parity,predictions,logits,payload
            del arms
        assert meter==dict(forward=448,backward=0,updates=0) and completed==64
        assert equal_tree(source.state_dict(),original)
        write(root/'public'/f'{mode.upper()}.json',dict(status='PASS',inputs=completed,counts=meter,source_parity=64,persistent_bn_exact_prior=64,semantic_changed_fraction=changed/total,independent_exact_replay=mode=='worker2',label_reads=0,negative_checks=negative))
    finally:
        write(root/'public'/f'{mode.upper()}_COUNTS.json',dict(counts=meter,completed_inputs=completed,seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))

def score(c):
    from runtime import identity
    from metric import metric
    root=Path(c['root']);old=Path(c['sealed_root'])
    for stage in ('worker1','worker2'):
        rec=read(root/'processes'/f'{stage}.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    assert read(root/'public/WORKER2.json')['independent_exact_replay']
    panel=read(old/'scorer/panel.json');registered={r['mask_path'] for r in read(old/'scorer/mask_identity.private.json')}
    assert len(panel)==8 and {p['mask_path'] for p in panel}==registered
    keys=('OD','OC','OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR','vCDR_signed','OC_area_ratio','OD_FP','OC_FP','OD_FN','OC_FN')
    values={a:{key:np.full((8,8),np.nan) for key in keys} for a in READOUTS}
    empty={a:0 for a in READOUTS};nest={a:0 for a in READOUTS}
    for i,p in enumerate(panel):
        with Image.open(p['mask_path']) as im:
            assert list(im.size)==p['image_size'];yy=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((yy<255,yy==0))
        for order in range(2):
            for k in range(4):
                col=order*4+k;src=unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits');b=unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits')
                with np.load(root/'private/measurement'/f'{order}_{i}_{k}.npz') as z:dynamic=np.unpackbits(z['masks']).reshape(4,2,512,512).astype(bool)
                for a,m in zip(READOUTS,[src,*dynamic,b]):
                    met,rat=metric(m,gt,b)
                    for j,h in enumerate(('OD','OC')):
                        values[a][h][i,col]=met[j]['dice_percent'];empty[a]+=met[j]['pred_empty']
                        for suf,key in (('_ASSD','assd_pixel'),('_HD95','hd95_pixel')):values[a][h+suf][i,col]=met[j][key] if met[j][key] is not None else np.nan
                        values[a][h+'_FP'][i,col]=(m[j]&~gt[j]).sum();values[a][h+'_FN'][i,col]=(~m[j]&gt[j]).sum()
                    values[a]['vCDR'][i,col]=rat['vcdr_abs_error'] if rat['vcdr_abs_error'] is not None else np.nan
                    values[a]['vCDR_signed'][i,col]=rat['predicted_vcdr']-rat['true_vcdr'] if rat['predicted_vcdr'] is not None else np.nan
                    values[a]['OC_area_ratio'][i,col]=m[1].sum()/gt[1].sum() if gt[1].any() else np.nan
                    nest[a]+=int((m[1]&~m[0]).sum())
    pairs=[('PERSIST_POST','RESET_POST'),('PERSIST_POST','SOURCE'),('RESET_POST','SOURCE'),('RESET_POST','B')]
    contrasts={a+' minus '+b:{key:aggregate(values[a][key]-values[b][key]) for key in keys} for a,b in pairs}
    slices={}
    for a,b in pairs:
        slices[a+' minus '+b]={}
        for name,cols in [('first',[0,4]),('later',[1,2,3,5,6,7])]:
            q=aggregate(np.repeat((values[a]['OC']-values[b]['OC'])[:,cols].mean(1)[:,None],4,1));q['defined_inputs']=8*len(cols);slices[a+' minus '+b][name]=q
    write(root/'private/PER_INPUT_VALUES.json',{a:{k:np.where(np.isfinite(v),v,None).tolist() for k,v in d.items()} for a,d in values.items()})
    write(root/'public/RESULTS.json',dict(status='COMPLETE',diagnostic_only=True,method_promoted=False,contrasts=contrasts,slices=slices,metrics={a:{k:aggregate(v) for k,v in d.items()} for a,d in values.items()},empty_heads=empty,nesting_pixels=nest,registered_contents=8,inputs=64))

def run(c):
    mode=os.environ['AUDIT_MODE']
    if mode=='check':check(c)
    elif mode in ('worker1','worker2'):worker(c)
    elif mode=='score':score(c)
    else:raise ValueError(mode)
