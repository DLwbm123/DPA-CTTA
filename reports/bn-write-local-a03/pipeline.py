"""Same-prestate local BN-write intervention; no parameter gradients."""
import os,sys,io,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from helpers import read,write,unpack,appearances,aggregate,audit
ARMS=('SOURCE','HOLD','RAW','FFT','MASK','FULL','B')

def check(c):
    from engine import check as operator_check
    operator_check();root=Path(c['root']);panel=read(Path(c['sealed_root'])/'private/panel.json')
    registered={r['image_relative'] for r in c['rgb_identity']}
    assert len(panel)==8 and all(sum(str(p['image_path']).endswith(r) for r in registered)==1 for p in panel)
    write(root/'private/panel.json',panel)
    write(root/'public/CHECK.json',dict(status='PASS',model_calls=0,gt_reads=0,buffer_restore=True))

def worker(c):
    import torch
    from engine import Arm,clone_cpu,equal_tree,seeded
    root=Path(c['root']);old=Path(c['sealed_root']);prior=Path(c['prior_root']);mode=os.environ['AUDIT_MODE']
    panel=read(root/'private/panel.json');torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    negative=audit(c,panel);sys.path.insert(0,str(old/'code/src'))
    from dpa_ctta.integrations.ctta_suite import build_reference_model
    seeded(0);source,_=build_reference_model('fundus')
    source.load_state_dict(torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True),strict=True)
    torch.cuda.set_per_process_memory_fraction(min(.8,10*1024**3/torch.cuda.get_device_properties(0).total_memory))
    source.cuda().eval().requires_grad_(False);original=clone_cpu(source.state_dict())
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True)
    mine=[v for v in gpu.splitlines() if v.split(',')[0].strip()==str(os.getpid())]
    assert len(mine)==1 and c['gpu_uuid'] in mine[0] and 'wangbomin' not in mine[0]
    out=root/'private/measurement';out.mkdir(exist_ok=True)
    meter=dict(forward=0,backward=0,updates=0);completed=selected_count=0;begin=time.perf_counter()
    try:
        with torch.no_grad():
            for order,sequence in enumerate((list(range(8)),list(reversed(range(8))))):
                arm=Arm(source,meter)
                for pos,i in enumerate(sequence):
                    p=panel[i]
                    with Image.open(io.BytesIO(Path(p['image_path']).read_bytes())) as im:
                        assert list(im.size)==p['image_size'];rgb=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),np.float32)
                        image=torch.from_numpy(rgb.transpose(2,0,1).copy()/255)[None]
                    for k,view in enumerate(appearances(image)):
                        start=time.perf_counter();x=((view-view.min())/(view.max()-view.min())).cuda()
                        meter['forward']+=1;z=source(x)[0]
                        assert np.array_equal(z.cpu().numpy()[0]>=0,unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits'))
                        selected=k==i%4
                        anchor,branches=arm.run(x,z,order*1000+pos*4+k,selected)
                        anchor_np=anchor.cpu().numpy()[0]
                        with np.load(prior/'private/measurement'/f'{order}_{i}_{k}.npz') as previous:
                            assert np.array_equal(anchor_np,previous['logits'][3]),'anchor != A02 N01 POST'
                            if selected:assert np.array_equal(branches[0].cpu().numpy()[0],previous['logits'][2]),'HOLD != A02 N01 PRE'
                        assert all(torch.equal(p,q) and p.grad is None for p,q in zip(arm.model.parameters(),source.parameters()))
                        branch_np=np.stack([v.cpu().numpy()[0] for v in branches]) if selected else np.empty((0,2,512,512),np.float32)
                        payload=dict(anchor=anchor_np,branches=branch_np)
                        assert all(np.isfinite(v).all() for v in payload.values())
                        path=out/f'{order}_{i}_{k}.npz'
                        if mode=='worker1':np.savez_compressed(path,**payload)
                        else:
                            with np.load(path) as previous:
                                for name,value in payload.items():assert np.array_equal(previous[name],value),name
                        completed+=1;selected_count+=int(selected)
                        write(root/'RUN_STATE.json',dict(stage=mode,completed_inputs=completed,selected=selected_count,counts=meter))
                        if mode=='worker1' and completed==1:
                            seconds=time.perf_counter()-start
                            estimate=seconds*127*1.5+(time.clock_gettime(time.CLOCK_BOOTTIME)-c['origin_boottime'])+300
                            write(root/'public/COST_ADMISSION.json',dict(first_selected_input_seconds=seconds,projected_total_seconds=estimate,passed=estimate<=3600))
                            assert estimate<=3600
                        del x,z,anchor,branches,branch_np,payload
                del arm
        assert meter==dict(forward=304,backward=0,updates=0) and completed==64 and selected_count==16
        assert equal_tree(source.state_dict(),original)
        write(root/'public'/f'{mode.upper()}.json',dict(status='PASS',counts=meter,completed_inputs=64,selected_positions=16,anchor_exact_prior=64,hold_exact_prior=16,parameter_immutable=True,independent_exact_replay=mode=='worker2',negative_checks=negative,label_reads=0))
    finally:write(root/'public'/f'{mode.upper()}_COUNTS.json',dict(counts=meter,completed_inputs=completed,seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))

def paired(a):
    # Equal two-order observations per content; preserve eight content bootstrap units.
    z=aggregate(np.repeat(a,4,axis=1));z['defined_inputs']=int(np.isfinite(a).sum());return z

def score(c):
    from runtime import identity
    from metric import metric
    root=Path(c['root']);old=Path(c['sealed_root'])
    for stage in ('worker1','worker2'):
        rec=read(root/'processes'/f'{stage}.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    assert read(root/'public/WORKER2.json')['independent_exact_replay']
    panel=read(old/'scorer/panel.json');registered={r['mask_path'] for r in read(old/'scorer/mask_identity.private.json')}
    assert len(panel)==8 and {p['mask_path'] for p in panel}==registered
    keys=('OD','OC','OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR','vCDR_signed','OD_area_ratio','OC_area_ratio','OD_fp_fraction','OC_fp_fraction','OD_fn_fraction','OC_fn_fraction')
    values={a:{key:np.full((8,2),np.nan) for key in keys} for a in ARMS};rows=[];empty={a:0 for a in ARMS};nest={a:0 for a in ARMS}
    for i,p in enumerate(panel):
        k=i%4
        with Image.open(p['mask_path']) as im:
            assert list(im.size)==p['image_size'];yy=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((yy<255,yy==0))
        for order in range(2):
            src=unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits');b=unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits')
            with np.load(root/'private/measurement'/f'{order}_{i}_{k}.npz') as z:branches=z['branches']>=0
            assert branches.shape==(5,2,512,512)
            for a,m in zip(ARMS,[src,*branches,b]):
                met,rat=metric(m,gt,b);empty[a]+=sum(v['pred_empty'] for v in met);nest[a]+=int((m[1]&~m[0]).sum())
                row=dict(content_index=i,order=order,appearance=k,arm=a,metrics={})
                for j,h in enumerate(('OD','OC')):
                    row['metrics'].update({h:met[j]['dice_percent'],h+'_ASSD':met[j]['assd_pixel'],h+'_HD95':met[j]['hd95_pixel'],h+'_area_ratio':float(m[j].sum()/gt[j].sum()) if gt[j].any() else None,h+'_fp_fraction':float((m[j]&~gt[j]).mean()),h+'_fn_fraction':float((~m[j]&gt[j]).mean())})
                row['metrics']['vCDR']=rat['vcdr_abs_error'];row['metrics']['vCDR_signed']=rat['predicted_vcdr']-rat['true_vcdr'] if rat['predicted_vcdr'] is not None and rat['true_vcdr'] is not None else None
                for key,v in row['metrics'].items():values[a][key][i,order]=np.nan if v is None else v
                rows.append(row)
    effects={h:{name:paired(v) for name,v in {'raw_write':values['RAW'][h]-values['HOLD'][h],'fft_given_raw':values['FFT'][h]-values['RAW'][h],'mask_given_raw':values['MASK'][h]-values['RAW'][h],'interaction':values['FULL'][h]-values['FFT'][h]-values['MASK'][h]+values['RAW'][h],'mask_given_fft':values['FULL'][h]-values['FFT'][h],'fft_given_mask':values['FULL'][h]-values['MASK'][h],'full_write':values['FULL'][h]-values['HOLD'][h]}.items()} for h in ('OD','OC')}
    write(root/'private/PER_INPUT.json',rows)
    write(root/'public/RESULTS.json',dict(status='COMPLETE_DIAGNOSTIC',method_promoted=False,metrics={a:{key:paired(v) for key,v in data.items()} for a,data in values.items()},effects=effects,empty_heads=empty,nesting_pixels=nest,contrasts={a:{base:{h:paired(values[a][h]-values[base][h]) for h in ('OD','OC')} for base in ('SOURCE','B','HOLD','FULL')} for a in ARMS},scope='16 fixed local positions from 8 development contents; same N01 prestate; no full-trajectory repair or independent validation claim.'))

def run(c):
    mode=os.environ['AUDIT_MODE']
    if mode=='check':check(c)
    elif mode in ('worker1','worker2'):worker(c)
    elif mode=='score':score(c)
    else:raise ValueError(mode)
