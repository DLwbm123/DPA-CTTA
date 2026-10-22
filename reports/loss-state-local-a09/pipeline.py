"""Fixed-position readouts; the actual learning trajectory remains A08."""
import os,sys,io,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from helpers import read,write,unpack,appearances,aggregate,audit
READOUTS=('FULL','HOLD','SEG','FFT','ZERO','SOURCE','B','RESET','BN_ONLY')


def check(c):
    from engine import check as operator_check
    operator_check()
    root=Path(c['root']);panel=read(Path(c['sealed_root'])/'private/panel.json')
    registered={r['image_relative'] for r in c['rgb_identity']}
    assert len(panel)==8 and all(sum(str(p['image_path']).endswith(r) for r in registered)==1 for p in panel)
    assert [sum(i%4==k for i in range(8)) for k in range(4)]==[2]*4
    write(root/'private/panel.json',panel)
    write(root/'public/CHECK.json',dict(status='PASS',registered_contents=8,selected_positions=16,model_calls=0,gt_reads=0))


def snapshot(arm):
    from engine import clone_cpu
    return dict(model=clone_cpu(arm.model.state_dict()),optimizer=clone_cpu(arm.opt.state_dict()),style=clone_cpu(arm.style.items),mode=arm.model.training,grads=[None if p.grad is None else p.grad.detach().cpu().clone() for p in arm.params])


def restore(arm,state):
    from engine import clone_cpu
    arm.model.load_state_dict(state['model'],strict=True);arm.opt.load_state_dict(clone_cpu(state['optimizer']));arm.style.items=clone_cpu(state['style']);arm.model.train(state['mode'])
    for p,g in zip(arm.params,state['grads']):p.grad=None if g is None else g.to(p.device).clone()


def assert_state(arm,state):
    import torch
    from engine import equal_tree
    assert equal_tree(arm.model.state_dict(),state['model'])
    assert equal_tree(arm.opt.state_dict(),state['optimizer'])
    assert arm.style.items.keys()==state['style'].keys() and all(np.array_equal(v,state['style'][k]) for k,v in arm.style.items.items())
    assert arm.model.training==state['mode']
    assert all((p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad.cpu(),g)) for p,g in zip(arm.params,state['grads']))


def probes(arm,x,before,post,teacher,features,key,actual_loss):
    import torch
    after=snapshot(arm);observed=arm.buffers();actual_components=arm.last_components.copy()
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state())
    predictions=[post.cpu().numpy()[0]];norms=[];components=[];gradients={}
    try:
        restore(arm,before);arm.restore_buffers(observed);arm.model.eval()
        with torch.no_grad():predictions.append(arm.forward(x)[0].cpu().numpy()[0])
        for mode in ('FULL','SEG','FFT','ZERO'):
            restore(arm,before);arm.loss_mode=mode
            pre,q,loss,parity=arm.run(x,teacher,features,key)
            assert arm.last_components==actual_components,'branch changed training evidence'
            assert all((p.grad is None)==(g is None) for p,g in zip(arm.params,after['grads'])),'gradient participation changed'
            if mode=='FULL':
                assert torch.equal(q,post) and loss==actual_loss
                assert_state(arm,after)
            else:predictions.append(q.cpu().numpy()[0])
            flat=torch.cat([p.grad[m].detach().cpu().double() if p.grad is not None else torch.zeros(int(m.sum()),dtype=torch.float64) for p,m in zip(arm.params,arm.support)])
            gradient_norm=float(flat.norm())
            delta=sum(float(((p.detach().cpu()-before['model'][n]).double().square()).sum()) for n,p in arm.model.named_parameters())**.5
            norms.append([gradient_norm,delta]);components.append(arm.last_components.copy())
            if mode in ('SEG','FFT'):gradients[mode]=flat
            if mode=='ZERO':assert gradient_norm==0 and loss==0
        denominator=float(gradients['SEG'].norm()*gradients['FFT'].norm())
        cosine=float(torch.dot(gradients['SEG'],gradients['FFT']))/denominator if denominator else 0.
        outputs=np.stack(predictions)
    finally:
        restore(arm,after);arm.loss_mode='FULL';arm.last_components=actual_components
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2])
    assert_state(arm,after)
    assert torch.equal(torch.get_rng_state(),rng[0]) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng[1]))
    now=np.random.get_state();assert now[0]==rng[2][0] and np.array_equal(now[1],rng[2][1]) and now[2:]==rng[2][2:]
    return outputs,np.asarray(components),np.asarray(norms),np.asarray(cosine)


def worker(c):
    import torch
    from engine import Arm,clone_cpu,equal_tree,seeded
    root=Path(c['root']);old=Path(c['sealed_root']);prior=Path(c['prior_root']);mode=os.environ['AUDIT_MODE']
    panel=read(root/'private/panel.json')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
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
    meter=dict(forward=0,backward=0,updates=0);completed=selected=0;begin=time.perf_counter()
    try:
        for order,sequence in enumerate((list(range(8)),list(reversed(range(8))))):
            arm=Arm(source,'REPAIR',meter)
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
                    assert np.array_equal(z.cpu().numpy()[0]>=0,unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits'))
                    chosen=k==i%4;before=snapshot(arm) if chosen else None
                    key=order*1000+pos*4+k
                    pre,post,loss,parity=arm.run(x,z.detach(),features[0].detach(),key)
                    with np.load(prior/'private/measurement'/f'{order}_{i}_{k}.npz') as previous:
                        assert np.array_equal(pre.cpu().numpy()[0],previous['logits'][0])
                        assert np.array_equal(post.cpu().numpy()[0],previous['logits'][1])
                        assert loss==previous['losses'][0]
                        assert np.array_equal(parity.cpu().numpy()[0],previous['logits'][8])
                    payload=dict(anchor=np.stack((pre.cpu().numpy()[0],post.cpu().numpy()[0])),loss=np.asarray(loss))
                    if chosen:
                        q,components,norms,cosine=probes(arm,x,before,post,z.detach(),features[0].detach(),key,loss);assert np.isfinite(q).all()
                        payload.update(logits=q,masks=np.packbits(q>=0),components=components,norms=norms,cosine=cosine);selected+=1
                    path=out/f'{order}_{i}_{k}.npz'
                    if mode=='worker1':np.savez_compressed(path,**payload)
                    else:
                        with np.load(path) as previous:
                            assert set(previous.files)==set(payload)
                            for name,value in payload.items():assert np.array_equal(previous[name],value),name
                    completed+=1
                    write(root/'RUN_STATE.json',dict(stage=mode,completed_inputs=completed,selected_positions=selected,counts=meter))
                    if mode=='worker1' and completed==1:
                        seconds=time.perf_counter()-input_begin;elapsed=time.clock_gettime(time.CLOCK_BOOTTIME)-c['origin_boottime']
                        estimate=seconds*(1440/27)*1.5+elapsed+300
                        write(root/'public/COST_ADMISSION.json',dict(first_input_seconds=seconds,projected_total_seconds=estimate,passed=estimate<=3600))
                        assert estimate<=3600,'full probes/replay exceed budget'
                    del x,z,features,pre,post,parity,payload,before
            del arm
        assert meter==dict(forward=720,backward=128,updates=128) and completed==64 and selected==16
        assert equal_tree(source.state_dict(),original)
        write(root/'public'/f'{mode.upper()}.json',dict(status='PASS',inputs=completed,selected_positions=selected,counts=meter,source_parity=64,a08_pre_post_loss_parity=64,a08_source_bn_output_parity=64,full_update_state_replay=16,zero_current_gradient=16,probe_full_state_restoration=16,independent_exact_replay=mode=='worker2',label_reads=0,negative_checks=negative))
    finally:
        write(root/'public'/f'{mode.upper()}_COUNTS.json',dict(counts=meter,completed_inputs=completed,selected_positions=selected,seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))


def score(c):
    from runtime import identity
    from metric import metric
    root=Path(c['root']);old=Path(c['sealed_root']);prior=Path(c['prior_root'])
    for stage in ('worker1','worker2'):
        rec=read(root/'processes'/f'{stage}.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    assert read(root/'public/WORKER2.json')['independent_exact_replay']
    panel=read(old/'scorer/panel.json');registered={r['mask_path'] for r in read(old/'scorer/mask_identity.private.json')}
    assert len(panel)==8 and {p['mask_path'] for p in panel}==registered
    keys=('OD','OC','OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR','vCDR_signed','OC_area_ratio','OD_FP','OC_FP','OD_FN','OC_FN')
    values={a:{key:np.full((8,2),np.nan) for key in keys} for a in READOUTS}
    empty={a:0 for a in READOUTS};nest={a:0 for a in READOUTS}
    for i,p in enumerate(panel):
        with Image.open(p['mask_path']) as im:
            assert list(im.size)==p['image_size'];yy=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((yy<255,yy==0));k=i%4
        for order in range(2):
            src=unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits');b=unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits')
            with np.load(root/'private/measurement'/f'{order}_{i}_{k}.npz') as z:dynamic=np.unpackbits(z['masks']).reshape(5,2,512,512).astype(bool)
            with np.load(prior/'private/measurement'/f'{order}_{i}_{k}.npz') as z:cached=np.unpackbits(z['masks']).reshape(9,2,512,512).astype(bool)
            for a,m in zip(READOUTS,[*dynamic,src,b,cached[2],cached[5]]):
                met,rat=metric(m,gt,b)
                for j,h in enumerate(('OD','OC')):
                    values[a][h][i,order]=met[j]['dice_percent'];empty[a]+=met[j]['pred_empty']
                    for suf,key in (('_ASSD','assd_pixel'),('_HD95','hd95_pixel')):values[a][h+suf][i,order]=met[j][key] if met[j][key] is not None else np.nan
                    values[a][h+'_FP'][i,order]=(m[j]&~gt[j]).sum();values[a][h+'_FN'][i,order]=(~m[j]&gt[j]).sum()
                values[a]['vCDR'][i,order]=rat['vcdr_abs_error'] if rat['vcdr_abs_error'] is not None else np.nan
                values[a]['vCDR_signed'][i,order]=rat['predicted_vcdr']-rat['true_vcdr'] if rat['predicted_vcdr'] is not None else np.nan
                values[a]['OC_area_ratio'][i,order]=m[1].sum()/gt[1].sum() if gt[1].any() else np.nan
                nest[a]+=int((m[1]&~m[0]).sum())
    def agg(v):
        # Collapse equal-weight orders before the existing content bootstrap.
        v=np.asarray(v);complete=np.isfinite(v).all(1);means=np.where(complete,v.mean(1),np.nan)
        z=aggregate(np.repeat(means[:,None],4,axis=1));z['defined_inputs']=int(np.isfinite(v).sum());return z
    pairs={'full_minus_hold':('FULL','HOLD'),'seg_minus_hold':('SEG','HOLD'),'fft_minus_hold':('FFT','HOLD'),'zero_minus_hold':('ZERO','HOLD'),'seg_minus_full':('SEG','FULL'),'fft_minus_full':('FFT','FULL'),'full_minus_zero':('FULL','ZERO'),'seg_minus_zero':('SEG','ZERO'),'fft_minus_zero':('FFT','ZERO')}
    contrasts={name:{key:agg(values[a][key]-values[b][key]) for key in keys} for name,(a,b) in pairs.items()}
    contrasts['interaction']={key:agg(values['FULL'][key]-values['SEG'][key]-values['FFT'][key]+values['ZERO'][key]) for key in keys}
    controls={a:{b:{key:agg(values[a][key]-values[b][key]) for key in keys} for b in ('SOURCE','B','RESET','BN_ONLY')} for a in READOUTS[:5]}
    write(root/'private/PER_INPUT_VALUES.json',{a:{k:np.where(np.isfinite(v),v,None).tolist() for k,v in d.items()} for a,d in values.items()})
    diagnostics=[]
    for i in range(8):
        for order in range(2):
            with np.load(root/'private/measurement'/f'{order}_{i}_{i%4}.npz') as q:diagnostics.append(dict(components=q['components'].tolist(),norms=q['norms'].tolist(),cosine=float(q['cosine'])))
    write(root/'private/UPDATE_DIAGNOSTICS.json',diagnostics)
    summary={key:np.asarray([d[key] for d in diagnostics]).mean(0).tolist() for key in ('components','norms','cosine')}
    write(root/'public/RESULTS.json',dict(status='COMPLETE_DIAGNOSTIC',method_promoted=False,update_diagnostics_mean=summary,component_order=['FULL','SEG','FFT','ZERO'],norm_columns=['gradient_on_support_l2','parameter_step_l2'],selected_positions=16,registered_contents=8,contrasts=contrasts,controls=controls,metrics={a:{key:agg(v) for key,v in d.items()} for a,d in values.items()},empty_heads=empty,nesting_pixels=nest,task_scope='Fixed k=i%4; equal content and order weighting. Descriptive development evidence, not patient validation or a qualification test.'))


def run(c):
    mode=os.environ['AUDIT_MODE']
    if mode=='check':check(c)
    elif mode in ('worker1','worker2'):worker(c)
    elif mode=='score':score(c)
    else:raise ValueError(mode)
