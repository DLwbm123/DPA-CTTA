"""Registered real-image frozen-model qualification, generation before scoring."""
import os,sys,io,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from helpers import read,write,unpack,appearances,aggregate,audit

ARMS=('SOURCE','CONTRAST','MEAN','REVERSE','SHUFFLED','B')

def check(c):
    from streams import check as operator_check
    operator_check();root=Path(c['root']);panel=read(Path(c['sealed_root'])/'private/panel.json')
    assert len(panel)==8
    registered={r['image_relative'] for r in c['rgb_identity']}
    assert all(sum(str(p['image_path']).endswith(r) for r in registered)==1 for p in panel)
    write(root/'private/panel.json',panel)
    write(root/'public/CHECK.json',dict(status='PASS',registered_contents=8,inputs=64,model_calls=0,gt_reads=0,operator_assertions=True))

def worker(c):
    import torch
    from streams import variants,Memory
    begin=time.perf_counter();root=Path(c['root']);old=Path(c['sealed_root']);mode=os.environ['AUDIT_MODE'];panel=read(root/'private/panel.json')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20261018);np.random.seed(20261018)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    negative=audit(c,panel);sys.path.insert(0,str(old/'code/src'))
    from dpa_ctta.integrations.ctta_suite import build_reference_model
    model,_=build_reference_model('fundus');model.load_state_dict(torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True));model.cuda().eval().requires_grad_(False)
    original={k:v.cpu().clone() for k,v in model.state_dict().items()}
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True)
    mine=[v for v in gpu.splitlines() if v.split(',')[0].strip()==str(os.getpid())]
    assert len(mine)==1 and c['gpu_uuid'] in mine[0] and 'wangbomin' not in mine[0]
    calls=0;parity=0;changed=0;count=0;out=root/'private/measurement';out.mkdir(exist_ok=True)
    for order,sequence in enumerate((list(range(8)),list(reversed(range(8))))):
        memory=Memory()
        for pos,i in enumerate(sequence):
            item=panel[i];proposals=[]
            assert set(memory.ids)==set(sequence[:pos]) and i not in memory.ids
            with Image.open(io.BytesIO(Path(item['image_path']).read_bytes())) as im:
                assert list(im.size)==item['image_size'];image=torch.from_numpy(np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),np.float32).transpose(2,0,1).copy()/255)[None]
            for k,view in enumerate(appearances(image)):
                span=view.max()-view.min();assert span>0;x=((view-view.min())/span)[0].numpy()
                with torch.no_grad():logits,diag,proposal=variants(model,torch.from_numpy(x.copy())[None].cuda(),memory)
                proposals.append(proposal)
                calls+=5
                masks=logits>=0;s=unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits');assert np.array_equal(masks[0],s);parity+=1
                changed+=int((np.abs(logits[1]-logits[4])>1e-3).sum());count+=logits[1].size
                payload=dict(logits=logits,masks=np.packbits(masks),diagnostic=diag);path=out/f'{order}_{i}_{k}.npz'
                if mode=='worker1':np.savez_compressed(path,**payload)
                else:
                    with np.load(path) as prev:
                        assert set(prev.files)==set(payload)
                        for name,v in payload.items():assert np.array_equal(prev[name],v),name
                assert all(torch.equal(v.cpu(),original[n]) for n,v in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
                write(root/'RUN_STATE.json',dict(stage=mode,completed_inputs=order*32+pos*4+k+1,model_calls=calls))
                if mode=='worker1' and order==0 and i==k==0:
                    torch.cuda.synchronize();first=time.perf_counter()-begin;elapsed=time.clock_gettime(time.CLOCK_BOOTTIME)-c['origin_boottime'];estimate=first*127*1.5+elapsed+180
                    write(root/'public/COST_ADMISSION.json',dict(first_input_seconds=first,projected_total_seconds=estimate,passed=estimate<=3600));assert estimate<=3600
            memory.add(i,proposals)
    assert calls==320 and parity==64
    write(root/'public'/f'{mode.upper()}.json',dict(status='PASS',inputs=64,model_calls=calls,decoder_only_calls=0,head_only_calls=0,content_causal=True,orders=2,source_bit_parity=64,source_immutable=True,no_parameter_grad=True,independent_exact_replay=mode=='worker2',access_negative_checks=negative,label_reads=0,neutral_nvidia=True,semantic_changed_fraction=changed/count))

def score(c):
    from runtime import identity
    from metric import metric
    root=Path(c['root']);old=Path(c['sealed_root'])
    for stage in ('worker1','worker2'):
        rec=read(root/'processes'/f'{stage}.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    assert read(root/'public/WORKER2.json')['independent_exact_replay']
    panel=read(old/'scorer/panel.json');registered={r['mask_path'] for r in read(old/'scorer/mask_identity.private.json')};assert len(panel)==8 and {p['mask_path'] for p in panel}==registered
    values={a:{key:np.full((8,8),np.nan) for key in ('OD','OC','OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR')} for a in ARMS}
    empty={a:0 for a in ARMS};nest={a:0 for a in ARMS};worst=0.;new_empty=False;new_nesting=False
    for i,p in enumerate(panel):
        with Image.open(p['mask_path']) as im:
            assert list(im.size)==p['image_size'];yy=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((yy<255,yy==0))
        for order in range(2):
            for k in range(4):
                b=unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits')
                with np.load(root/'private/measurement'/f'{order}_{i}_{k}.npz') as z:pred=np.unpackbits(z['masks']).reshape(5,2,512,512).astype(bool)
                allm=list(pred)+[b]
                for a,m in zip(ARMS,allm):
                    met,ratio=metric(m,gt,b)
                    for j,h in enumerate(('OD','OC')):
                        values[a][h][i,order*4+k]=met[j]['dice_percent'];empty[a]+=met[j]['pred_empty']
                        for suf,key in (('_ASSD','assd_pixel'),('_HD95','hd95_pixel')):
                            v=met[j][key];values[a][h+suf][i,order*4+k]=np.nan if v is None else v
                    values[a]['vCDR'][i,order*4+k]=ratio['vcdr_abs_error'] if ratio['vcdr_abs_error'] is not None else np.nan
                    nest[a]+=int((m[1]&~m[0]).sum())
                for j,h in enumerate(('OD','OC')):
                    worst=min(worst,values['CONTRAST'][h][i,order*4+k]-values['B'][h][i,order*4+k]);new_empty|=not pred[1,j].any() and b[j].any()
                new_nesting|=bool((pred[1,1]&~pred[1,0]).sum()>(b[1]&~b[0]).sum())
    contrast={a:aggregate(values['CONTRAST']['OC']-values[a]['OC']) for a in ARMS if a!='CONTRAST'}
    good=lambda z:z['ci95'] is not None and z['ci95'][0]>0
    od=aggregate(values['CONTRAST']['OD']-values['B']['OD']);vcdr=aggregate(values['CONTRAST']['vCDR']-values['B']['vCDR'])
    ratio={key:aggregate(values['CONTRAST'][key]/values['B'][key]) for key in ('OC_ASSD','OC_HD95')}
    gates=dict(primary=contrast['B']['mean']>=.3 and good(contrast['B']),source=good(contrast['SOURCE']),controls=good(contrast['MEAN']) and good(contrast['REVERSE']) and good(contrast['SHUFFLED']),od_noninferior=od['ci95'] is not None and od['ci95'][0]>=-.1,worst_head_loss=worst>=-1,new_empty=not new_empty,new_nesting=not new_nesting,full_surface=all(np.isfinite(values['CONTRAST'][key]).all() for key in ('OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR')),surface_noninferior=all(z['defined_inputs']==64 and z['ci95'] is not None and z['ci95'][1]<=1.05 for z in ratio.values()),vcdr=vcdr['ci95'] is not None and vcdr['ci95'][1]<=.005,semantic_null=read(root/'public/WORKER2.json')['semantic_changed_fraction']>=.1)
    order_contrasts=[aggregate(values['CONTRAST']['OC'][:,j*4:(j+1)*4]-values['B']['OC'][:,j*4:(j+1)*4]) for j in range(2)]
    gates['both_orders']=all(z['mean']>=.3 and good(z) for z in order_contrasts)
    gates={k:bool(v) for k,v in gates.items()}
    result=dict(status='COMPLETE',qualified=all(gates.values()),gates=gates,metrics={a:{key:aggregate(v) for key,v in data.items()} for a,data in values.items()},oc_contrasts_pp=contrast,order_contrasts_b_pp=order_contrasts,od_contrast_b_pp=od,oc_surface_ratios=ratio,vcdr_delta=vcdr,worst_head_delta_pp=worst,empty_heads=empty,nesting_pixels=nest,registered_contents=8,inputs=64,task_scope='Repeatedly used development contents; not independent patient validation.')
    assert abs(result['metrics']['B']['OC']['mean']-87.80529246678334)<1e-8
    write(root/'public/RESULTS.json',result)

def run(c):
    mode=os.environ['AUDIT_MODE']
    if mode=='check':check(c)
    elif mode in ('worker1','worker2'):worker(c)
    elif mode=='score':score(c)
    else:raise ValueError(mode)
