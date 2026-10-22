"""Fixed factorial diagnosis; prediction generation cannot read labels."""
import os,sys,io,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from helpers import read,write,unpack,appearances,aggregate,audit
ARMS=('N00','N10','N01','N11')
DYNAMIC=('N10_PRE','N10_POST','N01_PRE','N01_POST','N11_PRE','N11_POST')
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
    meter=dict(forward=0,backward=0,updates=0);completed=0;begin=time.perf_counter()
    try:
        for order,sequence in enumerate((list(range(8)),list(reversed(range(8))))):
            arms=[Arm(source,n,meter) for n in ARMS]
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
                    for arm in arms:
                        pre,post,loss=arm.run(x,z.detach(),features[0].detach(),order*1000+pos*4+k)
                        if arm.name=='N00':
                            assert torch.equal(post,z)
                            assert equal_tree(arm.model.state_dict(),original)
                        else:predictions.extend((pre.cpu().numpy()[0],post.cpu().numpy()[0]))
                        if not arm.keep_parameters:
                            assert all(torch.equal(p,q) for p,q in zip(arm.params,source.parameters()))
                        losses.append(loss)
                    with np.load(prior/'private/measurement'/f'{order}_{i}_{k}.npz') as previous:
                        assert np.array_equal(source_logits,previous['logits'][0])
                        assert np.array_equal(predictions[-1],previous['logits'][5]),'N11 must match prior DOMAIN_ONLY'
                    logits=np.stack(predictions)
                    assert np.isfinite(logits).all() and np.isfinite(losses).all()
                    with torch.no_grad():pseudo=((z.sigmoid()>=.5)|(uncertainty(z)>=.75)).cpu().numpy()[0]
                    payload=dict(logits=logits,masks=np.packbits(logits>=0),losses=np.asarray(losses),pseudo=np.packbits(pseudo))
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
                    del x,z,features,pre,post,predictions,logits,payload
            del arms
        assert meter==dict(forward=768,backward=128,updates=128) and completed==64
        assert equal_tree(source.state_dict(),original)
        write(root/'public'/f'{mode.upper()}.json',dict(status='PASS',inputs=completed,counts=meter,source_parity=64,n00_exact_source=64,n11_exact_prior_domain=64,independent_exact_replay=mode=='worker2',label_reads=0,negative_checks=negative))
    finally:
        write(root/'public'/f'{mode.upper()}_COUNTS.json',dict(counts=meter,completed_inputs=completed,seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))

def score(c):
    from runtime import identity
    from metric import metric
    root=Path(c['root']);old=Path(c['sealed_root']);prior=Path(c['prior_root'])
    for stage in ('worker1','worker2'):
        rec=read(root/'processes'/f'{stage}.json');assert rec['exit_code']==0 and not rec['active'] and identity(rec['pid'])!=rec['start_ticks']
    assert read(root/'public/WORKER2.json')['independent_exact_replay']
    panel=read(old/'scorer/panel.json');registered={r['mask_path'] for r in read(old/'scorer/mask_identity.private.json')}
    assert len(panel)==8 and {p['mask_path'] for p in panel}==registered
    keys=('OD','OC','OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR','vCDR_signed','OD_area_ratio','OC_area_ratio','OD_fp_fraction','OC_fp_fraction','OD_fn_fraction','OC_fn_fraction')
    values={a:{key:np.full((8,8),np.nan) for key in keys} for a in READOUTS}
    old_rows=[];pseudo_rows=[];private_rows=[]
    empty={a:0 for a in READOUTS};nest={a:0 for a in READOUTS}
    for i,p in enumerate(panel):
        with Image.open(p['mask_path']) as im:
            assert list(im.size)==p['image_size'];yy=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        gt=np.stack((yy<255,yy==0))
        for order in range(2):
            pos=i if order==0 else 7-i
            for k in range(4):
                b=unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits');src=unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits')
                with np.load(root/'private/measurement'/f'{order}_{i}_{k}.npz') as z:
                    dynamic=np.unpackbits(z['masks']).reshape(6,2,512,512).astype(bool)
                    pseudo=np.unpackbits(z['pseudo']).reshape(2,512,512).astype(bool)
                for a,m in zip(READOUTS,[src,*dynamic,b]):
                    met,ratio=metric(m,gt,b)
                    empty[a]+=sum(v['pred_empty'] for v in met);nest[a]+=int((m[1]&~m[0]).sum())
                    row=dict(order=order,position=pos*4+k,content_index=i,appearance=k,arm=a,metrics={})
                    for j,h in enumerate(('OD','OC')):
                        vals={h:met[j]['dice_percent'],h+'_ASSD':met[j]['assd_pixel'],h+'_HD95':met[j]['hd95_pixel'],h+'_area_ratio':float(m[j].sum()/gt[j].sum()) if gt[j].any() else None,h+'_fp_fraction':float((m[j]&~gt[j]).mean()),h+'_fn_fraction':float((~m[j]&gt[j]).mean())}
                        row['metrics'].update(vals)
                    row['metrics']['vCDR']=ratio['vcdr_abs_error']
                    row['metrics']['vCDR_signed']=ratio['predicted_vcdr']-ratio['true_vcdr'] if ratio['predicted_vcdr'] is not None and ratio['true_vcdr'] is not None else None
                    for key,v in row['metrics'].items():values[a][key][i,order*4+k]=np.nan if v is None else v
                    private_rows.append(row)
                added=pseudo&~src
                pseudo_rows.append(dict(order=order,position=pos*4+k,content_index=i,appearance=k,added_counts=[int(q.sum()) for q in added],correct_added_counts=[int((q&y).sum()) for q,y in zip(added,gt)],oc_added_outside_source_od=int((added[1]&~src[0]).sum())))
                with np.load(prior/'private/measurement'/f'{order}_{i}_{k}.npz') as z:
                    masks=np.unpackbits(z['masks']).reshape(6,2,512,512).astype(bool)
                # No new model calls: characterize the previous main and domain outputs.
                for a,m in (('OWNED_STATE',masks[1]),('DOMAIN_ONLY',masks[5])):
                    record=dict(order=order,position=pos*4+k,content_index=i,appearance=k,arm=a)
                    met,rat=metric(m,gt,b)
                    record.update(od=met[0]['dice_percent'],oc=met[1]['dice_percent'],od_area_ratio=float(m[0].sum()/gt[0].sum()),oc_area_ratio=float(m[1].sum()/gt[1].sum()),fp=[int((q&~y).sum()) for q,y in zip(m,gt)],fn=[int((~q&y).sum()) for q,y in zip(m,gt)],vcdr_signed=rat['predicted_vcdr']-rat['true_vcdr'] if rat['predicted_vcdr'] is not None else None)
                    old_rows.append(record)
    effects={}
    for timing in ('PRE','POST'):
        effects[timing]={}
        for h in ('OD','OC'):
            y00=values['SOURCE'][h];y10=values['N10_'+timing][h];y01=values['N01_'+timing][h];y11=values['N11_'+timing][h]
            effects[timing][h]={name:aggregate(v) for name,v in {'parameter':y10-y00,'bn':y01-y00,'interaction':y11-y10-y01+y00,'bn_given_parameters':y11-y10,'joint':y11-y00}.items()}
    # First appearance of each content only; keep the content as the bootstrap unit.
    first_pre={a:{h:aggregate(np.repeat(values[a][h][:,[0,4]],4,axis=1)) for h in ('OD','OC')} for a in ('SOURCE','N10_PRE','N01_PRE','N11_PRE')}
    for a in first_pre:
        for z in first_pre[a].values():z['defined_inputs']=16
    def mean_field(rows,key):return float(np.mean([v[key] for v in rows if v[key] is not None]))
    portrait={a:{key:mean_field([v for v in old_rows if v['arm']==a],key) for key in ('od','oc','od_area_ratio','oc_area_ratio','vcdr_signed')} for a in ('OWNED_STATE','DOMAIN_ONLY')}
    pseudostat={}
    for j,h in enumerate(('OD','OC')):
        n=sum(v['added_counts'][j] for v in pseudo_rows);good=sum(v['correct_added_counts'][j] for v in pseudo_rows)
        pseudostat[h]=dict(added_pixels=n,correct_added_pixels=good,precision=good/n if n else None,fraction_of_all_pixels=n/(64*512*512))
    pseudostat['oc_added_outside_source_od']=sum(v['oc_added_outside_source_od'] for v in pseudo_rows)
    write(root/'private/PER_INPUT.json',dict(current=private_rows,previous=old_rows,pseudo=pseudo_rows))
    write(root/'public/RESULTS.json',dict(status='COMPLETE_DIAGNOSTIC',method_promoted=False,metrics={a:{key:aggregate(v) for key,v in data.items()} for a,data in values.items()},effects=effects,empty_heads=empty,nesting_pixels=nest,contrasts={a:{base:{h:aggregate(values[a][h]-values[base][h]) for h in ('OD','OC')} for base in ('SOURCE','B')} for a in DYNAMIC},first_content_pre=first_pre,previous_portrait_unweighted=portrait,pseudo_pixel_pooled=pseudostat,scope='8 repeatedly used development contents; diagnostic factorial effects, not method qualification or independent patient validation.'))

def run(c):
    mode=os.environ['AUDIT_MODE']
    if mode=='check':check(c)
    elif mode in ('worker1','worker2'):worker(c)
    elif mode=='score':score(c)
    else:raise ValueError(mode)
