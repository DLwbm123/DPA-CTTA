"""Label-isolated, content-bound sparse-state experiment."""
import os
import sys
import io
import time
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from helpers import read, write, unpack, appearances, aggregate, audit

ARMS = ('SOURCE', 'OWNED_STATE', 'WEIGHT_ONLY', 'ALL_RESET', 'RANDOM_OWNED', 'DOMAIN_ONLY', 'B')


def check(c):
    from engine import check as operator_check
    operator_check()
    root = Path(c['root'])
    panel = read(Path(c['sealed_root'])/'private/panel.json')
    registered = {r['image_relative'] for r in c['rgb_identity']}
    assert len(panel) == 8
    assert all(sum(str(p['image_path']).endswith(r) for r in registered) == 1 for p in panel)
    write(root/'private/panel.json', panel)
    write(root/'public/CHECK.json', dict(status='PASS', registered_contents=8, model_calls=0,
                                        gt_reads=0, rectangle_union=True, optimizer_restore=True,
                                        inverse_transforms=True))


def worker(c):
    import torch
    from engine import Arm, load_contrast, clone_cpu, equal_tree, seeded
    root = Path(c['root']); old = Path(c['sealed_root']); mode = os.environ['AUDIT_MODE']
    panel = read(root/'private/panel.json')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    negative = audit(c, panel)
    sys.path.insert(0, str(old/'code/src'))
    from dpa_ctta.integrations.ctta_suite import build_reference_model
    seeded(0)
    source, _ = build_reference_model('fundus')
    source.load_state_dict(torch.load(c['checkpoint_path'], map_location='cpu', weights_only=True), strict=True)
    torch.cuda.set_per_process_memory_fraction(min(.8, 10*1024**3/torch.cuda.get_device_properties(0).total_memory))
    source.cuda().eval().requires_grad_(False)
    original = clone_cpu(source.state_dict())
    contrast_class = load_contrast(root/'private/vendor/contrast.py')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name', '--format=csv,noheader'], text=True)
    mine = [v for v in gpu.splitlines() if v.split(',')[0].strip() == str(os.getpid())]
    assert len(mine) == 1 and c['gpu_uuid'] in mine[0] and 'wangbomin' not in mine[0]
    out = root/'private/measurement'; out.mkdir(exist_ok=True)
    meter = dict(forward=0, backward=0, updates=0)
    changed = total = completed = parity = state_equal = 0
    begin = time.perf_counter()
    try:
        for order, sequence in enumerate((list(range(8)), list(reversed(range(8))))):
            arms = [Arm(source, name, contrast_class, meter) for name in ARMS[1:-1]]
            for pos, i in enumerate(sequence):
                item = panel[i]
                with Image.open(io.BytesIO(Path(item['image_path']).read_bytes())) as im:
                    assert list(im.size) == item['image_size']
                    rgb = np.asarray(im.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC), np.float32)
                    image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()/255)[None]
                for k, view in enumerate(appearances(image)):
                    input_begin = time.perf_counter()
                    assert view.max() > view.min()
                    x = ((view-view.min())/(view.max()-view.min())).cuda()
                    features = []
                    handle = source.up5.register_forward_hook(lambda mod, args, output: features.append(output))
                    try:
                        with torch.no_grad():
                            meter['forward'] += 1
                            z = source(x)[0]
                    finally:
                        handle.remove()
                    assert len(features) == 1
                    assert np.array_equal((z.cpu().numpy()[0] >= 0), unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits'))
                    parity += 1
                    predictions = [z.detach().cpu().numpy()[0]]
                    diagnostics = []
                    matched = None
                    for arm in arms:
                        prediction, diag, counts = arm.run(x, z.detach(), features[0].detach(),
                                                          order*1000+pos*4+k, matched)
                        if arm.name == 'OWNED_STATE':
                            matched = counts
                        if arm.name == 'RANDOM_OWNED':
                            assert counts == matched
                        predictions.append(prediction.detach().cpu().numpy()[0])
                        diagnostics.extend((diag['domain_loss'], diag['refine_loss']))
                        write(root/'RUN_STATE.json', dict(stage=mode, completed_inputs=completed,
                                                         arm_completed=arm.name, counts=meter))
                    # Restored refinement cannot change the future domain trajectory.
                    assert equal_tree(arms[0].model.state_dict(), arms[-1].model.state_dict())
                    assert equal_tree(arms[0].opt.state_dict(), arms[-1].opt.state_dict())
                    assert equal_tree(arms[2].model.state_dict(), arms[-1].model.state_dict())
                    state_equal += 1
                    logits = np.stack(predictions)
                    assert np.isfinite(logits).all() and np.isfinite(diagnostics).all()
                    if pos == k == 0:
                        assert np.array_equal(logits[1], logits[2]) and np.array_equal(logits[1], logits[3])
                    changed += int((np.abs(logits[1]-logits[4]) > 1e-3).sum())
                    total += logits[1].size
                    payload = dict(logits=logits, masks=np.packbits(logits >= 0), diagnostic=np.asarray(diagnostics))
                    path = out/f'{order}_{i}_{k}.npz'
                    if mode == 'worker1':
                        np.savez_compressed(path, **payload)
                    else:
                        with np.load(path) as previous:
                            assert set(previous.files) == set(payload)
                            for name, value in payload.items():
                                assert np.array_equal(previous[name], value), name
                    completed += 1
                    write(root/'RUN_STATE.json', dict(stage=mode, completed_inputs=completed, counts=meter))
                    if mode == 'worker1' and completed == 1:
                        torch.cuda.synchronize()
                        seconds = time.perf_counter()-input_begin
                        elapsed = time.clock_gettime(time.CLOCK_BOOTTIME)-c['origin_boottime']
                        estimate = seconds*127*1.5 + elapsed + 240
                        write(root/'public/COST_ADMISSION.json', dict(first_input_seconds=seconds,
                              projected_total_seconds=estimate, passed=estimate <= 7200))
                        assert estimate <= 7200, 'complete controls/replay do not fit the frozen budget'
                    del x, z, features, prediction, predictions, logits, payload
            assert equal_tree(source.state_dict(), original) and all(p.grad is None for p in source.parameters())
            del arms
        assert meter == dict(forward=2496, backward=576, updates=576)
        assert completed == parity == state_equal == 64
        write(root/'public'/f'{mode.upper()}.json', dict(status='PASS', inputs=64, counts=meter,
              source_bit_parity=parity, source_immutable=True, domain_state_equality=state_equal,
              independent_exact_replay=mode == 'worker2', access_negative_checks=negative,
              label_reads=0, neutral_nvidia=True, semantic_changed_fraction=changed/total,
              pixel_queue_commits='pre-refinement observations, except ALL_RESET',
              randomness='input/phase keyed, not arm keyed; physical counters never restored'))
    finally:
        write(root/'public'/f'{mode.upper()}_COUNTS.json', dict(counts=meter, completed_inputs=completed,
              seconds=time.perf_counter()-begin, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
              forward_backward_counters='started calls; successful run totals asserted'))
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
                with np.load(root/'private/measurement'/f'{order}_{i}_{k}.npz') as z:pred=np.unpackbits(z['masks']).reshape(6,2,512,512).astype(bool)
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
                    worst=min(worst,values['OWNED_STATE'][h][i,order*4+k]-values['B'][h][i,order*4+k]);new_empty|=not pred[1,j].any() and b[j].any()
                new_nesting|=bool((pred[1,1]&~pred[1,0]).sum()>(b[1]&~b[0]).sum())
    contrast={a:aggregate(values['OWNED_STATE']['OC']-values[a]['OC']) for a in ARMS if a!='OWNED_STATE'}
    good=lambda z:z['ci95'] is not None and z['ci95'][0]>0
    od=aggregate(values['OWNED_STATE']['OD']-values['B']['OD']);vcdr=aggregate(values['OWNED_STATE']['vCDR']-values['B']['vCDR'])
    ratio={key:aggregate(values['OWNED_STATE'][key]/values['B'][key]) for key in ('OC_ASSD','OC_HD95')}
    gates=dict(primary=contrast['B']['mean']>=.3 and good(contrast['B']),source=good(contrast['SOURCE']),controls=good(contrast['WEIGHT_ONLY']) and good(contrast['ALL_RESET']) and good(contrast['RANDOM_OWNED']) and good(contrast['DOMAIN_ONLY']),od_noninferior=od['ci95'] is not None and od['ci95'][0]>=-.1,worst_head_loss=worst>=-1,new_empty=not new_empty,new_nesting=not new_nesting,full_surface=all(np.isfinite(values['OWNED_STATE'][key]).all() for key in ('OD_ASSD','OC_ASSD','OD_HD95','OC_HD95','vCDR')),surface_noninferior=all(z['defined_inputs']==64 and z['ci95'] is not None and z['ci95'][1]<=1.05 for z in ratio.values()),vcdr=vcdr['ci95'] is not None and vcdr['ci95'][1]<=.005,semantic_null=read(root/'public/WORKER2.json')['semantic_changed_fraction']>=.1)
    order_contrasts=[aggregate(values['OWNED_STATE']['OC'][:,j*4:(j+1)*4]-values['B']['OC'][:,j*4:(j+1)*4]) for j in range(2)]
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
