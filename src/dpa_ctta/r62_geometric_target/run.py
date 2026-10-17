"""Fixed late-state geometric pseudo-target diagnostic; native host remains the implementation."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from ..r51_same_state.run import read, save, load_anchors, run_child

BRANCHES = ('no_update', 'native', 'original_target', 'native_target', 'original_view')


def target_delta(logits, native, original, weight):
    import torch.nn.functional as F
    return (weight * (F.binary_cross_entropy_with_logits(logits, original.detach(), reduction='none') -
                      F.binary_cross_entropy_with_logits(logits, native.detach(), reduction='none'))).mean()


@contextmanager
def original_target(host, passthrough=False):
    from ..r20_model_only_search.method import weights_for
    import torch
    original = host._criterion
    def criterion(logits, pseudo):
        base = original(logits, pseudo)
        q = pseudo.detach() if passthrough else host.weak[0].sigmoid().to(pseudo)
        p = host.modules['W']
        w = weights_for(torch.stack(host.weak).sigmoid(), p['variance_temperature'], p['boundary_boost']).to(logits)
        return base + target_delta(logits, pseudo, q, w)
    host._criterion = criterion
    try:
        yield
    finally:
        host._criterion = original


def self_check():
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    z=torch.tensor([-.7,.2,1.3],requires_grad=True)
    q=torch.tensor([.1,.7,.6]);o=torch.tensor([.2,.5,.9]);w=torch.tensor([.5,1.,1.5])
    base=(w*F.binary_cross_entropy_with_logits(z,q,reduction='none')).mean()
    actual=base+target_delta(z,q,o,w)
    expected=(w*F.binary_cross_entropy_with_logits(z,o,reduction='none')).mean()
    assert torch.allclose(actual,expected,atol=1e-7,rtol=0)
    assert torch.allclose(torch.autograd.grad(actual,z,retain_graph=True)[0],torch.autograd.grad(expected,z)[0],atol=1e-7,rtol=0)
    assert torch.equal(target_delta(z,q,q,w),torch.zeros(()))
    fn=lambda z,q:z
    host=SimpleNamespace(_criterion=fn)
    try:
        with original_target(host):raise RuntimeError('fixture')
    except RuntimeError:pass
    assert host._criterion is fn


def pair(host, anchor, pixels, repeat=False):
    import torch
    from ..r19_model_only.method import equal
    from ..r51_same_state.diagnostic import mask
    from ..joint_sensitivity.method import flatten
    host.restore(anchor)
    outputs={'no_update':mask(host._readonly(host.native.model,pixels))}
    states={}; weak={}; info={}
    for name in ('native','original_target'):
        host.restore(anchor)
        before=flatten(host.native.params)
        if name=='original_target':
            with original_target(host):
                z,detail=host.step(pixels)
        else:
            z,detail=host.step(pixels)
        outputs[name]=mask(z);states[name]=host.snapshot()
        weak[name]=torch.stack(host.weak).clone()
        info[name]=dict(update_norm=float((flatten(host.native.params)-before).norm()),
                        gradient_norm=float(flatten([p.grad for p in host.native.params]).norm()),
                        loss=float(host.native.last['cal_consis_loss']),counts=detail['native'][0]['counts'])
        if repeat:
            host.restore(anchor)
            if name=='original_target':
                with original_target(host): again,_=host.step(pixels)
            else: again,_=host.step(pixels)
            if not torch.equal(mask(again),outputs[name]) or not equal(host.snapshot(),states[name]):
                raise ValueError('repeated branch state mismatch: '+name)
    if not torch.equal(weak['native'],weak['original_target']):raise ValueError('weak targets changed')
    if not equal(states['native']['native_rng'],states['original_target']['native_rng']):raise ValueError('augmentation RNG mismatch')
    if info['native']['counts']!=info['original_target']['counts']:raise ValueError('branch operation mismatch')
    outputs['native_target']=(weak['native'].sigmoid().mean(0)[0]>=.5).cpu()
    outputs['original_view']=(weak['native'][0].sigmoid()[0]>=.5).cpu()
    if repeat:
        host.restore(anchor)
        with original_target(host, passthrough=True):
            same,_=host.step(pixels)
        if not torch.equal(mask(same),outputs['native']) or not equal(host.snapshot(),states['native']):
            raise ValueError('zero substitution native state mismatch')
    host.restore(anchor)
    if repeat and not equal(host.snapshot(),anchor):raise ValueError('final anchor restoration mismatch')
    host.check_frozen(True)
    return outputs,info


def worker(c):
    import torch
    from ..r37_cw_orientation.method import Host,candidates
    from ..r19_model_only.runtime import weights,image
    from ..r19_model_only.method import equal
    from ..r7_target_screen.runner import TargetReader,file_identity
    from ..r9_current_first.assets import gpu_policy
    from ..r36_incremental_modules.run import filesystem_guard
    root=Path(c['output_root']);started=time.time();host=None
    status=dict(status='QUALIFYING',formal_pairs=0,started=started)
    save(root/'RUN_STATE.json',status)
    def guard():
        if time.time()-started>c['gpu_cap_seconds']-30:raise TimeoutError('fixed GPU budget')
    guard.extra={'image_accesses':0}
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpu'])
        permit,opened,rejected=filesystem_guard(c)
        anchors=load_anchors(c);state=weights(c)
        panel=read(root/'private/images.json');qualification=read(root/'private/qualification.json')
        if len(panel)!=48 or len(anchors)!=2 or len(qualification)!=2:raise ValueError('fixed coverage')
        reader=TargetReader(c['target_root'],256*1024**2,'image');checks=[]
        for order,anchor in enumerate(anchors):
            host=Host(state,candidates()[3],17011,anchor['context']['sha256'],device='cuda:0')
            host.restore(anchor)
            if not equal(host.snapshot(),anchor):raise ValueError('initial restoration mismatch')
            pixels=image(reader,qualification[order],permit,guard)
            tick=time.time();outputs,info=pair(host,anchor,pixels,repeat=True);torch.cuda.synchronize()
            historical=np.unpackbits(np.frombuffer((root/'private'/f'qualification_o{order}.bits').read_bytes(),dtype=np.uint8)).reshape(2,512,512)
            if not np.array_equal(outputs['native'].numpy(),historical):raise ValueError('historical native mask mismatch')
            checks.append(dict(order=order,status='PASS',seconds=time.time()-tick,historical_mask_exact=True,
                               repeated_states_exact=True,weak_views_exact=True,native_rng_exact=True,zero_substitution_state_exact=True))
            host.close();host=None;guard()
        estimate=96*max(q['seconds'] for q in checks)*1.5
        if time.time()-started+estimate>c['gpu_cap_seconds']-30:raise TimeoutError('formal matrix budget not admitted')
        save(root/'QUALIFICATION.json',dict(status='PASS',checks=checks,formal_estimate_seconds=estimate))
        status['status']='RUNNING_FIXED_PAIRS';save(root/'RUN_STATE.json',status)
        with (root/'private/pairs.jsonl').open('x') as trace:
            for order,anchor in enumerate(anchors):
                host=Host(state,candidates()[3],17011,anchor['context']['sha256'],device='cuda:0')
                for i,row in enumerate(panel):
                    guard();pixels=image(reader,row,permit,guard);tick=time.time()
                    outputs,info=pair(host,anchor,pixels);torch.cuda.synchronize();pid=order*48+i
                    with (root/'predictions'/f'{pid:03d}.bits').open('xb') as f:
                        for branch in BRANCHES:f.write(np.packbits(outputs[branch].numpy()).tobytes())
                    trace.write(json.dumps(dict(pair=pid,seconds=time.time()-tick,diagnostics=info),allow_nan=False)+'\n');trace.flush()
                    status['formal_pairs']+=1
                    if status['formal_pairs']%12==0:save(root/'RUN_STATE.json',status)
                host.close();host=None
        for path,(_,identity) in reader.used.items():
            if file_identity(path)!=identity:raise ValueError('input metadata changed')
        save(root/'private/ACCESS_AUDIT.json',dict(label_reads=0,image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),rejected=rejected))
        status['status']='GPU_COMPLETE'
        return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',reason=str(e));traceback.print_exc();return 1
    finally:
        if host is not None:host.close()
        status.update(ended=time.time(),worker_seconds=time.time()-started);save(root/'RUN_STATE.json',status)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import interval,csvout
    if os.environ['CUDA_VISIBLE_DEVICES']!='':raise ValueError('CPU scoring required')
    root=Path(c['output_root']);status=read(root/'RUN_STATE.json');retired=read(root/'processes/worker.json')
    if status['status']!='GPU_COMPLETE' or status['formal_pairs']!=96 or retired['active'] or retired['exit_code']!=0:
        raise ValueError('successful retirement required')
    panel=read(root/'scorer/panel.json');reader=TargetReader(c['target_root'],256*1024**2,'mask')
    traces=[json.loads(x) for x in (root/'private/pairs.jsonl').read_text().splitlines()]
    if len(panel)!=48 or [t['pair'] for t in traces]!=list(range(96)):raise ValueError('scoring coverage')
    values=np.empty((48,2,5,2))
    for i,row in enumerate(panel):
        label=reader.read(row)
        for order in (0,1):
            raw=(root/'predictions'/f'{order*48+i:03d}.bits').read_bytes()
            if len(raw)!=5*65536:raise ValueError('prediction coverage')
            masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(5,2,512,512).astype(bool)
            for b,m in enumerate(masks):values[i,order,b]=[100*v['dice'] for v in hard_metrics(m,label)]
    reader.after_check();np.save(root/'scorer/scores.private.npy',values)
    domains=np.array([r['domain'] for r in panel]);assert all(sum(domains==d)==12 for d in set(domains))
    cells=[];summaries=[]
    for name,a,b in (('original_target_minus_native',2,1),('native_minus_no_update',1,0),('original_target_minus_no_update',2,0),('original_view_minus_native_target',4,3)):
        delta=values[:,:,a]-values[:,:,b]
        summaries.append(dict(contrast=name,mean_pp=float(delta.mean()),ci95_pp=interval(delta,domains),anchor_pp=delta.mean((0,2)).tolist()))
        for order in (0,1):
            for domain in sorted(set(domains)):
                for ch,channel in enumerate(('OD','OC')):
                    cells.append(dict(contrast=name,order=order,domain=domain,channel=channel,delta_pp=float(delta[domains==domain,order,ch].mean())))
    main=summaries[0];pc=[r for r in cells if r['contrast']==main['contrast']]
    rv=float(np.mean([r['delta_pp'] for r in pc if r['domain']=='REFUGE_Valid' and r['channel']=='OC']))
    gate=main['mean_pp']>=.1 and main['ci95_pp'][0]>0 and min(main['anchor_pp'])>0 and min(r['delta_pp'] for r in pc)>=-2 and rv>=1
    target=summaries[3]
    target_gate=target['mean_pp']>0 and min(target['anchor_pp'])>0
    result=dict(status='COMPLETE',update_gate=bool(gate),target_quality_gate=bool(target_gate),allocation_gate=bool(gate and target_gate),contrasts=summaries,refuge_valid_OC_pp=rv,independent_validation=False)
    out=root/'public';save(out/'RESULTS.json',result);csvout(out/'ALL_PAIRED_CELLS.csv',cells)
    csvout(out/'ALL_NEGATIVE_CELLS.csv',[r for r in cells if r['delta_pp']<0])
    csvout(out/'UPDATE_NORMS.csv',[dict(branch=b,mean_update_norm=float(np.mean([t['diagnostics'][b]['update_norm'] for t in traces])),mean_gradient_norm=float(np.mean([t['diagnostics'][b]['gradient_norm'] for t in traces]))) for b in ('native','original_target')])
    save(out/'QUALIFICATION.json',read(root/'QUALIFICATION.json'))
    save(out/'COMPLETION_AUDIT.json',dict(pairs=96,contents=48,anchors=2,predictions=480,online_label_reads=0,retired_exit=0))
    (out/'REPORT.md').write_text('# R62 paired geometric-target diagnostic\n\n'+json.dumps(result,indent=2)+'\n\nTwo late CW_LSO states, 48 development-exposed contents, fixed one-step branches. Gradient magnitudes are not matched. No full-stream, novel-method or independent-generalization claim. All adverse cells are retained.\n')
    status.update(status='COMPLETE',scored_predictions=480,ended=time.time());save(root/'RUN_STATE.json',status)


def main():
    mode=os.environ['DIAG_MODE']
    if mode=='self_check':self_check();print('SELF_CHECK_PASS');return 0
    c=read(os.environ['DIAG_CONFIG']);root=Path(c['output_root'])
    if mode=='worker':return worker(c)
    if mode=='score':score(c);return 0
    if mode!='controller':raise ValueError('unknown mode')
    with (root/'started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    if read(root/'CPU_CHECK.json')['status']!='PASS':raise ValueError('CPU qualification missing')
    attempts=[]
    try:
        attempts.append(run_child(c,'worker',c['gpu_cap_seconds'],True))
        if attempts[-1]['exit_code']==0:attempts.append(run_child(c,'score',c['cpu_cap_seconds'],False))
    finally:
        ledger=dict(attempts=attempts,gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),cpu_worker_seconds=sum(a['cpu_worker_seconds'] for a in attempts),all_workers_retired=True)
        save(root/'RESOURCE_LEDGER.json',ledger);save(root/'public/RESOURCE_LEDGER.json',ledger)
    return 0 if len(attempts)==2 and all(a['exit_code']==0 for a in attempts) else 1


if __name__=='__main__':raise SystemExit(main())
