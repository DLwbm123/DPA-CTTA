# Verbatim function excerpts from the pinned GitHub sources; imports supplied by probes.
# implementation: 0c08cece6a91bdf5e3d06c9862dcd192df7787d8
# validate_metric: src/dpa_ctta/p2_analysis.py
# smoke: src/dpa_ctta/r5_update_acceptance/execution.py

def validate_metric(m):
    na,nb,n,k=(m[v] for v in ['pred_pixels','gt_pixels','total_pixels','intersection'])
    if not all(type(v)==int for v in [na,nb,n,k]) or n<=0 or not 0<=k<=min(na,nb)<=max(na,nb)<=n:raise ValueError('pixel counts')
    if m['dice']!=(2*k/(na+nb) if na+nb else 1.):raise ValueError('Dice reconstruction')
    for key,value in dict(gt_empty=nb==0,gt_full=nb==n,pred_empty=na==0,pred_full=na==n).items():
        if type(m[key])!=bool or m[key]!=value:raise ValueError('metric flags')
    if (m['assd'] is None)!=(na==0 or nb==0) or (m['assd'] is not None and m['assd']<0):raise ValueError('ASSD undefined')


def smoke(state,device,out,identity,recipe,p_accept=None):
    import torch
    from ..r1.host import Host as Old
    from ..source_pilot import seed_all
    from ..b4_run import capture
    from ..host_diagnostic import close
    from ..m2_run import deterministic_smoke_pair
    from ..b3_runtime import process_audit
    from .host import Host
    sys.path.insert(0,str(ROOT/'tests'));from test_vptta_host import pixels
    saved=[];physical={k:0 for k in PHYSICAL};evidence={}
    arms=('OLD','C') if recipe['recipe']=='PROCEDURAL_OLD_C4_DIAG_C4_V1' else ('OLD','C','C_HALF','C_RANDOM','C_VERIFY')
    try:
        with deterministic_smoke_pair():
            for arm in arms:
                seed_all(20260907);h=Old('C',state,device) if arm=='OLD' else Host(arm,state,device,p_accept=p_accept if arm=='C_RANDOM' else None)
                if arm=='OLD' and torch.device(device).type=='cuda':process_audit(out,'smoke')
                for i in recipe['pixel_indices']:
                    z,t=h.step(pixels('fundus',i));core=h if arm=='OLD' else h.core
                    counts=t['counts'];actual=dict(network_forwards=counts['forwards'],loss_backward_calls=counts['backwards'],adam_calls=counts['base_adam'],jacobian_vjp_calls=0) if arm=='OLD' else counts
                    for k,v in actual.items():physical[k]+=v
                    value=dict(logits=z.detach().cpu().clone(),state=capture(core))
                    if arm=='OLD':saved.append(value)
                    elif arm!='C_HALF':close(value,saved[i],exact=torch.device(device).type=='cpu');close(core.rng,saved[i]['state']['rng'],exact=True)
                    if arm!='OLD':h.take_evaluation().clear()
                evidence[arm]=dict(visits=4,counts=core.counts.copy(),parity=arm not in ('OLD','C_HALF'))
                h.finish(state);del h,core,z,value;gc.collect()
        if physical!={k:recipe[k] for k in PHYSICAL}:raise ValueError('smoke physical budget')
        write(out/'smoke.completion.json',dict(binding=identity,status='MECHANICAL_SMOKE_COMPLETE',C_parity_valid=True,recipe=recipe,physical=physical,evidence=evidence,backend=backend_policy()))
    except BaseException as exc:
        write(out/'smoke.failure.json',dict(binding=identity,status='INCOMPLETE',physical=physical,reason=str(exc),scope=failure_scope(exc)));raise
