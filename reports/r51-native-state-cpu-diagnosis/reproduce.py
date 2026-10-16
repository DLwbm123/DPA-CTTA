"""CPU arithmetic only; reads saved anchor/vectors, never images or model forwards.

Run in the pinned environment with CUDA_VISIBLE_DEVICES='', the frozen source
on PYTHONPATH and EVIDENCE_ROOT pointing at the preserved retry2 directory.
Only aggregate results go to stdout. No production diagnostic code is modified.
"""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CUDA must be disabled before interpreter startup')
import copy
from fractions import Fraction
import json
from pathlib import Path
import time
import torch
from dpa_ctta.r51_same_state.diagnostic import write_delta


def flat(xs):
    return torch.cat([x.detach().reshape(-1) for x in xs])


def difference(a, b):
    d=a.double()-b.double()
    return dict(elements=a.numel(),different=int((a!=b).sum()),
                max_abs=float(d.abs().max()),l2=float(d.norm()))


def main():
    start=time.monotonic();torch.set_num_threads(2)
    root=Path(os.environ['EVIDENCE_ROOT'])
    anchor=torch.load(root/'private/anchor_o0.pt',map_location='cpu',weights_only=True)['host']
    vectors=torch.load(root/'private/qualification_vectors_o0.pt',map_location='cpu',weights_only=True)
    before=[p.clone() for p in anchor['parameters'].values()]
    assert len(before)==82 and sum(p.numel() for p in before)==19136
    assert all(p.dtype==torch.float32 for p in before)
    old=flat(before);delta=vectors['native'];saved_actual=vectors['actuals']['native']
    params=[torch.nn.Parameter(p.clone()) for p in before]
    actual=write_delta(params,before,delta);rewritten=flat(params)
    repeated=write_delta(params,before,delta)
    assert torch.equal(flat(params),rewritten) and torch.equal(actual,repeated)
    # Different post-step float32 parameters can encode exactly the same delta.
    up=torch.nextafter(rewritten,torch.full_like(rewritten,float('inf')))
    down=torch.nextafter(rewritten,torch.full_like(rewritten,float('-inf')))
    ambiguous_up=(up-old)==delta;ambiguous_down=(down-old)==delta
    ambiguous=ambiguous_up|ambiguous_down
    bounds=[]
    for index in ambiguous.nonzero().flatten().tolist():
        d=delta[index];previous=torch.nextafter(d,torch.tensor(float('-inf')))
        following=torch.nextafter(d,torch.tensor(float('inf')))
        base=Fraction.from_float(float(old[index]));encoded=Fraction.from_float(float(d))
        lo=base+(Fraction.from_float(float(previous))+encoded)/2
        hi=base+(encoded+Fraction.from_float(float(following)))/2
        center=Fraction.from_float(float(rewritten[index]))
        bounds.append(float(max(abs(lo-center),abs(hi-center))))
    # Closed midpoint intervals give conservative bounds, including tie endpoints.
    rewrite_only_upper_bound=max(bounds,default=0.)

    # Controlled scalar witness: save exact post-Adam value versus delta rewrite.
    scalar_before=[torch.tensor([1e-4],dtype=torch.float32)]
    scalar_raw=torch.tensor([-1e-12],dtype=torch.float32)
    scalar_param=[torch.nn.Parameter(scalar_raw.clone())]
    scalar_delta=scalar_raw-scalar_before[0]
    write_delta(scalar_param,scalar_before,scalar_delta)
    scalar_bad=scalar_param[0].detach().clone()
    assert not torch.equal(scalar_bad,scalar_raw)
    assert torch.equal(scalar_bad-scalar_before[0],scalar_delta)
    with torch.no_grad():scalar_param[0].copy_(scalar_raw)
    assert torch.equal(scalar_param[0],scalar_raw)

    # One CPU optimizer arithmetic evaluation, using already saved gradients.
    # This is not the unsaved GPU native-reference step and runs no model/data.
    params=[torch.nn.Parameter(p.clone()) for p in before]
    optimizer=torch.optim.Adam(params,lr=1e-4)
    optimizer.load_state_dict(copy.deepcopy(anchor['adam']))
    group=optimizer.param_groups[0]
    assert len(optimizer.param_groups)==1 and group['lr']==1e-4
    assert group['betas']==(.9,.999) and group['weight_decay']==0
    offset=0
    for p in params:
        p.grad=vectors['total_gradient'][offset:offset+p.numel()].reshape_as(p).clone()
        offset+=p.numel()
    assert offset==delta.numel()
    optimizer.step()
    raw=[p.detach().clone() for p in params];raw_flat=flat(raw)
    cpu_delta=flat([p-q for p,q in zip(params,before)])
    cpu_metric=flat([(optimizer.state[p]['exp_avg_sq']/(1-.999**int(optimizer.state[p]['step']))).sqrt()+1e-8 for p in params])
    cpu_actual=write_delta(params,before,cpu_delta)
    cpu_rewritten=flat(params)
    rewrite_changed=(cpu_rewritten!=raw_flat)
    same_saved_delta=cpu_delta==delta
    # The proposed exact-copy remedy must not change any optimizer state.
    saved_optimizer=copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        for p,q in zip(params,raw):p.copy_(q)
    assert torch.equal(flat(params),raw_flat)
    for key,state in saved_optimizer['state'].items():
        for name,value in state.items():
            current=optimizer.state_dict()['state'][key][name]
            assert torch.equal(value,current) if isinstance(value,torch.Tensor) else value==current
    assert not torch.cuda.is_initialized()
    result=dict(status='CPU_DIAGNOSIS_COMPLETE',torch_version=torch.__version__,
        model_forwards=0,image_reads=0,label_reads=0,cuda_initialized=False,gpu_seconds=0,
        saved_evidence=dict(parameter_tensors=len(before),parameters=delta.numel(),
            cpu_rewrite_delta_vs_saved_actual=difference(actual,saved_actual),
            saved_native_delta_vs_saved_actual=difference(delta,saved_actual),
            repeated_write_exact=True,ambiguous_parameter_coordinates=int(ambiguous.sum()),
            ambiguous_up=int(ambiguous_up.sum()),ambiguous_down=int(ambiguous_down.sum()),
            rewrite_only_changed_coordinates_upper_bound=int(ambiguous.sum()),
            rewrite_only_max_abs_error_upper_bound=rewrite_only_upper_bound,
            gradient_decomposition_relative_error=float((vectors['total_gradient'].double()-vectors['base_gradient'].double()-.1*vectors['lso_gradient'].double()).norm()/vectors['total_gradient'].double().norm())),
        scalar_witness=dict(before=float(scalar_before[0]),raw_post=float(scalar_raw),rewritten=float(scalar_bad),
            delta_exactly_preserved=True,rewrite_error=difference(scalar_bad,scalar_raw),exact_copy_fixed=True),
        cpu_adam_arithmetic=dict(cpu_delta_vs_saved_gpu_diagnostic_delta=difference(cpu_delta,delta),
            cpu_metric_vs_saved_gpu_diagnostic_metric=difference(cpu_metric,vectors['metric']),
            cpu_raw_post_vs_delta_rewrite=difference(raw_flat,cpu_rewritten),
            delta_before_vs_after_rewrite=difference(cpu_delta,cpu_actual),
            rewrite_changed_where_cpu_delta_equals_saved_gpu_delta=int((rewrite_changed&same_saved_delta).sum()),
            exact_copy_error=difference(flat(params),raw_flat),optimizer_state_unchanged=True),
        missing=['original native-reference post-step parameters and gradients',
                 'diagnostic raw post-Adam parameters before rewriting',
                 'per-component native-reference versus diagnostic state differences'],
        interpretation='Roundtrip non-invertibility is proven; CPU Adam arithmetic is not GPU native-reference evidence. Saved deltas alone cannot identify the cause or size of the original failed parameter comparison.',
        cpu_seconds=time.monotonic()-start)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
