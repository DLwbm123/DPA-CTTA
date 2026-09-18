"""Complete source tensor pipeline; filesystem/real execution entry remains disabled."""
import copy
import hashlib
import torch
from .source import (oracle_all,anchors,simulate,a_basis,shared_basis,SourceTrainer)
from .numerics import seg_loss,COUNTS
from .context import environment,make_context,validate_context,validate_provenance,provenance,HASH_COST

def new_method(group,basis,static=False):
    if group=='A':from ..r7_a_psf import build
    elif group=='B':from ..r7_b_rca import build
    elif group=='C':from ..r7_c_rbe import build
    else:raise ValueError('group A/B/C')
    return build(basis,static)

@torch.no_grad()
def scaler_observations(segmenter,data):
    # Fit-only frozen backbone observations; this extra preparation cost is counted.
    result=[]
    for aid,style in enumerate(anchors('fit')):
        for group in data.folds['fit']:
            image=simulate(data.get(group,'fit').image,style,f'R7_SCALER|{aid}|{group}')
            _,raw,_=segmenter(image,observe=True);result.append(raw)
    return torch.stack(result)

@torch.no_grad()
def evaluate_oracles(segmenter,data,oracles):
    oracles.validate(data);rows=[]
    for aid,style in enumerate(anchors(oracles.fold)):
        for group in data.folds[oracles.fold]:
            if group in oracles.support_pairs[aid]:continue
            record=data.get(group,oracles.fold);x=simulate(record.image,style,f'R7_ORACLE_QUERY|{oracles.fold}|{aid}|{group}')
            baseline=seg_loss(segmenter(x),record.label);changed=seg_loss(segmenter(x,oracles.values[:,aid].float()),record.label)
            rows.append(dict(anchor=aid,group=group,baseline=float(baseline),proxy=float(changed)))
    return rows

def prepare_tensors(segmenter,data,*,source_provenance=None):
    """For separately authorized future source entry or explicitly procedural callers.

    No disk reads, subprocess, device transfer, target loop, or publication occurs.
    FULL/STATIC get independent same-seed parameters, same schedule and full budget.
    """
    identity_before=HASH_COST.copy()
    source_provenance=provenance() if source_provenance is None else copy.deepcopy(source_provenance)
    validate_provenance(source_provenance)
    training_environment=environment(segmenter)
    before=COUNTS.copy()
    oracles={f:oracle_all(segmenter,data,f) for f in ('fit','cal','val')}
    u=shared_basis(oracles['fit'],data);b,baudit=a_basis(segmenter,data)
    raw=scaler_observations(segmenter,data);outputs={}
    for group in 'ABC':
        for static in (False,True):
            m=new_method(group,b if group=='A' else u,static);m.observer.fit_scaler(raw,'fit')
            trainer=SourceTrainer(segmenter,m,data,oracles['fit']);trainer.fit();trainer.start_calibration(oracles['cal']);trainer.calibrate()
            projections={f:[dict(anchor=i,residual_l2=float((m.basis@m.project(o.values[:,i])-o.values[:,i]).norm()),condition=float(torch.linalg.cond(m.basis))) for i in range(o.values.shape[1])] for f,o in oracles.items()}
            outputs[group+('_STATIC' if static else '_FULL')]=dict(**prepared_artifact(segmenter,m,source_provenance,training_environment),projection_audit=projections,validation=trainer.validate(oracles['val']),fit_steps=trainer.fit_steps,cal_steps=trainer.cal_steps)
    return dict(models=outputs,oracle_query={f:evaluate_oracles(segmenter,data,o) for f,o in oracles.items()},basis_audit=baudit,cost=dict(COUNTS-before),identity_cost=dict(HASH_COST-identity_before))

def prepared_artifact(segmenter,method,source_provenance=None,training_environment=None):
    """Capture at source release; caller must preserve this trusted metadata."""
    if method is not None and (method.stage!='online' or any(p.requires_grad for p in method.parameters())):raise ValueError('freeze before source release')
    context=make_context(segmenter,method,source=source_provenance,expected_environment=training_environment)
    return dict(schema='R7_PREPARED_TENSORS_V2',weights=None if method is None else copy.deepcopy(method.state_dict()),binding=context,method_digest=context['payload']['method']['weights_sha256'])

def inference_from_tensors(segmenter,group,static,basis,weights,expected_context,ablation=None):
    from .host import OnlineHost
    validate_context(expected_context)
    if group=='C0':
        if static or basis is not None or weights is not None or ablation is not None:raise ValueError('C0 has no method assets')
        return OnlineHost(segmenter,expected_context=expected_context)
    m=new_method(group,basis,static);m.load_state_dict(weights,strict=True)
    # Check stored eta before freeze recomputes it; corrupt assets must not be repaired.
    if group=='B':
        from .numerics import normalize,temperature
        expected=1/(torch.linalg.matrix_norm(normalize(m.Hraw.double(),0),2).square()/temperature(m.cal_raw).double().square()+.1)
        if not torch.allclose(m.frozen_eta,expected,rtol=1e-10,atol=1e-12):raise ValueError('stored deployment eta')
    if m.digest()!=expected_context['payload']['method']['weights_sha256']:raise ValueError('training/inference asset binding')
    m.freeze()
    return OnlineHost(segmenter,m,ablation=ablation,expected_context=expected_context)
