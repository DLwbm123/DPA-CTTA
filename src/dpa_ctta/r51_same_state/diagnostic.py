"""One fixed-state image, full loss gradients, native proposal and seven shadows."""
import copy
import math
import torch
from torch.nn import functional as F

from ..b1_host import official, finite
from ..host_diagnostic import rng, restore
from ..hosts.vptta import model_input_from_pixels
from ..joint_sensitivity.method import Sensor, photo_probe, filter_update, flatten
from ..r19_model_only.method import equal
from ..r20_model_only_search.method import weights_for
from ..r36_incremental_modules.method import local_structure_loss
from .geometry import proposals, check_committed

BRANCHES = ('no_update', 'native', 'joint', 'radial', 'null0', 'null1', 'null2', 'null3')


@torch.no_grad()
def write_delta(params, before, delta):
    offset = 0
    for p, old in zip(params, before):
        p.copy_(old+delta[offset:offset+p.numel()].reshape_as(p).to(p))
        offset += p.numel()
    if offset != delta.numel():
        raise ValueError('full parameter vector length')
    return flatten([p-old for p,old in zip(params,before)])


def mask(logits):
    return (logits.detach().float().sigmoid() >= .5).cpu()[0]


@torch.enable_grad()
def one_pair(host, pixels):
    """The caller restores the anchor before every call; no trajectory is advanced."""
    if pixels.shape != (1,3,512,512) or pixels.requires_grad:
        raise ValueError('one frozen current RGB image')
    n = host.native
    restore(n.rng)
    before = [p.detach().clone() for p in n.params]
    counts_before = n.counts.copy()
    n.last = {}; host.diag = {}; host.weak = []; host.collect = True
    x = model_input_from_pixels(pixels, 'fundus')
    try:
        z0 = n.model(x.to(host.device))[0]
        rotate = official().Rotate_and_Flip()
        with torch.no_grad():
            for index in range(5):
                n.model(rotate(x.to(host.device), index))
    finally:
        host.collect = False
    predictions = torch.stack(host.weak).sigmoid()
    target = predictions.mean(0).to(z0).detach()
    sensor = Sensor(z0, predictions.to(z0))
    outputs = {}
    scores = sensor(z0)
    anatomy = [flatten(torch.autograd.grad(scores[c], n.params, retain_graph=c == 0)) for c in range(2)]
    del scores, z0
    summed = (anatomy[0]+anatomy[1])/math.sqrt(2)
    acquisition = []
    saved_rng = rng()
    try:
        for index in range(2):
            transformed = model_input_from_pixels(photo_probe(pixels,index),'fundus').to(host.device)
            logits = n.model(transformed)[0]
            response = sensor(logits).sum()/math.sqrt(2)
            acquisition.append(flatten(torch.autograd.grad(response,n.params))-summed)
            del logits, response
    finally:
        restore(saved_rng)
    n.base.zero_grad()
    strong = official().augmentation_strong_style({'data':x.numpy().copy()})
    strong = official().normalize_image_to_0_1(torch.from_numpy(strong).float().to(host.device))
    logits = n.model(strong)[0]
    total = host._criterion(logits,target)
    wconfig = host.modules['W']
    reliability = weights_for(predictions,wconfig['variance_temperature'],wconfig['boundary_boost']).to(logits)
    base_loss = (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*reliability).mean()
    lso_loss, _ = local_structure_loss(logits,target,reliability)
    if not torch.equal(total,base_loss+.1*lso_loss):
        raise ValueError('CW and LSO loss decomposition differs from original criterion')
    g0 = flatten(torch.autograd.grad(base_loss,n.params,retain_graph=True))
    gs = flatten(torch.autograd.grad(lso_loss,n.params,retain_graph=True))
    total.backward()
    gradient = flatten([p.grad for p in n.params])
    decomposition = float((gradient.double()-g0.double()-.1*gs.double()).norm()/gradient.double().norm().clamp_min(1e-30))
    if decomposition > 1e-5:
        raise ValueError('full-gradient decomposition tolerance exceeded')
    n.last['cal_consis_loss'] = float(total.detach())
    n.base.step()
    native = flatten([p-old for p,old in zip(n.params,before)])
    beta2 = n.base.param_groups[0]['betas'][1]; eps=n.base.param_groups[0]['eps']
    metric = flatten([(n.base.state[p]['exp_avg_sq']/(1-beta2**int(n.base.state[p]['step']))).sqrt()+eps for p in n.params])
    filtered,jdiag = filter_update(native,metric,torch.stack(acquisition,1),torch.stack(anatomy,1),1.,shared_span=True)
    joint = write_delta(n.params,before,filtered)
    controls,matching = proposals(native,joint,metric,g0,gs)
    evaluation_rng = rng()
    buffers = {k:v.clone() for k,v in n.model.named_buffers()}
    modes = {k:m.training for k,m in n.model.named_modules()}
    candidates = {'no_update':torch.zeros_like(native),'native':native,'joint':joint,**controls}
    match_errors, branch_geometry, actuals = {}, {}, {}
    with torch.no_grad():
        for name,delta in candidates.items():
            restore(evaluation_rng)
            for k,v in n.model.named_buffers():v.copy_(buffers[k])
            for k,m in n.model.named_modules():m.training=modes[k]
            actual = write_delta(n.params,before,delta)
            if name == 'joint' and not torch.equal(actual,joint):
                raise ValueError('J displacement did not survive repeated representable write')
            if name not in ('no_update','native','joint'):
                match_errors[name] = check_committed(native,joint,actual,metric,g0,gs,radial=name=='radial')
            out = n.model(x.to(host.device))[0]
            outputs[name] = mask(out)
            actuals[name] = actual.detach().cpu()
            branch_geometry[name] = dict(metric_norm=float((metric.double().sqrt()*actual.double()).norm()),
                metric_modification_norm=float((metric.double().sqrt()*(actual.double()-native.double())).norm()),
                euclidean_norm=float(actual.norm()),base_dot=float(g0.double()@actual.double()),
                lso_dot=float(gs.double()@actual.double()))
    # Expose native state for qualification; all branches have used identical Adam moments.
    write_delta(n.params,before,native)
    n.steps += 1; host.visits += 1; n.rng = rng()
    finite([n.params,n.base.state,metric,g0,gs])
    counts = {k:n.counts[k]-counts_before[k] for k in n.counts}
    expected = dict(forwards=17,backwards=7,base_adam=1,perturb=0,restore=0)
    if counts != expected:
        raise ValueError(f'counterfactual physical counts: {counts}')
    host.check_frozen(True)
    return outputs,dict(counts=counts,joint=jdiag,matching=matching,committed_matching=match_errors,
                        gradient_decomposition_relative_error=decomposition,branch_geometry=branch_geometry,
                        base_loss=float(base_loss.detach()),LSO=float(lso_loss.detach()),
                        base_gradient_norm=float(g0.norm()),lso_gradient_norm=float(gs.norm()),
                        total_gradient_norm=float(gradient.norm()),structure_evidence=sensor.evidence_mean),dict(
                        base_gradient=g0.cpu(),lso_gradient=gs.cpu(),total_gradient=gradient.cpu(),
                        metric=metric.cpu(),native=native.cpu(),joint=joint.cpu(),
                        acquisition=torch.stack(acquisition,1).cpu(),anatomy=torch.stack(anatomy,1).cpu(),
                        actuals=actuals)


def assert_native_equivalence(reference, candidate):
    # Counts differ because the diagnostic observes extra VJPs and predictions.
    for key in ('parameters','buffers','gradients','adam','grata','native_rng','steps','visits','modes','module_frequency'):
        if not equal(reference[key],candidate[key]):
            raise ValueError('diagnostic changed native state: '+key)
