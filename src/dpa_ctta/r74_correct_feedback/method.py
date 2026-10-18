"""Only the weighted BCE target changes; native W and LSO remain intact."""
import torch
from ..r37_cw_orientation.method import Host as RetainedHost
from ..r20_model_only_search.method import weights_for
from ..r72_crop_target.method import delta_loss
from ..r19_model_only.method import equal

BRANCHES = ('NO_UPDATE', 'NATIVE', 'CORRECT')


class Host(RetainedHost):
    feedback = None

    def _criterion(self, z, pseudo):
        base = super()._criterion(z, pseudo)
        q = pseudo.detach()
        target = q if self.feedback is None else self.feedback.to(q).detach()
        if target.shape != q.shape or not torch.isfinite(target).all():
            raise ValueError('feedback target shape/finite mismatch')
        self.actual_target = target.cpu().clone()
        self.native_target = q.cpu().clone()
        if self.feedback is None:
            return base
        if not ((target == 0) | (target == 1)).all():
            raise ValueError('feedback must be binary independent OD/OC channels')
        p = self.modules['W']
        w = weights_for(torch.stack(self.weak).sigmoid(), p['variance_temperature'], p['boundary_boost']).to(z)
        return base + delta_loss(z, q, target, w)


def readonly(host, state, pixels):
    host.restore(state)
    z = host._readonly(host.native.model, pixels).cpu()
    after = host.snapshot()
    # Native forward instrumentation counts probes; discard only those counters.
    after['counts'] = state['counts']
    if not equal(after, state):
        raise ValueError('read-only probe changed learning state or RNG')
    host.restore(state)
    return z


def branches(host, anchor, pixels, truth, qualify=False):
    from ..joint_sensitivity.method import flatten
    states = [anchor]
    outputs = [readonly(host, anchor, pixels)]
    targets = []
    info = []
    weak0 = None
    for name in BRANCHES[1:]:
        host.restore(anchor)
        host.feedback = truth if name == 'CORRECT' else None
        before = flatten(host.native.params).clone()
        z, detail = host.step(pixels)
        state = host.snapshot()
        if name == 'NATIVE':
            q0 = host.native_target.clone()
            weak0 = torch.stack(host.weak).clone()
            rng0 = state['native_rng']
        elif not torch.equal(host.native_target, q0) or not torch.equal(torch.stack(host.weak), weak0):
            raise ValueError('paired native targets or weak views differ')
        if not equal(state['native_rng'], rng0):
            raise ValueError('paired native RNG differs')
        target = host.actual_target.clone()
        if name == 'CORRECT' and not torch.equal(target, truth):
            raise ValueError('actual BCE target differs from feedback truth')
        info.append(dict(branch=name, update_norm=float((flatten(host.native.params)-before).norm()),
                         gradient_norm=detail['diagnostics']['BN_gradient_norm'],
                         native_pseudo_loss=detail['diagnostics']['pseudo_loss'],
                         total_training_loss=float(host.native.last['cal_consis_loss']),
                         LSO=detail['diagnostics']['LSO'], learning_rate=detail['diagnostics']['BN_lr']))
        states.append(state)
        outputs.append(z)
        targets.append(target)
        if qualify:
            host.restore(anchor)
            again, _ = host.step(pixels)
            if not torch.equal(z, again) or not equal(state, host.snapshot()) or not torch.equal(target, host.actual_target):
                raise ValueError('repeated branch differs: '+name)
    host.feedback = None
    host.restore(anchor)
    if not equal(host.snapshot(), anchor):
        raise ValueError('anchor restoration mismatch')
    host.check_frozen(True)
    return states, torch.cat(outputs), torch.cat(targets), info
