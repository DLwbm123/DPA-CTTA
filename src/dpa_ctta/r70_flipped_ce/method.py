"""TeSLA-inspired Bernoulli f-CE component on unchanged CW_LSO."""
import copy
import torch
from torch.nn import functional as F
from ..r37_cw_orientation.method import Host as RetainedHost, candidates
from ..r20_model_only_search.method import weights_for

ARMS = ('NATIVE', 'HARD', 'FCE')


def configuration(arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    config = copy.deepcopy(candidates()[3])
    config.update(id='CW_LSO_'+arm, loss_arm=arm)
    return config


def pixel_loss(logits, target, arm):
    q = target.detach()
    if arm == 'NATIVE':
        return F.binary_cross_entropy_with_logits(logits, q, reduction='none')
    if arm == 'HARD':
        return F.binary_cross_entropy_with_logits(logits, (q >= .5).to(logits), reduction='none')
    if arm != 'FCE':
        raise ValueError('unregistered arm')
    p = logits.sigmoid()
    return -p * (q + 1e-6).log() - (1-p) * (1-q + 1e-6).log()


class Host(RetainedHost):
    def _criterion(self, logits, pseudo):
        base = super()._criterion(logits, pseudo)
        arm = self.config['loss_arm']
        if arm == 'NATIVE':
            return base
        w = self.modules['W']
        reliability = weights_for(torch.stack(self.weak).sigmoid(),
                                  w['variance_temperature'], w['boundary_boost']).to(logits)
        old = pixel_loss(logits, pseudo, 'NATIVE')
        new = pixel_loss(logits, pseudo, arm)
        p, q = logits.detach().sigmoid(), pseudo.detach()
        g_old = p-q
        g_new = p-(q >= .5).to(p) if arm == 'HARD' else p*(1-p)*((1-q+1e-6).log()-(q+1e-6).log())
        self.diag.update(loss_arm=arm, replacement_loss=float((reliability*new).mean().detach()),
                         pixel_gradient_opposite_fraction=float((g_old*g_new < 0).float().mean()))
        return base + (reliability*(new-old)).mean()


def self_check():
    z = torch.tensor([[-2., .4, 2., 100., -100.]], requires_grad=True)
    q = torch.tensor([[.8, .6, .7, 1., 0.]], requires_grad=True)
    p = z.sigmoid().detach()
    loss = pixel_loss(z, q, 'FCE')
    grad = torch.autograd.grad(loss.sum(), (z, q), allow_unused=True)
    expected = p*(1-p)*((1-q.detach()+1e-6).log()-(q.detach()+1e-6).log())
    assert torch.isfinite(loss).all() and torch.isfinite(grad[0]).all()
    assert grad[1] is None and torch.allclose(grad[0], expected, atol=1e-7, rtol=1e-6)
    # The overconfident positive pixel reverses soft-BCE direction, unlike positive reweighting.
    assert grad[0][0,2] < 0 and p[0,2]-q[0,2] > 0
    neutral = pixel_loss(z, torch.full_like(q, .5), 'FCE')
    assert torch.equal(torch.autograd.grad(neutral.sum(), z)[0], torch.zeros_like(z))
    w = torch.tensor([[.2, 1., 2., .5, 1.3]])
    for arm in ARMS:
        a = pixel_loss(z, q, 'NATIVE')
        b = pixel_loss(z, q, arm)
        got = (w*a).mean()+(w*(b-a)).mean()
        want = (w*b).mean()
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)
        assert torch.allclose(torch.autograd.grad(got,z,retain_graph=True)[0],torch.autograd.grad(want,z,retain_graph=True)[0],atol=1e-7,rtol=1e-6)
    print('FCE_SELF_CHECK_PASS')
