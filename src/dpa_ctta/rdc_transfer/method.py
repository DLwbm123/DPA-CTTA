"""RDC equations (4), (5), (7), (8); not its semi-supervised training system.

Reference: https://openreview.net/pdf?id=SzNyAJVTuM (ICLR 2027 submission).
"""
import copy
import torch
from torch.nn import functional as F
from ..r37_cw_orientation.method import Host as RetainedHost, candidates

ARMS = ('NATIVE', 'VALUE', 'RANK')
AGENTS, ALPHA, COEFFICIENT, GRID = 64, 2., .1, 32


def configuration(arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    c = copy.deepcopy(candidates()[3])
    c.update(id='CW_LSO_'+arm, relation_arm=arm, relation_coefficient=COEFFICIENT)
    return c


@torch.no_grad()
def select_agents(features, seed, count=AGENTS):
    """Greedy min-max cosine selection; exclude already selected spatial indices."""
    if features.ndim != 2 or not 2 <= count <= features.shape[0]:
        raise ValueError('pixel x feature matrix and valid agent count required')
    unit = F.normalize(features.detach(), dim=1, eps=1e-8)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    current = int(torch.randint(len(unit), (1,), generator=generator))
    chosen = []
    maximum = unit.new_full((len(unit),), -float('inf'))
    for _ in range(count):
        chosen.append(current)
        maximum = torch.maximum(maximum, unit @ unit[current])
        maximum[chosen] = float('inf')
        current = int(maximum.argmin())
    return torch.tensor(chosen, device=features.device, dtype=torch.long)


def relation_loss(teacher, student, indices, arm):
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError('teacher/student spatial grids must match')
    # Cosine is used ONLY for selection. The paper's affinities are raw dot products.
    tw = teacher.detach() @ teacher.detach()[indices].T
    sw = student @ student[indices].T
    pairs = torch.triu_indices(len(indices), len(indices), 1, device=student.device)
    td = tw[:, pairs[0]] - tw[:, pairs[1]]
    sd = sw[:, pairs[0]] - sw[:, pairs[1]]
    if arm == 'RANK':
        loss = 1 - (torch.tanh(ALPHA*td/2) * torch.tanh(ALPHA*sd/2)).mean()
    elif arm == 'VALUE':
        loss = F.kl_div(sw.log_softmax(-1), tw.softmax(-1), reduction='batchmean')
    else:
        raise ValueError('unknown relation loss')
    diagnostics = dict(relation_loss=float(loss.detach()),
        teacher_gap_mean=float(td.abs().mean()),
        teacher_tanh_strength=float(torch.tanh(ALPHA*td/2).abs().mean()),
        rank_disagreement=float((td.sign() != sd.detach().sign()).float().mean()),
        student_tanh_saturation=float((torch.tanh(ALPHA*sd.detach()/2).abs() > .99).float().mean()))
    return loss, diagnostics


class Host(RetainedHost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation_teacher = self.relation_student = None
        if self.config['relation_arm'] != 'NATIVE':
            if not hasattr(self.native.model, 'seg_head'):
                raise ValueError('verified pre-classifier feature required')
            self.handles.append(self.native.model.seg_head.register_forward_pre_hook(self._capture_features))

    def _capture_features(self, module, args):
        if not self.collect:
            return
        # First original weak view and the native strong photometric view share coordinates.
        if len(self.weak) not in (0, 6):
            return
        if args[0].shape[-2:] != (512, 512):
            raise ValueError('registered 512 square pre-classifier grid required')
        h = F.avg_pool2d(args[0], 512//GRID, 512//GRID)[0].flatten(1).T
        if len(self.weak) == 0:
            self.relation_teacher = h.detach()
        elif torch.is_grad_enabled():
            self.relation_student = h

    def _criterion(self, z, pseudo):
        base = super()._criterion(z, pseudo)
        if self.config['relation_arm'] == 'NATIVE':
            return base
        tw, sw = self.relation_teacher, self.relation_student
        if tw is None or sw is None or tw.requires_grad or not sw.requires_grad:
            raise ValueError('missing detached teacher or differentiable student')
        indices = select_agents(tw, self.seed + self.visits)
        extra, diagnostics = relation_loss(tw, sw, indices, self.config['relation_arm'])
        if not torch.isfinite(extra):
            raise ValueError('nonfinite relation objective')
        self.diag.update(diagnostics, agent_count=len(indices), relation_grid=GRID)
        coefficient = float(self.config['relation_coefficient'])
        # Zero-control must preserve the native graph and optimizer exactly.
        return base if coefficient == 0 else base + coefficient*extra

    def step(self, x):
        self.relation_teacher = self.relation_student = None
        try:
            return super().step(x)
        finally:
            self.relation_teacher = self.relation_student = None


def self_check():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    f = torch.randn(100, 8)
    state = torch.random.get_rng_state().clone()
    indices = select_agents(f, 17)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(indices.unique()) == AGENTS
    assert torch.equal(indices, select_agents(f, 17))
    assert torch.equal(indices, select_agents(f*3, 17))
    # Independently compute the finite Kendall surrogate on a tiny example.
    tw = torch.tensor([[1., 0.], [0., 1.], [.4, .8]], requires_grad=True)
    sw = torch.tensor([[.1, .9], [.9, .1], [.7, .3]], requires_grad=True)
    ix = torch.tensor([0, 1, 2])
    loss, _ = relation_loss(tw, sw, ix, 'RANK')
    expected = sum(1-torch.tanh((tw[j] @ (tw[p]-tw[q])).detach()) *
                   torch.tanh(sw[j] @ (sw[p]-sw[q]))
                   for j in range(3) for p in range(3) for q in range(p+1,3))/9
    assert torch.allclose(loss, expected)
    gt, gs = torch.autograd.grad(loss, (tw, sw), allow_unused=True)
    assert gt is None and torch.isfinite(gs).all() and gs.norm() > 0
    moved, _ = relation_loss(tw, sw.detach()-.01*gs, ix, 'RANK')
    assert moved < loss.detach()
    same, _ = relation_loss(tw, tw.detach(), ix, 'VALUE')
    assert abs(float(same)) < 1e-6
    # A teacher tie supplies no direction, even though the rank loss has a constant 1.
    tie = torch.ones(3, 2)
    student = torch.randn(3, 2, requires_grad=True)
    tied, _ = relation_loss(tie, student, ix, 'RANK')
    assert torch.equal(torch.autograd.grad(tied, student)[0], torch.zeros_like(student))
    print('RDC_TRANSFER_SELF_CHECK_PASS')
