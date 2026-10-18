"""WCW equation 8/9, without global translation, for two binary heads.

Readouts never feed back into the host. Cumulative statistics therefore share
one native continuous trajectory, rather than four separately trained models.
"""
import torch
from torch.nn import functional as F

ARMS = ('NATIVE', 'DIAGONAL', 'WITHIN_STREAM', 'WITHIN_IMAGE')


class Moments:
    def __init__(self, dimensions, device):
        self.n = 0
        self.images = 0
        self.s = torch.zeros(dimensions, device=device, dtype=torch.float64)
        self.ss = torch.zeros(dimensions, dimensions, device=device, dtype=torch.float64)
        self.counts = torch.zeros(2, 2, device=device, dtype=torch.float64)
        self.sums = torch.zeros(2, 2, dimensions, device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, x, labels):
        if x.ndim != 2 or labels.shape != (2, len(x)) or labels.dtype != torch.bool:
            raise ValueError('two independent binary assignments required')
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite observed features')
        x = x.double()
        self.n += len(x)
        self.images += 1
        self.s += x.sum(0)
        self.ss += x.T @ x
        for head in range(2):
            for group in range(2):
                subset = x[labels[head] == bool(group)]
                self.counts[head, group] += len(subset)
                self.sums[head, group] += subset.sum(0)

    @torch.no_grad()
    def covariance(self, head):
        # Each head's two groups partition the same grid; no equal-class prior.
        counts = self.counts[head]
        safe = counts.clamp_min(1)
        means = self.sums[head] / safe[:, None]
        result = self.ss / self.n - torch.einsum('c,ci,cj->ij', counts/self.n, means, means)
        return (result + result.T) / 2


def transform(covariance, diagonal=False):
    scale = covariance.trace() / len(covariance)
    if not torch.isfinite(scale) or scale <= 1e-12:
        raise ValueError('degenerate covariance')
    ridge = .01 * scale
    if diagonal:
        return torch.diag((covariance.diag().clamp_min(0) + ridge).rsqrt())
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    tolerance = 1e-7 * scale
    if eigenvalues.min() < -tolerance:
        raise ValueError('within covariance is not positive semidefinite')
    return (eigenvectors * (eigenvalues.clamp_min(0) + ridge).rsqrt()) @ eigenvectors.T


class Readouts:
    def __init__(self, head):
        if (not isinstance(head, torch.nn.Conv2d) or head.kernel_size != (1, 1)
                or head.out_channels != 2 or head.groups != 1 or head.stride != (1, 1)):
            raise ValueError('verified two-output pointwise head required')
        self.head = head
        self.moments = Moments(head.in_channels, head.weight.device)

    @torch.no_grad()
    def emit(self, features, native):
        # Fixed grid point sampling avoids pooling across an anatomical boundary.
        # shortcut: pixels remain spatially correlated; grid size is not an iid sample certificate.
        small = features[0, :, 8::16, 8::16].flatten(1).T
        labels = native[0, :, 8::16, 8::16].flatten(1) >= 0
        current = Moments(self.head.in_channels, features.device)
        current.add(small, labels)
        self.moments.add(small, labels)
        output = {'NATIVE': native}
        diagnostics = {}
        for arm in ARMS[1:]:
            bank = current if arm == 'WITHIN_IMAGE' else self.moments
            corrected = []
            channels = []
            for k in range(2):
                supported = (self.moments.images > 24 and bank.n >= 2*self.head.in_channels
                             and bool((bank.counts[k] >= self.head.in_channels).all()))
                if not supported:
                    corrected.append(native[:, k:k+1])
                    channels.append(dict(active=False, counts=bank.counts[k].tolist()))
                    continue
                covariance = bank.covariance(k)
                mapping = transform(covariance, arm == 'DIAGONAL')
                original_norm = small.double().norm(dim=1).mean()
                changed_norm = (small.double() @ mapping.T).norm(dim=1).mean()
                gamma = original_norm / changed_norm.clamp_min(1e-12)
                mapping = (gamma * mapping).to(features)
                weight = (self.head.weight[k:k+1, :, 0, 0] @ mapping).unsqueeze(-1).unsqueeze(-1)
                bias = None if self.head.bias is None else self.head.bias[k:k+1]
                logits = F.conv2d(features, weight, bias)
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite corrected logits')
                corrected.append(logits)
                channels.append(dict(active=True, counts=bank.counts[k].tolist(), gamma=float(gamma),
                                     operator_change=float((mapping-torch.eye(len(mapping),device=mapping.device)).norm()),
                                     offdiagonal_fraction=float((covariance-torch.diag(covariance.diag())).norm()/covariance.norm().clamp_min(1e-12)),
                                     sampled_native_mean_logit=float(native[0,k,8::16,8::16].mean()),
                                     sampled_corrected_mean_logit=float(logits[0,0,8::16,8::16].mean())))
            output[arm] = torch.cat(corrected, dim=1)
            diagnostics[arm] = channels
        return output, diagnostics


def self_check():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(20261018)
    x = torch.randn(300, 5, generator=generator, dtype=torch.float64)
    x[:, 1] += .7*x[:, 0]
    labels = torch.stack((x[:, 0] > 0, x[:, 2] > .4))
    whole = Moments(5, 'cpu'); whole.add(x, labels)
    split = Moments(5, 'cpu')
    split.add(x[:110], labels[:, :110]); split.add(x[110:], labels[:, 110:])
    for k in range(2):
        direct = sum((group-group.mean(0)).T @ (group-group.mean(0)) for group in
                     [x[~labels[k]], x[labels[k]]]) / len(x)
        assert torch.allclose(whole.covariance(k), direct, atol=1e-12)
        assert torch.allclose(split.covariance(k), direct, atol=1e-12)
        mapping = transform(direct)
        ridge = .01*direct.trace()/5
        assert torch.allclose(mapping @ (direct+ridge*torch.eye(5,dtype=torch.float64)) @ mapping.T,
                              torch.eye(5,dtype=torch.float64),atol=1e-10)
    head = torch.nn.Conv2d(5, 2, 1).double()
    features = torch.randn(1,5,32,32,generator=generator,dtype=torch.float64)
    mapping = transform(whole.covariance(0))
    dense = F.conv2d(torch.einsum('ij,bjhw->bihw',mapping,features),head.weight[:1],head.bias[:1])
    effective = F.conv2d(features,(head.weight[:1,:,0,0]@mapping)[:,:,None,None],head.bias[:1])
    assert torch.allclose(dense,effective,atol=1e-12)
    before = generator.get_state().clone()
    bank = Moments(5,'cpu'); bank.add(x,torch.zeros_like(labels))
    assert torch.isfinite(bank.covariance(0)).all() and bank.counts[0,1] == 0
    assert torch.equal(generator.get_state(),before)
    # OD/OC overlap belongs to both positive binary groups, not a softmax class.
    assert ((labels[0]) & (labels[1])).any()
    print('DENSE_WCW_SELF_CHECK_PASS',flush=True)
