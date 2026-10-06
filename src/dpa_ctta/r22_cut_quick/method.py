from collections import deque
import numpy as np
from scipy.ndimage import label
import torch
from torch.nn import functional as F
from ..r20_model_only_search.method import Host as Base
from ..r7_shared.context import tensor_digest


def nested(q):
    return torch.cat((torch.maximum(q[:, :1], q[:, 1:]), q[:, 1:]), 1)


def classes(q):
    return (q >= .5).long().sum(1)


def probabilities(q):
    return torch.cat((1-q[:, :1], q[:, :1]-q[:, 1:], q[:, 1:]), 1)


def neighborhood(features, bank, size):
    """Same frozen-feature mutual-kNN graph for both predictions; past images only."""
    if len(bank) < 2:
        return None, None
    query = F.normalize(F.adaptive_avg_pool2d(features, (32, 32)).flatten(2)[0].T, dim=1)
    keys = torch.cat([b['features'] for b in bank]).to(query)
    sim = query @ keys.T
    k = min(16, len(keys)); values, idx = sim.topk(k, dim=1)
    reverse = sim.topk(min(16, len(query)), dim=0).indices
    mutual = torch.zeros_like(sim, dtype=torch.bool).scatter_(0, reverse, True).gather(1, idx)
    weight = ((values-values[:, :1])/.1).exp()*mutual
    denom = weight.sum(1, keepdim=True).clamp_min(1e-8)
    votes = []
    for name in ('student', 'source'):
        labels = torch.cat([b[name] for b in bank]).to(query.device)[idx]
        vote = torch.stack([(weight*(labels == c)).sum(1)/denom[:, 0] for c in range(3)], 1)
        votes.append(F.interpolate(vote.T.reshape(1,3,32,32), size=size, mode='bilinear', align_corners=False))
    support = F.interpolate((mutual.sum(1)>=4).float().reshape(1,1,32,32), size=size, mode='nearest')
    return votes, support


def judge(student, source, votes, support):
    """Fixed region rules. Scores are detached and never enter the training loss."""
    a, b = classes(student), classes(source)
    regions, n = label((a != b)[0].cpu().numpy())
    sizes = np.bincount(regions.ravel(), minlength=n+1)
    masks = [(student[0]>=.5).cpu().numpy(), (source[0]>=.5).cpu().numpy()]
    outputs = {name:masks[0].copy() for name in ('cut', 'confidence', 'entropy')}
    collapsed = any(masks[0][k].any() and not masks[1][k].any() for k in (0,1))
    maps = {}
    for name, q, y in [('student',student,a), ('source',source,b)]:
        pi = probabilities(q).clamp_min(1e-8)
        maps[name+'_confidence'] = pi.gather(1,y[:,None])[0,0].cpu().numpy()
        maps[name+'_entropy'] = (-(pi*pi.log()).sum(1))[0].cpu().numpy()
    maps['confidence'] = maps['source_confidence']-maps['student_confidence']
    maps['entropy'] = maps['student_entropy']-maps['source_entropy']
    if votes is None:
        maps['cut'] = np.zeros_like(regions, dtype=float); valid = np.zeros_like(regions, dtype=bool)
    else:
        # z_student - z_source = neighbor support(source label) - support(student label).
        maps['cut'] = (votes[1].gather(1,b[:,None])-votes[0].gather(1,a[:,None]))[0,0].cpu().numpy()
        valid = support[0,0].cpu().numpy() > 0
    records=[]
    for i in range(1,n+1):
        if sizes[i] < 32: continue
        region=regions==i; fraction=float(valid[region].mean())
        eligible=fraction>=.5 and not collapsed
        scores={name:float(maps[name][region].mean()) for name in outputs}
        choices={name:bool(eligible and score>.05) for name,score in scores.items()}
        for name,take in choices.items():
            if take: outputs[name][:,region]=masks[1][:,region]
        records.append(dict(region=i,area=int(sizes[i]),eligible=eligible,support_fraction=fraction,scores=scores,choices=choices))
    return outputs, records, collapsed


class Host(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.f0=self._new_model(self.initial,adaptive=False)
        self.handles.append(self.f0.up3.register_forward_hook(self._feature))
        self.f0_digest=tensor_digest(list(self.f0.state_dict().items()))
        self.bank=deque(maxlen=8)

    def _prepare_prototypes(self,x):
        self.current_proto={}
        self.source=nested(self._readonly(self.f0,x).sigmoid())
        self.features=self.feature.clone()

    def _criterion(self,z,q_native):
        loss=super()._criterion(z,q_native)
        with torch.no_grad():
            self.student=nested(q_native.detach())
            self.stable=(torch.stack(self.weak).sigmoid().var(0,unbiased=False)<=.001)
            votes,support=neighborhood(self.features,self.bank,q_native.shape[-2:])
            self.outputs,self.regions,self.collapsed=judge(self.student,self.source,votes,support)
            self.diag.update(judge_regions=len(self.regions),eligible_regions=sum(r['eligible'] for r in self.regions),history_images=len(self.bank),collapse_guard=self.collapsed)
        return loss

    def step(self,x):
        z,t=super().step(x)
        self.bank.append(dict(features=F.normalize(F.adaptive_avg_pool2d(self.features,(16,16)).flatten(2)[0].T,dim=1).cpu(),student=classes(F.adaptive_avg_pool2d(self.student,(16,16))).flatten().cpu(),source=classes(F.adaptive_avg_pool2d(self.source,(16,16))).flatten().cpu()))
        return z,t
