"""Fixed-capacity whole-image replay; source-stratified Bernoulli coreset."""
import itertools
import math
import random
import torch
from .method import Host as AffineHost, entropy, strata, l2, normalize_parameters

KINDS=('DIVERSITY','RANDOM_MEMORY','FIFO','CURRENT')
CAPACITY=2
MICROBATCHES=3


@torch.no_grad()
def descriptor(source_logits):
    if source_logits.ndim!=4 or source_logits.shape[:2]!=(1,2):
        raise ValueError('one image, independent OD/OC logits required')
    if not torch.isfinite(source_logits).all():raise ValueError('nonfinite source logits')
    p=source_logits.detach().double().sigmoid().cpu()
    regions=strata(p)
    q=torch.arange(16,dtype=torch.int64)
    values=[];counts=[]
    for ch in range(2):
        for region in regions:
            v=p[0,ch][region[0,ch]]
            if not v.numel():raise ValueError('empty source descriptor stratum; no fallback')
            # Inverse empirical CDF at j/15, integer ranks: repeating every
            # observation does not alter the descriptor or add support.
            indices=((q*v.numel()+14)//15-1).clamp_min(0)
            values.append(torch.sort(v,stable=True).values[indices]);counts.append(v.numel())
    result=torch.stack(values)
    if not torch.isfinite(result).all():raise ValueError('nonfinite descriptor')
    return result,counts


def distance(a,b):
    if a.shape!=b.shape or a.dtype!=torch.float64 or b.dtype!=torch.float64:
        raise ValueError('matched FP64 source descriptors required')
    def h(p):return -torch.special.xlogy(p,p)-torch.special.xlogy(1-p,1-p)
    value=(h((a+b)/2)-(h(a)+h(b))/2).mean()/math.log(2)
    if not torch.isfinite(value) or float(value)<-1e-12:raise ValueError('invalid Bernoulli JSD')
    return math.sqrt(max(0.,float(value)))


class Memory:
    def __init__(self,kind):
        if kind not in KINDS:raise ValueError('unknown memory policy')
        self.kind=kind;self.entries=[];self.seen=0;self.rng=random.Random(20261018)

    def admit(self,x,d):
        self.seen+=1
        if self.kind=='CURRENT':return
        item=dict(uid=self.seen,x=x.detach().cpu().clone(),descriptor=d.detach().cpu().clone())
        if len(self.entries)<CAPACITY:self.entries.append(item);return
        if self.kind=='FIFO':self.entries=self.entries[1:]+[item]
        elif self.kind=='RANDOM_MEMORY':
            slot=self.rng.randrange(self.seen)
            if slot<CAPACITY:self.entries[slot]=item
        else:
            pool=self.entries+[item]
            # Capacity two: max-min coreset is the farthest pair. Exact ties keep
            # the lexicographically newest arrival IDs; no case identifiers.
            pairs=list(itertools.combinations(pool,2))
            self.entries=list(max(pairs,key=lambda pair:(distance(pair[0]['descriptor'],pair[1]['descriptor']),
                tuple(sorted(v['uid'] for v in pair)))))
        self.entries.sort(key=lambda v:v['uid'])

    def batch(self,x):
        if not self.entries:return [x]*MICROBATCHES,[]
        old=[v['x'].to(x) for v in self.entries]
        if len(old)==1:old=old*2
        return [x]+old,[v['uid'] for v in self.entries]

    def exact(self,other):
        return (self.kind==other.kind and self.seen==other.seen and self.rng.getstate()==other.rng.getstate()
            and len(self.entries)==len(other.entries) and all(a['uid']==b['uid'] and
            torch.equal(a['x'],b['x']) and torch.equal(a['descriptor'],b['descriptor'])
            for a,b in zip(self.entries,other.entries)))


class Host(AffineHost):
    def __init__(self,base,kind):
        self.kind=kind
        super().__init__(base,'ENT_NORM')

    def reset(self):
        super().reset();self.memory=Memory(self.kind)

    def repeat_state_equal(self,other):
        if not self.memory.exact(other.memory) or self.segment_steps!=other.segment_steps:return False
        return all(torch.equal(self.opt.state[a][k],other.opt.state[b][k])
            for a,b in zip(self.params,other.params) for k in ('step','exp_avg','exp_avg_sq'))

    def step(self,x,xw,xs,z0,seed):
        desc,counts=descriptor(z0)
        before_count=len(self.memory.entries);seen=self.memory.seen
        inputs,ids=self.memory.batch(x)
        assert len(inputs)==MICROBATCHES and all(i<=seen for i in ids)
        self.opt.zero_grad(set_to_none=True);losses=[];pre=None
        for i,item in enumerate(inputs):
            z=self.forward(item)
            if i==0:pre=z.detach().clone()
            loss=(entropy(z.sigmoid())*math.log(2)).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite replay objective')
            losses.append(float(loss.detach()));(loss/MICROBATCHES).backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.params):
            raise ValueError('invalid replay affine gradient')
        grad=l2([p.grad for p in self.params])
        if grad<=1e-12:raise ValueError('zero replay gradient; no fallback')
        before=[p.detach().clone() for p in self.params];self.opt.step()
        with torch.no_grad():
            raw,actual,cosine=normalize_parameters(self.params,before)
            post=self.forward(x).detach()
        if raw<=1e-12:raise ValueError('zero proposal')
        self.total_steps+=1;self.segment_steps+=1
        if any(int(self.opt.state[p]['step'])!=self.segment_steps for p in self.params):raise ValueError('Adam age')
        names,buffers=dict(self.model.named_parameters()),dict(self.model.named_buffers())
        if any(names[n]._version!=v for n,v in self.fixed.items()):raise ValueError('fixed parameter changed')
        if any(not torch.equal(v,buffers[n]) for n,v in self.buffers.items()):raise ValueError('BN buffer changed')
        # Admission is after PRE/optimization/POST. Source output guides selection,
        # never supplies a training target or any future input.
        self.memory.admit(x,desc)
        bank=self.memory.entries
        log=dict(loss=sum(losses)/3,raw_step_norm=raw,actual_step_norm=actual,direction_cosine=cosine,
            norm_matched=abs(actual-.01)<=5e-7,zero_proposal=False,affine_gradient_norm=grad)
        return pre,post,torch.ones_like(pre,dtype=torch.bool),pre,dict(steps=[log],
            total_affine_norm=l2([p-v for p,v in zip(self.params,self.initial)]),optimizer_age=self.segment_steps,
            forwards=4,optimizer_steps=1,loss_backward_calls=3,diagnostic_anchor_gradient_calls=0,
            memory_seen_before=seen,memory_before=before_count,memory_after=len(bank),
            replay_slots=2 if ids else 0,padded_current_slots=2 if not ids else 0,
            unique_replay_entries=len(ids),replayed_arrival_ids=ids,
            memory_age_mean=sum(seen-i for i in ids)/len(ids) if ids else 0.,
            memory_pair_distance=distance(bank[0]['descriptor'],bank[1]['descriptor']) if len(bank)==2 else None,
            descriptor_counts=counts)
