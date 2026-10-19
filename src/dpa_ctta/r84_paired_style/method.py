"""Same two-view VERTEX loss; second view uses the next existing style draw."""
import torch
from ..r79_mechanism_control.method import Host as Previous,SEEDS,NEXT,style
from ..r20_model_only_search.method import weights_for
from ..r81_qw_factorial.method import loss_parts
from ..b1_host import finite,official
ARMS=('VERTEX',)
CELLS=('01',)


class Host(Previous):
    def __init__(self,*args,qw=None,**kwargs):
        self.alternative=None
        super().__init__(*args,**kwargs)
        if self.arm!='VERTEX':raise ValueError('one fixed candidate')

    def intervention_step(self,x,raw,alternative):
        if self.alternative is not None:raise ValueError('stale second style')
        self.alternative=alternative
        try:return self.keyed_step(x,raw)
        finally:self.alternative=None

    def _criterion(self,z,q):
        if self.alternative is None or self.strong is None:raise ValueError('two strong sources required')
        w=weights_for(torch.stack(self.weak).sigmoid(),.1,2.).to(q)
        self.target_cache=(q.detach().cpu().clone(),w.detach().cpu().clone())
        other_input=official().normalize_image_to_0_1(torch.as_tensor(self.alternative,device=z.device,dtype=z.dtype))
        self.diag['second_style_equals_first_input']=torch.equal(other_input,self.strong)
        self.extra=True
        try:other=torch.flip(self.native.model(torch.flip(other_input,(-1,)))[0],(-1,))
        finally:self.extra=False;self.strong=None
        self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
        first,p=loss_parts(z,q,w);second,r=loss_parts(other,q,w);loss=(first+second)*.5
        self.diag.update(loss_parts={k:(p[k]+r[k])*.5 for k in ('BCE','LSO')},loss_branch0=p,loss_branch1=r,joint_loss=float(loss.detach()))
        finite(loss);return loss


def reference_criterion(host,alternative):
    """Direct original formula on an independently constructed host."""
    def criterion(z,q):
        from torch.nn import functional as F
        from ..r36_incremental_modules.method import local_structure_loss
        w=weights_for(torch.stack(host.weak).sigmoid(),.1,2.).to(q)
        host.extra=True
        try:other=torch.flip(host.native.model(torch.flip(official().normalize_image_to_0_1(torch.as_tensor(alternative,device=z.device,dtype=z.dtype)),(-1,)))[0],(-1,))
        finally:host.extra=False;host.strong=None
        host._tag(z,'vertex0',.5);host._tag(other,'vertex1',.5)
        def one(v):
            orientation,_=local_structure_loss(v,q,w)
            return (F.binary_cross_entropy_with_logits(v,q,reduction='none')*w).mean()+.1*orientation
        return (one(z)+one(other))*.5
    return criterion


def self_check():
    from ..r79_mechanism_control.method import self_check as previous_check
    previous_check()
    assert len(SEEDS)*2*192==1152 and len(ARMS)==1
    print('PAIRED_STYLE_CPU_SELF_CHECK_PASS',flush=True)
