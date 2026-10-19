"""Separate q and whole-W provenance; continuous BN and Adam in every cell."""
import torch
from torch.nn import functional as F
from ..r80_history_factorial.method import Host as Previous
from ..r79_mechanism_control.method import ARMS,SEEDS,style
from ..r20_model_only_search.method import weights_for
from ..r36_incremental_modules.method import local_structure_loss
from ..b1_host import official,finite
from ..hosts.vptta import model_input_from_pixels

CELLS=('10','01')


def loss_parts(z,q,w):
    bce=(F.binary_cross_entropy_with_logits(z,q,reduction='none')*w).mean()
    orientation,magnitude=local_structure_loss(z,q,w)
    return bce+.1*orientation,dict(BCE=float(bce.detach()),LSO=float(orientation.detach()),LSM=float(magnitude.detach()))


class Host(Previous):
    def __init__(self,*args,qw='11',**kwargs):
        if qw not in ('00','10','01','11'):raise ValueError('unregistered q/W cell')
        self.qw=qw;self.initial_predictions=None
        # The diagonal implementations remain the retained original endpoints.
        super().__init__(*args,cell='10' if qw=='00' else '11',**kwargs)

    @torch.no_grad()
    def initial_weak(self,x):
        saved=self.snapshot()
        try:
            self.collect=False;self._copy_affine(self.initial_affine)
            z=model_input_from_pixels(x,'fundus').to(self.device);transform=official().Rotate_and_Flip()
            preds=[self.native.model(z)[0].detach().cpu()]
            for k in range(5):preds.append(transform.inverse(self.native.model(transform(z,k))[0],k).detach().cpu())
            out=torch.stack(preds).sigmoid();finite(out);return out
        finally:self.restore(saved)

    def intervention_step(self,x,raw):
        if self.initial_predictions is not None:raise ValueError('stale initial prediction bank')
        try:
            if self.qw in CELLS:self.initial_predictions=self.initial_weak(x)
            return super().intervention_step(x,raw)
        finally:self.initial_predictions=None

    def _criterion(self,z,q_native):
        if self.qw not in CELLS:return super()._criterion(z,q_native)
        current=torch.stack(self.weak).sigmoid();initial=self.initial_predictions
        if initial is None or len(self.weak)!=6:raise ValueError('two complete six-weak sources required')
        if not torch.equal(current.mean(0).to(q_native),q_native):raise ValueError('current target capture mismatch')
        qp=current if self.qw[0]=='1' else initial;wp=current if self.qw[1]=='1' else initial
        q=qp.mean(0).to(z).detach();w=weights_for(wp,.1,2.).to(z)
        self.target_cache=(q.cpu().clone(),w.cpu().clone())
        def at(value):
            loss,parts=loss_parts(value,q,w)
            self.diag.update(pseudo_loss=parts['BCE'],LSO=parts['LSO'],LSM=parts['LSM'],module_loss=parts['LSO'],module_coefficient=.1)
            return loss,{k:parts[k] for k in ('BCE','LSO')}
        if self.arm=='NATIVE':self._tag(z,'original',1.);loss,parts=at(z)
        else:
            if self.strong is None:raise ValueError('missing original strong tensor')
            self.extra=True
            try:other=torch.flip(self.native.model(torch.flip(self.strong,(-1,)))[0],(-1,))
            finally:self.extra=False;self.strong=None
            if self.arm=='VERTEX':
                self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
                first,p=at(z);second,r=at(other);loss=(first+second)*.5
                parts={k:(p[k]+r[k])*.5 for k in p};self.diag['loss_branch0']=p;self.diag['loss_branch1']=r
            else:
                averaged=(z+other)*.5;self._tag(averaged,'barycenter',1.);loss,parts=at(averaged)
        qq=q.clamp(1e-6,1-1e-6)
        self.diag.update(loss_parts=parts,joint_loss=float(loss.detach()),pseudo_target_change=float((q-q_native).abs().mean()),
            target_entropy=float((-(qq*qq.log()+(1-qq)*(1-qq).log())).mean()),weight_channel_means=w.mean((-2,-1)).flatten().tolist(),
            q_source=self.qw[0],W_source=self.qw[1])
        return loss


def reference_criterion(host,q0,w0,qw):
    """Qualification oracle uses direct original formulas and an independent host."""
    def criterion(z,native_q):
        pred=torch.stack(host.weak).sigmoid();q=native_q if qw[0]=='1' else q0.to(z)
        w=weights_for(pred,.1,2.).to(z) if qw[1]=='1' else w0.to(z)
        def one(value):
            base=(F.binary_cross_entropy_with_logits(value,q,reduction='none')*w).mean()
            orientation,_=local_structure_loss(value,q,w)
            return base+.1*orientation
        if host.arm=='NATIVE':return one(z)
        host.extra=True
        try:other=torch.flip(host.native.model(torch.flip(host.strong,(-1,)))[0],(-1,))
        finally:host.extra=False;host.strong=None
        return (one(z)+one(other))*.5 if host.arm=='VERTEX' else one((z+other)*.5)
    return criterion


def self_check():
    from ..r79_mechanism_control.method import self_check as original_check
    original_check()
    a=torch.linspace(.1,.9,6*2*16*16).reshape(6,1,2,16,16);b=a.flip(0)*.7
    q=a.mean(0);w=weights_for(b,.1,2.);z=torch.zeros_like(q,requires_grad=True)
    loss,parts=loss_parts(z,q,w);loss.backward()
    assert torch.isfinite(z.grad).all() and parts['BCE']>0
    assert len(CELLS)*len(ARMS)*len(SEEDS)*2*192==6912
    print('Q_W_FORMULA_AND_COUNTS_CPU_SELF_CHECK_PASS',flush=True)
