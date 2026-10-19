"""Continuous BN trajectories with Adam-history and q/W-source interventions."""
import torch
from ..r79_mechanism_control.method import Host as Previous, ARMS, SEEDS, style
from ..b1_host import official,finite
from ..hosts.vptta import model_input_from_pixels

CELLS=('00','10','01')


class Host(Previous):
    def __init__(self,*args,cell='11',**kwargs):
        if cell not in (*CELLS,'11'):raise ValueError('unregistered factorial cell')
        self.cell=cell;self.donor_student=None
        super().__init__(*args,**kwargs)
        self.initial_affine=[p.detach().clone() for p in self.native.params]
        self.handles.append(self.native.model.register_forward_pre_hook(self._donor_input))

    @torch.no_grad()
    def _copy_affine(self,values):
        for p,v in zip(self.native.params,values):p.copy_(v)

    def _donor_input(self,m,args):
        if self.cell[1]=='0' and self.collect and not self.weak and not torch.is_grad_enabled():
            if self.donor_student is not None:raise ValueError('nested initial donor')
            self.donor_student=[p.detach().clone() for p in self.native.params]
            self._copy_affine(self.initial_affine)

    def _weak_hook(self,m,args,out):
        super()._weak_hook(m,args,out)
        if self.donor_student is not None and len(self.weak)==6:
            if torch.is_grad_enabled():raise ValueError('donor entered training graph')
            self._copy_affine(self.donor_student);self.donor_student=None

    def intervention_step(self,x,raw):
        if self.donor_student is not None:raise ValueError('unclosed donor state')
        if self.cell[0]=='0':
            # Adam age belongs to the intervention; visit/physical counters remain continuous.
            self.native.base.state.clear();self.native.opt.state.clear();self.native.steps=0
        try:
            out,detail=self.keyed_step(x,raw)
            if self.donor_student is not None:raise ValueError('six weak donor forwards incomplete')
            if self.cell[0]=='0' and any(int(s['step'])!=1 for s in self.native.base.state.values()):
                raise ValueError('fresh Adam age drift')
            return out,detail
        finally:
            if self.donor_student is not None:
                self._copy_affine(self.donor_student);self.donor_student=None

    @torch.no_grad()
    def explicit_initial_targets(self,x):
        """Qualification oracle only; complete state restoration, no update."""
        from ..r20_model_only_search.method import weights_for
        saved=self.snapshot()
        try:
            self.collect=False;self._copy_affine(self.initial_affine)
            z=model_input_from_pixels(x,'fundus').to(self.device);transform=official().Rotate_and_Flip()
            preds=[self.native.model(z)[0].detach().cpu()]
            for k in range(5):preds.append(transform.inverse(self.native.model(transform(z,k))[0],k).detach().cpu())
            preds=torch.stack(preds).sigmoid();q=preds.mean(0);w=weights_for(preds,.1,2.)
            finite(q);finite(w);return q,w
        finally:self.restore(saved)


def self_check():
    from ..r79_mechanism_control.method import self_check as old_check
    old_check()
    # Real Adam: an empty state is age one again, while the current parameter survives.
    p=torch.nn.Parameter(torch.tensor([2.]));a=torch.optim.Adam([p],lr=1e-4)
    p.grad=torch.ones_like(p);a.step();current=p.detach().clone();a.state.clear()
    assert torch.equal(p,current)
    p.grad=torch.ones_like(p);a.step();assert int(a.state[p]['step'])==1
    assert len(CELLS)*len(ARMS)*len(SEEDS)*2*192==10368
    print('FACTORIAL_CPU_SELF_CHECK_PASS',flush=True)
