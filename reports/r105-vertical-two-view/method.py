"""Change only the second strong branch from horizontal to vertical flip."""
import torch
from plain import close_enough
from dpa_ctta.r85_loss_aggregation.method import Host as Original,combine
from dpa_ctta.r20_model_only_search.method import weights_for
from dpa_ctta.b1_host import finite
class Two(Original):
    def __init__(self,state,cell,seed,identity,**kwargs):
        assert cell in ('H2','T2')
        self.axis=-1 if cell=='H2' else -2
        super().__init__(state,'VV',seed,identity,**kwargs)
    def _criterion(self,z,q):
        assert self.strong is not None and len(self.weak)==6
        pred=torch.stack(self.weak).sigmoid();w=weights_for(pred,.1,2.).to(q)
        assert torch.equal(pred.mean(0).to(q),q) and not q.requires_grad
        self.target_cache=(q.detach().cpu().clone(),w.detach().cpu().clone())
        self.extra=True
        try:other=torch.flip(self.native.model(torch.flip(self.strong,(self.axis,)))[0],(self.axis,))
        finally:self.extra=False;self.strong=None
        self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
        loss,parts=combine(z,other,q,w,'VV')
        self.diag.update(loss_parts=parts,joint_loss=float(loss.detach()),q_source='current',W_source='current_whole')
        finite(loss);return loss

def self_check():
    x=torch.arange(24).reshape(1,2,3,4)
    for axis in (-1,-2):assert torch.equal(x,torch.flip(torch.flip(x,(axis,)),(axis,)))
    assert not torch.equal(torch.flip(x,(-1,)),torch.flip(x,(-2,)))
