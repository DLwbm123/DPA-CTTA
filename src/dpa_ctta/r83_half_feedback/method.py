"""One fixed OD half blend, current OC and original current W."""
import torch
from ..r81_qw_factorial.method import Host as Previous,ARMS,SEEDS,style,reference_criterion as old_reference
CELLS=('01',)


class Host(Previous):
    def _criterion(self,z,q_native):
        if self.qw=='01':
            if self.initial_predictions is None:raise ValueError('missing initial reference')
            current=torch.stack(self.weak).sigmoid()
            self.initial_predictions=torch.cat(((self.initial_predictions[:,:,:1]+current[:,:,:1])*.5,current[:,:,1:]),dim=2)
        loss=super()._criterion(z,q_native)
        if self.qw=='01':self.diag['q_source']='OD_HALF_OC1'
        return loss


def reference_criterion(host,q0,w0,qw,initial=None):
    if qw!='01':return old_reference(host,q0,w0,qw)
    def criterion(z,native_q):
        current=torch.stack(host.weak).sigmoid()
        mixed=torch.cat((((initial[:,:,:1]+current[:,:,:1])*.5).mean(0).to(z),native_q[:,1:]),dim=1)
        return old_reference(host,mixed,w0,'01')(z,native_q)
    return criterion


def self_check():
    from ..r81_qw_factorial.method import self_check as previous_check
    previous_check()
    a=torch.zeros(6,1,2,8,8);b=torch.ones_like(a)
    p=torch.cat(((a[:,:,:1]+b[:,:,:1])*.5,b[:,:,1:]),dim=2).mean(0)
    assert (p[:,:1]==.5).all() and (p[:,1:]==1).all()
    assert len(CELLS)*len(ARMS)*len(SEEDS)*2*192==3456
    print('HALF_FEEDBACK_CPU_SELF_CHECK_PASS',flush=True)
