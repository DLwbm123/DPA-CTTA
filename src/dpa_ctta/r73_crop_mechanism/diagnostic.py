"""Paired target quality and one-step branches on unchanged R72 crop settings."""
import torch
from ..r72_crop_target.method import Host as CropHost, configuration, matrices, warp, inverse, refine
from ..r19_model_only.method import equal
from ..joint_sensitivity.method import flatten

ARMS = ('NATIVE','RESAMPLE','CROP')
BRANCHES = ('NO_UPDATE',)+ARMS


class Host(CropHost):
    def _criterion(self,z,pseudo):
        loss=super()._criterion(z,pseudo)
        arm=self.config['crop_arm'];q=pseudo.detach()
        if arm=='NATIVE': target=q
        else:
            back=([warp(warp(q,a),inverse(a)) for a in matrices(q)]
                  if arm=='RESAMPLE' else self.crop_back)
            target=refine(q,back,self.crop_support).clamp(0,1)
        self.recorded_target=target.cpu().clone()
        self.recorded_native=q.cpu().clone()
        return loss


def pair(host,anchor,x,qualify=False):
    host.restore(anchor)
    no_update=host._readonly(host.native.model,x)
    masks=[(no_update[0]>=0).cpu()];targets=[];updates=[];details={};native_state=None
    for arm in ARMS:
        host.config['crop_arm']=arm;host.restore(anchor)
        before=flatten(host.native.params).clone()
        z,d=host.step(x);after=host.snapshot()
        if arm=='NATIVE':
            native_state=after;native_output=z.clone();q0=host.recorded_native.clone()
            weak=torch.stack(host.weak).clone()
        elif not torch.equal(host.recorded_native,q0) or not torch.equal(torch.stack(host.weak),weak):
            raise ValueError('same-state native targets differ across branches')
        if not equal(after['native_rng'],native_state['native_rng']):
            raise ValueError('strong augmentation RNG differs')
        delta=flatten(host.native.params)-before;updates.append(delta.detach().clone())
        denom=delta.norm()*updates[0].norm()
        details[arm]=dict(update_norm=float(delta.norm()),gradient_norm=d['diagnostics']['BN_gradient_norm'],
                         cosine_to_native=float(torch.dot(delta,updates[0])/denom) if denom>0 else None,
                         target_change=float((host.recorded_target-q0).abs().mean()),
                         preview_forwards=0 if arm=='NATIVE' else 2)
        masks.append(z[0]>=0);targets.append(host.recorded_target[0].clone())
        if qualify:
            host.restore(anchor);again,_=host.step(x)
            if not torch.equal(again,z) or not equal(host.snapshot(),after):
                raise ValueError('repeated branch state differs: '+arm)
    if qualify:
        host.restore(anchor);host.prepare_crops(x);host.config['crop_arm']='NATIVE'
        z,_=host.step(x)
        if not torch.equal(z,native_output) or not equal(host.snapshot(),native_state):
            raise ValueError('COMPUTE_ONLY differs from native after update')
        host.crop_back=None;host.crop_support=None
    host.restore(anchor)
    if not equal(host.snapshot(),anchor): raise ValueError('final anchor restoration failed')
    host.check_frozen(True)
    return torch.stack(masks).cpu().numpy(),torch.stack(targets).cpu().numpy(),details


def self_check():
    from ..r72_crop_target.method import self_check as prior_check
    prior_check()
    q=torch.tensor([.1,.7,.6,.2]);y=torch.tensor([0.,1.,0.,1.]);r=torch.tensor([.05,.9,.4,.8])
    delta=r-q
    improvement=(q-y).square()-(r-y).square()
    assert torch.allclose(improvement,2*delta*(y-q)-delta.square(),atol=1e-7)
    assert improvement.mean()>0
    old=(q>=.5)==y.bool();new=(r>=.5)==y.bool()
    assert int((~old&new).sum())==2 and int((old&~new).sum())==0
    import numpy as np
    from .run import target_statistics
    truth=np.zeros((2,16,16),dtype=bool);truth[:,4:12,4:12]=True
    fields=np.full((3,2,16,16),.2,dtype=np.float32);fields[2]=truth
    errors,corrected,spoiled=target_statistics(fields,truth)
    assert np.all(errors[0]>errors[2]) and np.array_equal(errors[0],errors[1])
    assert np.all(corrected[:2]>0) and not spoiled.any() and not corrected[2].any()
    print('SAME_STATE_MECHANISM_SELF_CHECK_PASS')
