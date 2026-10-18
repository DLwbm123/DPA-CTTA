"""Preserve the current native score mean, without changing R75's whitening map."""
import copy
import torch
from ..r75_dense_wcw.method import Readouts as Original, self_check as original_check

ARMS=('NATIVE','ORIGINAL_WITHIN','SHIFT_ONLY','MEAN_PRESERVE')


def decompose(native,whitened):
    delta=(whitened[:,:,8::16,8::16]-native[:,:,8::16,8::16]).mean((-2,-1),keepdim=True)
    return native+delta,whitened-delta,delta


class Readouts(Original):
    @torch.no_grad()
    def emit(self,features,native):
        original,details=super().emit(features,native)
        whitened=original['WITHIN_STREAM']
        shift,preserved,delta=decompose(native,whitened)
        corrected_mean=preserved[:,:,8::16,8::16].mean((-2,-1))
        original_mean=native[:,:,8::16,8::16].mean((-2,-1))
        if not torch.allclose(corrected_mean,original_mean,atol=2e-6,rtol=2e-6):
            raise ValueError('current native score mean preservation failed')
        out=dict(NATIVE=native,ORIGINAL_WITHIN=whitened,SHIFT_ONLY=shift,MEAN_PRESERVE=preserved)
        diagnostics={a:copy.deepcopy(details['WITHIN_STREAM']) for a in ARMS[1:]}
        for k in range(2):
            if not diagnostics['MEAN_PRESERVE'][k]['active']:continue
            diagnostics['MEAN_PRESERVE'][k]['sampled_corrected_mean_logit']=float(corrected_mean[0,k])
            diagnostics['SHIFT_ONLY'][k].update(gamma=1.,operator_change=0.,
                sampled_corrected_mean_logit=float(original_mean[0,k]+delta[0,k,0,0]))
        return out,diagnostics


def self_check():
    original_check()
    torch.manual_seed(20261018)
    native=torch.randn(1,2,64,64,dtype=torch.float64)
    whitened=native*1.7-2.9
    shifted,preserved,delta=decompose(native,whitened)
    assert torch.allclose(preserved[:,:,8::16,8::16].mean((-2,-1)),
                          native[:,:,8::16,8::16].mean((-2,-1)),atol=1e-12)
    assert torch.allclose(shifted-native,delta.expand_as(native),atol=1e-12)
    assert torch.allclose(preserved-whitened,-delta.expand_as(native),atol=1e-12)
    assert torch.allclose(preserved[:,:,:,1:]-preserved[:,:,:,:-1],
                          whitened[:,:,:,1:]-whitened[:,:,:,:-1],atol=1e-12)
    _,identity,_=decompose(native,native)
    assert torch.equal(identity,native)
    print('MEAN_TRANSPORT_DECOMPOSITION_SELF_CHECK_PASS',flush=True)
