"""Reset reproduces fresh C under the continued arrival RNG, without reseeding."""
import copy
import torch
from .run import reset_episode
from ..r20_model_only_search.method import Host
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal
from ..r8_ba.rng import capture


def main():
    torch.set_num_threads(2);torch.manual_seed(42);model=Tiny();x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    cfg=dict(family='control',host='C',id='C',params={})
    h=Host(None,cfg,20260907,'check',device='cpu',model=copy.deepcopy(model));initial=h.snapshot();h.step(x)
    global_rng=capture();native_rng=copy.deepcopy(h.native.rng);reset_episode(h,initial)
    assert equal(capture(),global_rng) and equal(h.native.rng,native_rng)
    assert equal(h.snapshot()['parameters'],initial['parameters'])
    fresh=Host(None,cfg,20260907,'check',device='cpu',model=copy.deepcopy(model));fresh.native.rng=copy.deepcopy(native_rng)
    a,_=h.step(x.flip(-1));b,_=fresh.step(x.flip(-1));assert torch.equal(a,b)
    for key in ('parameters','adam','native_rng'):assert equal(h.snapshot()[key],fresh.snapshot()[key])
    h.check_frozen(True);fresh.check_frozen(True);h.close();fresh.close()
    print('PASS episodic parameters/Adam reset, RNG preservation and fresh-host next-output parity')


if __name__=='__main__':main()
