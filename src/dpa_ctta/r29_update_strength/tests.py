"""Assert true skip, half-step semantics and exact full/rollback behavior."""
import copy
import torch
from .method import Host
from ..r28_common_state.method import Host as Previous
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2);torch.manual_seed(17);m=Tiny()
    c=dict(id='GREEDY',family='GREEDY',host='C',params={},components={})
    h=Host(None,c,17,'test',device='cpu',model=copy.deepcopy(m))
    old=Previous(None,c,17,'test',device='cpu',model=copy.deepcopy(m))
    x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    h.step(x);old.step(x);before=h.snapshot()
    z=h.evaluate(x.flip(-1),1.);zold,_=old.step(x.flip(-1))
    assert torch.equal(z,zold) and equal(before,h.snapshot())
    calls=[];hook=h.native.base.register_step_pre_hook(lambda *a:calls.append(1))
    h.evaluate(x,0.);assert calls==[] and equal(before,h.snapshot())
    h.strength=1.;h.step(x);full=h.snapshot();h.restore(before)
    h.strength=.5;h.step(x);half=h.snapshot();h.restore(before);h.strength=1.
    assert half['adam']['param_groups'][0]['lr']==5e-5 and half['adam']['param_groups'][1]['lr']==1.5e-4
    assert equal(full['adam']['state'],half['adam']['state'])
    changed=False
    for family in ('parameters','adapter'):
        for k,v in before[family].items():
            df=full[family][k]-v;dh=half[family][k]-v
            assert torch.allclose(dh,.5*df,atol=1e-6,rtol=1e-4)
            changed|=bool((df!=0).any())
    assert changed
    masks=h.probe_strength(x);assert masks.shape==(3,2,512,512) and equal(before,h.snapshot())
    assert (masks==h.probe_strength(x)).all()
    hook.remove();h.check_frozen(True);h.close();old.close()
    print('PASS: full parity, true skip, both LR groups halved, same Adam moments, exact rollback')


if __name__=='__main__':main()
