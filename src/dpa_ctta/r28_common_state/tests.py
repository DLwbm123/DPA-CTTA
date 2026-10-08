"""One runnable check for replay, independent noise and state-preserving probes."""
import copy
import torch
from .method import Host, actions_for
from .audit import replay
from ..r24_c_context.method import Host as Anchor
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    r=replay([0,0,0,0],[.1,.2,.3,.4])
    assert r['tied_max']==1 and r['actual_gain_pp']<0 and abs(r['uniform_tie_gain_pp'])<1e-12
    a=actions_for(17,0,4);torch.randint(4,())
    assert torch.equal(a,actions_for(17,0,4)) and not torch.equal(a,actions_for(17,0,5))
    model=Tiny();c=dict(id='GREEDY',family='GREEDY',host='C',params={},components={})
    h=Host(None,c,17,'test',device='cpu',model=copy.deepcopy(model))
    ref=Anchor(None,dict(c,id='ANCHOR'),17,'test',device='cpu',model=copy.deepcopy(model))
    x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    h.step(x);ref.step(x)
    before=h.snapshot();m,d=h.probe(x.flip(-1),a);after=h.snapshot()
    h.restore(before);z,_=h.step(x.flip(-1))
    assert equal(after,h.snapshot()) and (m[0]==(z.numpy()[0]>=0)).all()
    zref,_=ref.step(x.flip(-1));assert torch.equal(z,zref)
    for k in ('parameters','adam','adapter','native_rng'):assert equal(h.snapshot()[k],ref.snapshot()[k]),k
    h.restore(before);m2,d2=h.probe(x.flip(-1),a)
    assert (m==m2).all() and d==d2 and h.visits==2 and not h.policy_optimizer.state
    h.check_frozen(True);h.close();ref.close();print('PASS: ties, independent noise, zero-action parity, full replay, probe isolation')


if __name__=='__main__':main()
