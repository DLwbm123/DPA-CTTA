import copy
import torch
from .method import Host, judge, neighborhood, nested
from ..r20_model_only_search.method import Host as Base
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2)
    a=torch.full((1,2,16,16),.1);a[:,:,4:12,4:12]=.9
    b=a.clone();b[:,:,4:12,4:12]=.1;b[:,:,0:2,0:2]=.9
    source_votes=torch.zeros(1,3,16,16);source_votes[:,0]=1
    student_votes=torch.zeros_like(source_votes);student_votes[:,0]=1
    out,rr,blocked=judge(a,b,[student_votes,source_votes],torch.ones(1,1,16,16))
    assert any(r['choices']['cut'] for r in rr) and not blocked
    _,rr,_=judge(a,b,None,None);assert not any(r['eligible'] for r in rr)
    _,rr,blocked=judge(a,torch.full_like(a,.1),[student_votes,source_votes],torch.ones(1,1,16,16))
    assert blocked and not any(r['choices']['cut'] for r in rr)
    assert neighborhood(torch.ones(1,3,8,8),[],(16,16))==(None,None)
    torch.manual_seed(17);model=Tiny();x=torch.rand(1,3,512,512)
    cfg=dict(family='combo',host='C',params={},components={'W':dict(variance_temperature=.1,boundary_boost=2)})
    h=Host(None,cfg,20260907,'check',device='cpu',model=copy.deepcopy(model))
    ref=Base(None,cfg,20260907,'check',device='cpu',model=copy.deepcopy(model))
    for i in range(3):
        za,ta=h.step(x);zb,_=ref.step(x)
        assert torch.equal(za,zb)
        for k in ('parameters','adam','native_rng'):assert equal(h.snapshot()[k],ref.snapshot()[k])
        assert ta['diagnostics']['history_images']==i and len(h.bank)==i+1
    h.check_frozen(True);h.close();ref.close()
    print('PASS: baseline output/Adam/RNG parity; past-only graph; cold-start abstention; cut direction; collapse guard')


if __name__=='__main__':main()
