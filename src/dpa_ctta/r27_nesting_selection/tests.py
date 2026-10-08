"""Reward direction, unchanged random control, exact replay and frozen weights."""
import copy,json
import torch
from .method import Host,nesting_reward
from ..r26_residual_action.method import Host as Previous
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    p=torch.zeros(1,2,64,64);p[:,0]=.8;p[:,1]=.2
    good=nesting_reward(p,p,p)
    q=p.flip(1);bad=nesting_reward(q,q,p)
    assert good['retain']==0 and bad['retain']<0 and 'legacy_retain' in good
    model=Tiny();x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    for arm in ('ANCHOR','RANDOM4','GREEDY'):
        c=dict(id=arm,family=arm,host='C',params={},components={})
        a=Host(None,c,20260907,'test',device='cpu',model=copy.deepcopy(model))
        if arm!='GREEDY':
            b=Previous(None,c,20260907,'test',device='cpu',model=copy.deepcopy(model))
            za,_=a.step(x);zb,_=b.step(x)
            assert torch.equal(za,zb),arm
            for key in ('parameters','adam','native_rng','adapter'):assert equal(a.snapshot()[key],b.snapshot()[key])
            b.close()
        else:a.step(x)
        start=a.snapshot();z,t=a.step(x.flip(-1));end=a.snapshot();a.restore(start);z2,t2=a.step(x.flip(-1))
        assert torch.equal(z,z2) and equal(end,a.snapshot()) and t['diagnostics']==t2['diagnostics']
        if arm!='ANCHOR':
            d=t['diagnostics'];assert d['rewards']==[-v['nesting'] for v in d['reward_components']]
            if arm=='GREEDY':assert d['selected']==max(range(4),key=lambda i:d['rewards'][i])
            assert not a.policy_optimizer.state
        assert a.visits==2 and a.native.steps==2;a.check_frozen(True);a.close();print('PASS',arm,flush=True)
    print(json.dumps(dict(status='PASS',tests=5,online_labels=False)))


if __name__=='__main__':main()
