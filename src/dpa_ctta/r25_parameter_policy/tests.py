"""Runnable checks for proposal rollback, policy credit, preferences and resume."""
import copy
import json
import torch
from .method import Host, Policy, reinforce, region_preference, reward, structure
from ..r24_c_context.method import Host as Previous
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model=Tiny()
    x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    def cfg(arm):return dict(id=arm,family='control' if arm=='C' else 'parameter_policy',host='C',params={},components={})
    def host(arm):return Host(None,cfg(arm),20260907,'test',device='cpu',model=copy.deepcopy(model))
    for arm in ('C','ANCHOR'):
        a=host(arm);b=Previous(None,cfg(arm),20260907,'test',device='cpu',model=copy.deepcopy(model))
        for _ in range(2):
            za,_=a.step(x);zb,_=b.step(x)
            assert torch.equal(za,zb),arm
            for key in ('parameters','adam','native_rng','adapter'):
                assert equal(a.snapshot()[key],b.snapshot()[key]),(arm,key)
        a.close();b.close()
    for arm in ('REPEAT5','RANDOM4','GREEDY','PG_CONS','PG_STRUCT','PG_RETAIN','PG_COV','RN_DPO'):
        a=host(arm)
        a.step(x)
        initial=a.snapshot()
        z1,t1=a.step(x.flip(-1));end=a.snapshot()
        a.restore(initial)
        z2,t2=a.step(x.flip(-1))
        assert torch.equal(z1,z2) and equal(end,a.snapshot()),arm
        assert t1['diagnostics']==t2['diagnostics'],arm
        assert a.visits==2 and a.native.steps==(10 if arm=='REPEAT5' else 2),arm
        if a.policy is not None:
            assert len(t1['diagnostics']['rewards'])==4 and len(t1['proposal_masks_b64'])==4
            assert len(a.memory)==1
            if arm.startswith('PG_'):
                assert t1['diagnostics']['policy_gradient_norm']>=0
                assert a.policy_optimizer.state
            else:assert not a.policy_optimizer.state
        a.check_frozen(True);a.close()
        print('PASS',arm,flush=True)
    policy=Policy(2,19)
    optimizer=torch.optim.SGD(policy.parameters(),lr=.01)
    context=torch.zeros(2);history=torch.zeros(5)
    actions=torch.tensor([[.4,0,0,0],[-.4,0,0,0],[.2,0,0,0],[-.2,0,0,0]])
    reinforce(policy,optimizer,context,history,actions,[1.,-1.,.5,-.5],1.)
    assert policy.distribution(context,history).loc[0]>0, 'reward must move probability toward preferred action'
    policy=Policy(2,20,True)
    d=policy.distribution(context,history)
    assert torch.linalg.eigvalsh(d.covariance_matrix).min()>0
    logits=torch.zeros(1,2,4,4,requires_grad=True);reference=torch.zeros_like(logits)
    win=torch.zeros_like(logits,dtype=torch.bool);lose=win.clone();win[...,1,1]=True
    loss=region_preference(logits,reference,win,lose);loss.backward()
    assert logits.grad[...,1,1].max()<0 and logits.grad[...,0,0].abs().max()==0
    padded=lambda a:torch.nn.functional.pad(a,(0,4,0,4))
    assert torch.equal(loss,region_preference(padded(logits),padded(reference),padded(win),padded(lose)))
    p=torch.zeros(1,2,64,64);p[:,0,16:48,16:48]=.9;p[:,1,24:40,24:40]=.9
    r=reward(p,p,p)
    shifted=p.roll(20,-1)
    assert structure(p,shifted)>structure(p,p)
    assert reward(torch.zeros_like(p),torch.zeros_like(p),p)['extent']>r['extent']
    assert reward(p,p,p,p,p)['retention']==0
    assert reward(p,p,p,shifted,p)['retention']>0
    print(json.dumps(dict(status='PASS',tests=12,checks=['exact C/ANCHOR parity','all arms exact resume',
        'one committed branch','separate policy optimizer','reward gradient sign','positive covariance',
        'region normalized preference','spatial OT evidence','collapse penalty','causal memory'],online_labels=False)))


if __name__=='__main__':main()
