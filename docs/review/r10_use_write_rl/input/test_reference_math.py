import unittest
import torch
from reference_math import Memory,policy_observation,use_and_write,normal_log_prob,reference_kl,retention,group_advantage,clipped_surrogate

class MathChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p=torch.randn(64,dtype=torch.float64)
        self.d=torch.randn(32,dtype=torch.float64)
        self.s=torch.ones(64,dtype=torch.float64)
        self.a=torch.zeros(10,dtype=torch.float64)
        self.m=Memory.zero()

    def test_current_use_unchanged_by_write_intervention(self):
        u0,m0=use_and_write(self.p,self.d,self.m,self.s,self.a,forced_write=0.)
        u1,m1=use_and_write(self.p,self.d,self.m,self.s,self.a,forced_write=1.)
        self.assertTrue(torch.equal(u0,u1)); self.assertTrue(torch.equal(m0.m,self.m.m))
        self.assertTrue(torch.equal(m0.q,self.m.q)); self.assertTrue(torch.equal(m0.h,self.m.h))
        self.assertTrue(torch.equal(m1.m,u1)); self.assertTrue(torch.equal(m1.q,self.d))
        self.assertEqual(m1.h.item(),1.)

    def test_mass_difference_has_no_fake_cold_history(self):
        obs=policy_observation(self.d,self.p,self.m,self.s)
        self.assertEqual(obs.shape,(193,));self.assertTrue(torch.equal(obs[160:192],torch.zeros(32,dtype=torch.float64)))
        _,m=use_and_write(self.p,self.d,self.m,self.s,self.a,forced_write=.5)
        self.assertTrue(torch.equal(m.h*self.d-m.q,torch.zeros_like(self.d)))

    def test_identity_override_matches_full_state_write(self):
        u,m=use_and_write(self.p,self.d,self.m,self.s,self.a,forced_write=1.,identity_use=True)
        self.assertTrue(torch.equal(u,self.p));self.assertTrue(torch.equal(m.m,self.p))
        self.assertTrue(torch.equal(m.q,self.d))

    def test_immediate_write_gradient_zero_but_future_state_gradient_nonzero(self):
        a=self.a.clone().requires_grad_()
        u,m=use_and_write(self.p,self.d,self.m,self.s,a)
        g=torch.autograd.grad(u.square().sum(),a,retain_graph=True)[0]
        self.assertEqual(g[-1].item(),0.)
        later=torch.autograd.grad(m.m.square().sum(),a)[0]
        self.assertNotEqual(later[-1].item(),0.)

    def test_retention_changes_same_task_reward_credit(self):
        anchor=torch.ones(2,2,dtype=torch.float64)*.8
        candidate=anchor.expand(4,2,2).clone()
        candidate[1]-=.01;candidate[2]-=.02;candidate[3]-=.03
        r=.7+.05*retention(anchor,candidate)
        a,_=group_advantage(r)
        self.assertGreater(a[0],a[-1])
        self.assertAlmostEqual(float(a.mean()),0.,places=10)

    def test_channel_harm_is_not_hidden_by_other_channel_gain(self):
        anchor=torch.ones(2,2,dtype=torch.float64)*.8
        cand=anchor.clone();cand[:,0]+=.1;cand[:,1]-=.1
        self.assertLess(retention(anchor,cand).item(),1.)

    def test_equal_rewards_and_constant_reward_shift(self):
        equal,_=group_advantage(torch.ones(4))
        self.assertTrue(torch.equal(equal,torch.zeros_like(equal)))
        a,_=group_advantage(torch.tensor([.5,.6,.7,.8],dtype=torch.float64))
        b,_=group_advantage(torch.tensor([.5,.6,.7,.8],dtype=torch.float64)+.02)
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))

    def test_old_log_ratio_and_kl(self):
        raw=torch.randn(4,4,10,dtype=torch.float64)
        mu=torch.zeros_like(raw,requires_grad=True)
        old=normal_log_prob(raw,mu).detach()
        self.assertTrue(torch.equal((normal_log_prob(raw,mu)-old).exp(),torch.ones(4,4,dtype=torch.float64)))
        self.assertTrue(torch.equal(reference_kl(mu,mu.detach()),torch.zeros(4,4,dtype=torch.float64)))
        advantage=torch.tensor([-1.,-.5,.5,1.],dtype=torch.float64)[:,None]
        loss=clipped_surrogate(normal_log_prob(raw,mu),old,advantage)
        loss.backward();self.assertGreater(mu.grad.abs().sum().item(),0.)

    def test_myopic_log_prob_excludes_writer(self):
        raw=torch.randn(4,10,dtype=torch.float64)
        mu=torch.zeros_like(raw,requires_grad=True)
        normal_log_prob(raw,mu,active_dims=9).sum().backward()
        self.assertTrue(torch.equal(mu.grad[:,-1],torch.zeros(4,dtype=torch.float64)))

    def test_ema_resume_exact(self):
        rewards=[torch.tensor([.5,.51,.52,.53],dtype=torch.float64),
                 torch.tensor([.6,.61,.64,.65],dtype=torch.float64)]
        _,s=group_advantage(rewards[0],ema=.01)
        a,s2=group_advantage(rewards[1],ema=s)
        saved=float(s)
        b,s3=group_advantage(rewards[1],ema=saved)
        self.assertTrue(torch.equal(a,b));self.assertEqual(s2,s3)

if __name__=='__main__':unittest.main(verbosity=2)
