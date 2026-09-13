"""Procedural CPU tests only; do not load a checkpoint or any real image."""
import unittest
import json
from pathlib import Path
import torch
from dpa_ctta.r3 import kernels as k

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]

class MathTests(unittest.TestCase):
    def setUp(self):
        self.g=torch.Generator().manual_seed(6109)
    def randn(self,*shape):
        return torch.randn(*shape,generator=self.g,dtype=torch.float64)
    def cov(self,d=32):
        x=self.randn(d,d)
        return x@x.T/d
    def test_01_covariance_models_psd(self):
        c0,c1=self.cov(),self.cov()
        for mode in ('LR','DIAG','ISO'):
            a,b=k.density_pair(c0,c1,mode)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(a).min()),1e-4-1e-10)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(b).min()),1e-4-1e-10)
            if mode=='ISO': self.assertTrue(torch.equal(a,b))
            if mode=='DIAG': self.assertTrue(torch.equal(a,torch.diag(a.diag())))
    def test_02_lowrank_tail_and_trace(self):
        c0,c1=self.cov(),self.cov()
        lr,_=k.density_pair(c0,c1,'LR')
        vals=torch.linalg.eigvalsh(lr)
        torch.testing.assert_close(vals[:24],vals[0].expand(24))
        torch.testing.assert_close(lr.trace(),k.shrink_covariance(c0).trace())
    def test_03_density_matches_explicit_inverse(self):
        x=self.randn(7,32);mean=self.randn(32);cov=k.shrink_covariance(self.cov())
        score,maha=k.log_density(x,mean,cov)
        explicit=((x-mean)@torch.linalg.inv(cov)*(x-mean)).sum(-1)
        torch.testing.assert_close(maha,explicit/32)
        torch.testing.assert_close(score,-.5*(explicit+torch.linalg.slogdet(cov)[1]))
    def test_04_density_teacher_zero_and_bounded(self):
        x=k.unit(self.randn(5,32));c=torch.eye(32,dtype=torch.float64)
        means=[torch.zeros(32,dtype=torch.float64),torch.zeros(32,dtype=torch.float64)]
        self.assertTrue(torch.equal(k.teacher_delta(x,[],[],ready=False),torch.zeros(5,dtype=x.dtype)))
        self.assertEqual(float(k.teacher_delta(x,means,[c,c],ready=True).abs().max()),0.)
        means[1]=self.randn(32)*100
        delta=k.teacher_delta(x,means,[c,c],ready=True)
        self.assertLessEqual(float(delta.abs().max()),2.)
    def test_05_density_unsupported_not_confident_fallback(self):
        x=torch.ones(2,32,dtype=torch.float64)*100
        mean=torch.zeros(32,dtype=torch.float64);cov=torch.eye(32,dtype=torch.float64)
        self.assertEqual(float(k.teacher_delta(x,[mean,mean+1],[cov,cov],ready=True).abs().max()),0.)
    def test_06_procrustes_recovers_known_rotation(self):
        x=self.randn(128,32);q,_=torch.linalg.qr(self.randn(32,32));y=x@q.T
        found=k.procrustes_step(x,y,torch.ones(128,dtype=torch.float64),tau=0.)
        torch.testing.assert_close(found,q,atol=1e-10,rtol=1e-10)
    def test_07_transport_matches_reencoded_statistics(self):
        x=self.randn(70,32);mu=x.mean(0);m2=(x-mu).T@(x-mu)
        Q,_=torch.linalg.qr(self.randn(32,32));U,_=torch.linalg.qr(self.randn(32,8))
        m,s,b=k.transport_state(mu,m2,U,Q);y=x@Q.T
        torch.testing.assert_close(m,y.mean(0))
        torch.testing.assert_close(s,(y-y.mean(0)).T@(y-y.mean(0)))
        torch.testing.assert_close(b.T@b,torch.eye(8,dtype=b.dtype))
    def test_08_procrustes_identity_and_no_input_mutation(self):
        x=k.unit(self.randn(64,32));copy=x.clone()
        Q=k.procrustes_step(x,x,torch.ones(64,dtype=x.dtype))
        torch.testing.assert_close(Q,torch.eye(32,dtype=x.dtype),atol=1e-10,rtol=1e-10)
        self.assertTrue(torch.equal(x,copy))
    def test_09_projection_matches_full_solve(self):
        A=self.randn(8,23);delta=self.randn(23);out,_=k.constrain_displacement(delta,A)
        B=A/A.norm(dim=1,keepdim=True)
        target=torch.linalg.solve(torch.eye(23,dtype=A.dtype)+B.T@B,delta)
        torch.testing.assert_close(out,target,atol=1e-10,rtol=1e-10)
    def test_10_projection_nullspace_and_zero_rows(self):
        A=torch.tensor([[1.,0.,0.],[0.,0.,0.]],dtype=torch.float64)
        d=torch.tensor([2.,3.,4.],dtype=torch.float64)
        out,info=k.constrain_displacement(d,A)
        torch.testing.assert_close(out,torch.tensor([1.,3.,4.],dtype=torch.float64))
        self.assertEqual(info['rows'],1)
        zero,_=k.constrain_displacement(d,A*0)
        self.assertTrue(torch.equal(zero,d))
    def test_11_norm_control_and_probe_contraction(self):
        for _ in range(6):
            A=self.randn(8,51);d=self.randn(51);out,info=k.constrain_displacement(d,A)
            control=k.norm_matched_displacement(d,out)
            torch.testing.assert_close(out.norm(),control.norm())
            self.assertLessEqual(info['new_norm'],info['base_norm']+1e-10)
            self.assertLessEqual(info['probe_after'],info['probe_before']+1e-10)
            torch.testing.assert_close(control/control.norm(),d/d.norm())
    def test_12_feature_jacobian_actual_Adam_step(self):
        # A tiny differentiable model illustrates independent Jacobian VJPs and
        # one Adam proposal, followed by an actual-delta transform.
        p=torch.nn.Parameter(self.randn(7));W=self.randn(5,7)
        features=torch.tanh(W@p)
        rows=torch.stack([torch.autograd.grad(features[j],p,retain_graph=True)[0] for j in range(2)])
        optimizer=torch.optim.Adam([p],lr=1e-4)
        loss=features.square().mean();optimizer.zero_grad();loss.backward()
        before=p.detach().clone();optimizer.step();proposal=p.detach()-before
        out,_=k.constrain_displacement(proposal,rows)
        with torch.no_grad():p.copy_(before+out)
        self.assertEqual(int(optimizer.state[p]['step']),1)
        torch.testing.assert_close(p.detach()-before,out,atol=1e-15,rtol=1e-10)
    def test_13_nested_projection_formula(self):
        q=torch.tensor([[.2,.8],[.8,.2]],dtype=torch.float64);a=torch.ones_like(q)
        p=k.nested_bernoulli_projection(q,a)
        torch.testing.assert_close(p,torch.tensor([[.5,.5],[.8,.2]],dtype=q.dtype))
        self.assertTrue((p[:,1]<=p[:,0]).all())
    def test_14_weighted_nested_projection_optimum(self):
        q=torch.tensor([[.2,.8]],dtype=torch.float64);a=torch.tensor([[10.,1.]],dtype=q.dtype)
        p=k.nested_bernoulli_projection(q,a)
        gradient=(a*(torch.logit(p)-torch.logit(q))).sum()
        self.assertAlmostEqual(float(gradient),0.,places=10)
    def test_15_grid_edges_local_unique(self):
        e=k.grid_edges();self.assertEqual(e.shape,(1984,2))
        self.assertEqual(len(set(map(tuple,e.tolist()))),1984)
        self.assertTrue(((e[:,1]-e[:,0]==1)|(e[:,1]-e[:,0]==32)).all())
    def test_16_graph_isotropic_identity_metrics(self):
        z=k.unit(self.randn(16,32));q=torch.rand((16,2),generator=self.g,dtype=z.dtype);e=k.grid_edges(4,4)
        a=k.graph_weights(z,q,e,None)
        b=k.graph_weights(z,q,e,[torch.eye(32,dtype=z.dtype)]*4)
        torch.testing.assert_close(a,b)
    def test_17_graph_mirror_proximal_energy_and_order(self):
        q=torch.rand((25,2),generator=self.g,dtype=torch.float64)*.8+.1
        e=k.grid_edges(5,5);w=torch.ones(len(e),dtype=q.dtype);rel=q>.7
        a=1+9*rel.to(q.dtype);initial=k.nested_bernoulli_projection(q,a)
        out=k.graph_refine(q,rel,e,w)
        self.assertTrue((out[:,1]<=out[:,0]).all())
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(float(k.graph_energy(out,q,a,e,w)),float(k.graph_energy(initial,q,a,e,w))+1e-10)
    def test_18_graph_zero_edges_equals_order_control(self):
        q=torch.rand((16,2),generator=self.g,dtype=torch.float64)*.8+.1;e=k.grid_edges(4,4);rel=q>.75
        out=k.graph_refine(q,rel,e,torch.zeros(len(e),dtype=q.dtype))
        torch.testing.assert_close(out,k.nested_bernoulli_projection(q,1+9*rel.to(q.dtype)),atol=1e-10,rtol=1e-10)
    def test_19_router_capacity_and_readonly_choice(self):
        router=k.DescriptorRouter();v=torch.eye(4,dtype=torch.float64)
        for j in range(4):
            for _ in range(16):
                size=len(router.entries);idx,new,d=router.choose(v[j]);self.assertEqual(len(router.entries),size)
                router.commit(v[j],idx,new,d)
        self.assertEqual(len(router.entries),3)
        self.assertEqual(sum(e.count for e in router.entries),64)
    def test_20_router_does_not_mutate_unselected(self):
        router=k.DescriptorRouter();v=torch.eye(3,dtype=torch.float64)
        for j in range(2):
            for _ in range(16):
                idx,new,d=router.choose(v[j]);router.commit(v[j],idx,new,d)
        old=router.entries[1].descriptor_sum.clone();count=router.entries[1].count
        idx,new,d=router.choose(v[0]);router.commit(v[0],idx,new,d)
        self.assertTrue(torch.equal(old,router.entries[1].descriptor_sum));self.assertEqual(count,router.entries[1].count)
    def test_21_recurrence_single_use_and_order_preserved(self):
        names=['A','B','C','D'];data={d:[f'{d}:{i}' for i in range(n)] for d,n in zip(names,[400,650,800,101])}
        seq=k.recurring_stream(data,names);self.assertEqual(len(seq),1951);self.assertEqual(len(set(seq)),1951)
        for name in names:self.assertEqual([i for i in seq if i.startswith(name+':')],data[name])
        self.assertNotEqual(seq,[x for n in names for x in data[n]])
    def test_22_recurrence_rejects_duplicate(self):
        with self.assertRaises(ValueError):k.recurring_stream({'A':['x'],'B':['x']},['A','B'])
    def test_23_full_matrix_budget(self):
        from dpa_ctta.r3.plan import matrix
        m=matrix();jobs=m['jobs']
        self.assertEqual(len(jobs),85);self.assertEqual(len(set(j['job_id'] for j in jobs)),85)
        self.assertEqual(sum(j['records'] for j in jobs),165835)
        self.assertEqual(sum(j['network_forwards'] for j in jobs),1385210)
        self.assertEqual(sum(j['jacobian_vjp_upper'] for j in jobs),234120)
        self.assertEqual(m['per_gpu_smoke']['adam_calls'],38)
        # Each method sees both devices over the four original orders.
        for arm in {j['arm'] for j in jobs}:
            self.assertEqual({(list(dict.fromkeys(j['arm'] for j in jobs)).index(arm)+j['order'])%2 for j in jobs if j['arm']==arm and j['order']<4},{0,1})
    def test_24_nonfinite_rejected_and_cuda_unused(self):
        with self.assertRaises(ValueError):k.shrink_covariance(torch.tensor([[float('nan')]]))
        with self.assertRaises(ValueError):k.constrain_displacement(torch.zeros(3),torch.zeros(2,4))
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)
