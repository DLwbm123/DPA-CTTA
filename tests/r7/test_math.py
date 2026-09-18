import unittest
import numpy as np
import torch
from common import obs,method
from dpa_ctta.r7_shared.numerics import predictive_basis,stable,svd_basis,COUNTS
from dpa_ctta.r7_a_psf import gaussian_filter
from dpa_ctta.r7_b_rca import correct as sparse
from dpa_ctta.r7_c_rbe import correct as robust
import MATH_REFERENCE as ref

class MathTests(unittest.TestCase):
    def setUp(self):self.g=torch.Generator().manual_seed(924)
    def rand(self,*s):return torch.randn(*s,generator=self.g,dtype=torch.float64)
    def close(self,a,b):np.testing.assert_allclose(a.detach().numpy(),b,rtol=1e-8,atol=1e-10)
    def test_A_filter_independent_reference(self):
        x=self.rand(16,16);p=x@x.T+torch.eye(16);f=stable(self.rand(16,16));q=torch.eye(16,dtype=torch.float64)*.01;r=torch.diag(self.rand(16).exp());m=self.rand(16);o=self.rand(16)
        a,b=gaussian_filter(m,p,f,q,o,r);c,d=ref.gaussian_filter(*[v.numpy() for v in (m,p,f,q,o,r)]);self.close(a,c);self.close(b,d)
    def test_A_full_transition_correlations_and_high_noise(self):
        f=torch.eye(16,dtype=torch.float64)*.8;f[0,1]=.2;m=self.rand(16);eye=torch.eye(16,dtype=torch.float64)
        a,p=gaussian_filter(m,eye,f,eye*.01,self.rand(16),eye)
        self.assertGreater(abs(float(p[0,1])),1e-5);torch.linalg.cholesky(p)
        a,_=gaussian_filter(m,eye,f,eye*.01,self.rand(16),eye*1e12);self.close(a,(f@m).numpy())
    def test_A_predictive_basis_independent_SVD_covariance(self):
        jc,jp,w=self.rand(40,24),self.rand(35,24),self.rand(40).sigmoid()
        b,c=predictive_basis(jc,jp,w,8);br,cr=ref.predictive_subspace(jc.numpy(),jp.numpy(),w.numpy(),8)
        self.close(c,cr);self.close(b@b.T,br@br.T)
        e,u=torch.linalg.eigh(c);root=(u*e.sqrt())@u.T
        uu,ss,_=torch.linalg.svd(jp@root,full_matrices=False)
        self.close(jp@b@b.T@jp.T,((uu[:,:8]*ss[:8].square())@uu[:,:8].T).numpy())
    def test_rank_failures_no_random_padding(self):
        with self.assertRaises(ValueError):svd_basis(torch.zeros(1024,128),32)
        with self.assertRaises(ValueError):predictive_basis(torch.ones(8,5),torch.zeros(8,5),torch.ones(8),3)
    def test_B_five_step_reference_temperature_and_monotonicity(self):
        h,o,p=self.rand(64,32),self.rand(64),self.rand(32)
        for k in (.25,1.,3.):
            before=COUNTS['ISTA_iterations'];z,a=sparse(h,o,p,k);zr,_,ener=ref.sparse_correct(h.numpy(),o.numpy(),p.numpy(),kappa=k)
            self.close(z,zr);self.close(a['energy'],ener);self.assertTrue(torch.all(torch.diff(a['energy'])<=1e-10));self.assertEqual(COUNTS['ISTA_iterations']-before,5)
            self.close(a['eta'],1/(np.linalg.norm(h.numpy(),2)**2/k**2+.1))
    def test_B_diagonal_closed_form_and_gradient(self):
        h=torch.eye(4,dtype=torch.float64,requires_grad=True);o=torch.tensor([2.,-2.,.001,0.],dtype=torch.float64,requires_grad=True);p=torch.zeros(4,dtype=torch.float64)
        z,_=sparse(h,o,p);self.close(z,(o.detach().sign()*(o.detach().abs()-.01).clamp_min(0)/1.1).numpy())
        z.square().sum().backward();self.assertGreater(float(h.grad.norm()),0);self.assertGreater(float(o.grad.norm()),0)
    def test_C_IRLS_reference_outliers_and_fixed_steps(self):
        h,o,r,p=self.rand(512,32),self.rand(512)*4,self.rand(512).exp(),self.rand(32)
        before=COUNTS['IRLS_iterations'];z,a=robust(h,o,r,p);zr,ener,ws=ref.robust_correct(h.numpy(),o.numpy(),r.numpy(),p.numpy())
        self.close(z,zr);self.close(a['energy'],ener);self.assertEqual(COUNTS['IRLS_iterations']-before,3);self.assertTrue(torch.all(torch.diff(a['energy'])<=1e-10))
        for w,wr in zip(a['weights'],ws):self.close(w,wr)
    def test_C_ridge_limit_mean_scaling_and_zero_residual(self):
        h=self.rand(20,4)*.1;o=self.rand(20)*.01;r=torch.ones(20,dtype=torch.float64);p=torch.zeros(4,dtype=torch.float64)
        z,_=robust(h,o,r,p);expected=torch.linalg.solve(h.T@h/20+.1*torch.eye(4),h.T@o/20)
        self.close(z,expected.numpy());zz,_=robust(h.repeat(2,1),o.repeat(2),r.repeat(2),p);self.close(z,zz.numpy())
        z,a=robust(h,h@p,r,p);self.close(z,p.numpy());self.close(a['weights'][0],np.ones(20))
    def test_invalid_nonfinite_spd_variance(self):
        with self.assertRaises(ValueError):sparse(torch.ones(2,2),torch.ones(2),torch.zeros(2),0)
        with self.assertRaises(ValueError):robust(torch.ones(2,2),torch.ones(2),torch.zeros(2),torch.zeros(2))
        with self.assertRaises(ValueError):sparse(torch.full((2,2),float('nan')),torch.ones(2),torch.zeros(2))
    def test_A_tau_square_and_B_first_difference(self):
        raw,e=obs();a=method('A');r=a.observe(raw,e)['R'];a.set_stage('cal');a.cal_raw.data.fill_(0);self.close(a.observe(raw,e)['R'],(r*2.125**2).detach().numpy())
        b=method('B');_,aux=b.update(raw,e,b.initial());self.assertEqual(float(aux['difference'].norm()),0)
    def test_all_shapes_state_static_and_source_gradients(self):
        for g in 'ABC':
            m=method(g);raw,e=obs();s,a=m.update(raw,e,m.initial());s,a=m.update(*obs(1),s)
            loss=m.code(s).square().sum()+m.fit_loss(a,m.observe(raw,e),torch.zeros(m.rank,dtype=torch.float64),s);loss.backward()
            expected={'A':['W','qraw','style','content','head.0.weight'],'B':['W','G','Hraw','head.0.weight','bias.0.weight'],'C':['Pc','Pa','O','query.0.weight','mapping.0.weight']}[g]
            for name in expected:self.assertGreater(float(dict(m.named_parameters())[name].grad.norm()),0,name)
            stat=method(g,True);s1,_=stat.update(raw,e,stat.initial());s2,_=stat.update(raw,e,s1);self.close(stat.code(s1),stat.code(s2).detach().numpy())
    def test_C_calibration_only_variance_parameters(self):
        m=method('C');m.requires_grad_(False);m.set_stage('cal');s,a=m.update(*obs(),m.initial());m.cal_loss(a,torch.ones(32,dtype=torch.float64),s).backward()
        for n,p in m.named_parameters():self.assertEqual(p.grad is not None,n.startswith('reliability.'))
    def test_A_ambient_1024_basis_and_projection(self):
        from dpa_ctta.r7_shared.numerics import project
        jc,jp=self.rand(32,1024),self.rand(32,1024)
        b,cov=predictive_basis(jc,jp,torch.full((32,),.2),16)
        self.assertEqual(b.shape,(1024,16));self.assertEqual(int(torch.linalg.matrix_rank(b)),16)
        z=self.rand(16);got,a=project(b,b@z);self.close(got,z.numpy());self.assertLess(a['residual_l2'],1e-9)
    def test_B_dictionary_normalization_spectral_bounds(self):
        m=method('B');_,a=m.update(*obs(),m.initial())
        self.close(a['H'].norm(dim=0),np.ones(32))
        self.assertLessEqual(float(torch.linalg.matrix_norm(stable(m.W.double()),2)),.95+1e-10)
    def test_C_variance_bounds_constant_geometric_mean(self):
        from dpa_ctta.r7_shared.numerics import variance
        r=variance(torch.tensor([-100.,0,100.]));self.assertGreaterEqual(float(r.min()),1e-4-1e-10);self.assertLessEqual(float(r.max()),10.)
        m=method('C');a=m.observe(*obs());self.assertTrue(torch.equal(a['R'],torch.ones(64,8,dtype=torch.float64)))
        m.freeze();v=torch.stack([m.observe(*obs(i))['R'] for i in range(3)])
        gm=v.log().mean((0,1)).exp();m.set_constant_variance(gm,'cal')
        self.close(m.constant_R.double(),gm.float().double().numpy())
