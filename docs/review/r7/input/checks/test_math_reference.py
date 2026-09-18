"""Actual small-matrix sanity tests; these are NOT complete R7 host tests."""
import json,os,platform,sys,time,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'references'))
import MATH_REFERENCE as r

class ReferenceTests(unittest.TestCase):
    def setUp(self):self.g=np.random.default_rng(20260918)
    def test_film_zero_identity(self):
        h=self.g.normal(size=(4,5));self.assertTrue(np.array_equal(h,r.film(h,np.zeros_like(h),np.zeros_like(h))))
    def test_film_zero_live_derivative(self):
        h=np.array([2.]);eps=1e-6
        derivative=(r.film(h,np.array([eps]),np.array([0.]))-r.film(h,np.array([-eps]),np.array([0.])))/(2*eps)
        np.testing.assert_allclose(derivative,.1*h,atol=1e-9)
    def test_stable_matrix_bound(self):
        for scale in (.01,1,100):self.assertLessEqual(np.linalg.norm(r.stable_matrix(self.g.normal(size=(8,8))*scale),2),.95+1e-12)
    def test_filter_against_independent_kalman_gain(self):
        x=self.g.normal(size=(5,5));p=x@x.T+.1*np.eye(5);f=r.stable_matrix(self.g.normal(size=(5,5)))
        m=self.g.normal(size=5);q=.02*np.eye(5);obs=self.g.normal(size=5);rv=np.diag(np.linspace(.1,.5,5))
        actual,cov=r.gaussian_filter(m,p,f,q,obs,rv);pm=f@m;pp=f@p@f.T+q
        k=np.linalg.solve((pp+rv).T,pp.T).T;expected=pm+k@(obs-pm)
        eye=np.eye(5);joseph=(eye-k)@pp@(eye-k).T+k@rv@k.T
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(cov,joseph,rtol=1e-10,atol=1e-12)
    def test_filter_full_transition_creates_correlations(self):
        f=np.array([[.8,.2],[0,.7]]);_,p=r.gaussian_filter(np.zeros(2),np.eye(2),f,.01*np.eye(2),np.ones(2),.1*np.eye(2))
        self.assertGreater(abs(p[0,1]),1e-8);self.assertGreater(np.linalg.eigvalsh(p).min(),0)
    def test_filter_high_noise_limit(self):
        m=np.array([.3,-.2]);f=.9*np.eye(2)
        actual,_=r.gaussian_filter(m,np.eye(2),f,.01*np.eye(2),np.array([2.,3.]),1e12*np.eye(2))
        np.testing.assert_allclose(actual,f@m,atol=1e-10)
    def test_filter_causality_and_reset(self):
        def seq(last):
            m=np.zeros(2);p=np.eye(2);out=[]
            for o in (np.ones(2),np.array([2.,-1.]),last):
                m,p=r.gaussian_filter(m,p,.9*np.eye(2),.1*np.eye(2),o,.2*np.eye(2));out.append(m.copy())
            return out
        a,b=seq(np.zeros(2)),seq(np.ones(2)*10)
        np.testing.assert_array_equal(a[:2],b[:2])
        q=lambda:r.gaussian_filter(np.zeros(2),np.eye(2),.9*np.eye(2),.1*np.eye(2),np.ones(2),.2*np.eye(2))[0]
        np.testing.assert_array_equal(q(),q())
    def test_filter_rejects_non_spd(self):
        with self.assertRaises(np.linalg.LinAlgError):r.gaussian_filter(np.zeros(2),np.eye(2),np.eye(2),np.eye(2),np.ones(2),-np.eye(2))
    def test_predictive_subspace_against_direct_covariance_truncation(self):
        jc=self.g.normal(size=(20,9));jp=self.g.normal(size=(12,9));b,cov=r.predictive_subspace(jc,jp,np.ones(20)*.2,3)
        prediction=jp@cov@jp.T;v,u=np.linalg.eigh(prediction);sel=np.argsort(v)[-3:]
        expected=(u[:,sel]*v[sel])@u[:,sel].T;actual=jp@b@b.T@jp.T
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-9)
    def test_predictive_subspace_full_rank(self):
        j=self.g.normal(size=(10,6));b,cov=r.predictive_subspace(j,j,np.ones(10),6)
        np.testing.assert_allclose(b@b.T,cov,rtol=1e-10,atol=1e-10)
    def test_soft_threshold_sign_and_zero(self):
        np.testing.assert_array_equal(r.soft_threshold([-2.,-.1,0,.1,2],.1),[-1.9,0,0,0,1.9])
    def test_ista_diagonal_closed_form(self):
        h=np.eye(4);o=np.array([1.,-.5,.001,0]);prior=np.array([.2,.1,0,0]);k=1.7;mu=.1;lam=.01
        z,d,energy=r.sparse_correct(h,o,prior,lam=lam,mu=mu,kappa=k)
        expected=r.soft_threshold((o-prior)/k**2,lam)/(1/k**2+mu)
        np.testing.assert_allclose(d,expected,rtol=1e-10,atol=1e-12);self.assertEqual(len(energy),6)
    def test_ista_monotone_five_steps(self):
        h=self.g.normal(size=(24,8));o=self.g.normal(size=24);p=self.g.normal(size=8)
        for k in (.25,1.,4.):
            _,_,v=r.sparse_correct(h,o,p,kappa=k);self.assertEqual(len(v),6);self.assertTrue(np.all(np.diff(v)<=1e-10))
    def test_predict_only_is_no_correction(self):
        p=self.g.normal(size=3);z,d,v=r.sparse_correct(np.eye(3),np.ones(3),p,steps=0)
        np.testing.assert_array_equal(z,p);np.testing.assert_array_equal(d,np.zeros(3));self.assertEqual(len(v),1)
    def test_sparse_zero_residual(self):
        h=self.g.normal(size=(8,3));p=self.g.normal(size=3);z,d,_=r.sparse_correct(h,h@p,p)
        np.testing.assert_allclose(z,p,atol=1e-14);np.testing.assert_allclose(d,0,atol=1e-14)
    def test_huber_definition(self):np.testing.assert_allclose(r.huber([-2,-.5,0,.5,2]),[1.5,.125,0,.125,1.5])
    def test_irls_matches_ridge_in_quadratic_regime(self):
        h=self.g.normal(size=(20,5))*.1;o=self.g.normal(size=20)*.01;p=np.zeros(5);v=np.ones(20);lam=.1
        z,energy,ws=r.robust_correct(h,o,v,p)
        expected=np.linalg.solve(h.T@h/20+lam*np.eye(5),h.T@o/20)
        np.testing.assert_allclose(z,expected,rtol=1e-10,atol=1e-12);self.assertEqual(len(ws),3)
    def test_irls_outlier_downweight_and_monotonicity(self):
        h=np.ones((10,1));o=np.zeros(10);o[-1]=100;v=np.ones(10);p=np.zeros(1)
        z,energy,ws=r.robust_correct(h,o,v,p);self.assertLess(ws[0][-1],ws[0][0]);self.assertTrue(np.all(np.diff(energy)<=1e-10));self.assertLess(z[0],1.)
    def test_irls_observation_replication_does_not_change_mean_objective(self):
        h=self.g.normal(size=(17,4));o=self.g.normal(size=17);v=np.exp(self.g.normal(size=17));p=self.g.normal(size=4)
        a=r.robust_correct(h,o,v,p)[0];b=r.robust_correct(np.tile(h,(2,1)),np.tile(o,2),np.tile(v,2),p)[0]
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
    def test_irls_no_observation_residual_keeps_prior(self):
        h=self.g.normal(size=(9,3));p=self.g.normal(size=3);z,_,_=r.robust_correct(h,h@p,np.ones(9),p)
        np.testing.assert_allclose(z,p,atol=1e-12)
    def test_nonfinite_and_invalid_variance_rejected(self):
        with self.assertRaises(ValueError):r.soft_threshold([np.nan],.1)
        with self.assertRaises(ValueError):r.robust_correct(np.ones((1,1)),np.ones(1),np.zeros(1),np.zeros(1))
        with self.assertRaises(ValueError):r.sparse_correct(np.eye(1),np.ones(1),np.zeros(1),kappa=0)
    def test_matrix_budgets(self):
        n=1951;self.assertEqual(n*3*(8+1+6*2),122913)
        self.assertEqual(n*3*3*2,35118);self.assertEqual(n*2*(8+1+4*2),66334)

if __name__=='__main__':
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={'scope':'independent NumPy small-matrix equations only; NOT R7 host/source-training/target/GPU verification',
            'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'python':platform.python_version(),'numpy':np.__version__,'wall_seconds':time.monotonic()-start,
            'model_forwards':0,'model_backwards':0,'optimizers':0,'GPU_calls':0,'real_asset_reads':0,'exit_code':0 if result.wasSuccessful() else 1}
    print(json.dumps(record,indent=2))
    if os.environ.get('R7_MATH_RESULT'):
        Path(os.environ['R7_MATH_RESULT']).write_text(json.dumps(record,indent=2)+'\n')
    sys.exit(record['exit_code'])
