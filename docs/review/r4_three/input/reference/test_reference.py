"""Only synthetic CPU mathematics and budget checks; no model/data/GPU."""
import copy, json, math, sys, unittest
from pathlib import Path
import torch
from kernel_geometry import KernelGeometry, cayley_from_upper, split_spatial_dc

torch.set_num_threads(2)

class MathChecks(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(71409)
        self.w = torch.randn(5, 4, 3, 3, generator=self.g, dtype=torch.float64)

    def change(self, m):
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(.08*torch.randn(p.shape, generator=self.g, dtype=p.dtype))

    def test_dc_decomposition(self):
        a,h,b = split_spatial_dc(self.w)
        torch.testing.assert_close(a[...,None,None]*b+h,self.w)
        torch.testing.assert_close((h*b).sum((-2,-1)),torch.zeros_like(a),atol=1e-14,rtol=0)
        self.assertAlmostEqual(b.square().sum().item(),1.)

    def test_all_modes_zero_initialization_exact_and_rng_unchanged(self):
        for dtype in (torch.float32,torch.float64):
            for mode in ('KDG','K_ALL','K_MAG','K_FREE'):
                state=torch.random.get_rng_state().clone()
                m=KernelGeometry(self.w.to(dtype),mode)
                self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
                self.assertTrue(torch.equal(m(),self.w.to(dtype)))

    def test_cayley_orthogonality_and_identity_derivative(self):
        u=torch.zeros(6,dtype=torch.float64,requires_grad=True)
        q=cayley_from_upper(u,4)
        self.assertTrue(torch.equal(q,torch.eye(4,dtype=torch.float64)))
        grad=torch.autograd.grad(q[0,1],u)[0]
        self.assertGreater(grad.abs().sum().item(),0.9)
        with torch.no_grad():u.copy_(torch.randn(6,generator=self.g,dtype=torch.float64))
        q=cayley_from_upper(u,4)
        torch.testing.assert_close(q.T@q,torch.eye(4,dtype=q.dtype),atol=1e-12,rtol=1e-12)

    def test_kdg_preserves_detail_and_dc_angles(self):
        m=KernelGeometry(self.w,'KDG');self.change(m);a=m.audit()
        self.assertLess(a['detail_change_l2'],1e-12)
        self.assertLess(a['dc_cosine_gram_max_error'],1e-12)
        self.assertGreater(a['effective_change_l2'],1e-4)

    def test_local_zero_mean_perturbation_response(self):
        m=KernelGeometry(self.w,'KDG');self.change(m)
        x=torch.randn(4,3,3,generator=self.g,dtype=torch.float64)
        x=x-x.mean((-2,-1),keepdim=True)
        difference=torch.einsum('oihw,ihw->o',m()-self.w,x)
        torch.testing.assert_close(difference,torch.zeros_like(difference),atol=2e-14,rtol=0)

    def test_kall_changes_detail_preserves_full_filter_angles(self):
        m=KernelGeometry(self.w,'K_ALL');self.change(m)
        self.assertGreater(m.audit()['detail_change_l2'],1e-3)
        a=torch.nn.functional.normalize(m().flatten(1),dim=1)
        b=torch.nn.functional.normalize(self.w.flatten(1),dim=1)
        torch.testing.assert_close(a@a.T,b@b.T,atol=1e-12,rtol=1e-12)

    def test_free_preserves_detail_not_angular_structure(self):
        m=KernelGeometry(self.w,'K_FREE');self.change(m);a=m.audit()
        self.assertLess(a['detail_change_l2'],1e-12)
        self.assertGreater(a['dc_cosine_gram_max_error'],1e-4)

    def test_gradients_finite_difference(self):
        for mode in ('KDG','K_ALL','K_MAG','K_FREE'):
            m=KernelGeometry(self.w,mode)
            probe=torch.randn(self.w.shape,generator=self.g,dtype=torch.float64)
            loss=(m()*probe).sum();loss.backward()
            for p in m.parameters():
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
                analytic=p.grad.flatten()[0].item();eps=1e-6
                with torch.no_grad():p.flatten()[0]+=eps
                plus=(m()*probe).sum().item()
                with torch.no_grad():p.flatten()[0]-=2*eps
                minus=(m()*probe).sum().item()
                with torch.no_grad():p.flatten()[0]+=eps
                self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),places=7)

    def test_original_weights_immutable_after_adam(self):
        for mode in ('KDG','K_ALL','K_MAG','K_FREE'):
            m=KernelGeometry(self.w,mode);saved=copy.deepcopy(m.state_dict())
            opt=torch.optim.Adam(m.parameters(),lr=1e-4)
            (m()*torch.randn(self.w.shape,generator=self.g,dtype=torch.float64)).sum().backward();opt.step()
            self.assertTrue(torch.equal(m.base_weight,saved['base_weight']))
            self.assertTrue(torch.equal(m.a0,saved['a0']))
            self.assertIsNone(m.base_weight.grad)

    def test_zero_dc_rows_finite_and_cannot_invent_signal(self):
        w=torch.arange(-4,5,dtype=torch.float64).reshape(1,1,3,3).expand(5,4,3,3).clone()
        for mode in ('KDG','K_MAG','K_FREE'):
            m=KernelGeometry(w,mode);self.change(m)
            self.assertTrue(torch.equal(m(),w));self.assertTrue(torch.isfinite(m()).all())
        m=KernelGeometry(w,'K_ALL');self.change(m)
        self.assertTrue(torch.isfinite(m()).all())
        self.assertGreater((m()-w).norm().item(),1e-5)

    def test_pinned_parameter_budgets(self):
        expected={'KDG':2147,'K_ALL':2147,'K_MAG':128,'K_FREE':4288}
        for mode,count in expected.items():
            ms=[KernelGeometry(torch.zeros(64,3,7,7),mode),KernelGeometry(torch.zeros(64,64,3,3),mode)]
            self.assertEqual(sum(p.numel() for m in ms for p in m.parameters()),count)

    def test_matrix_and_smoke_budgets(self):
        path=Path(__file__).resolve().parents[1]/'inherited'/'R4D_SCIENCE_PROPOSAL.json'
        c=json.loads(path.read_text());arms=c['arms'];n=1951;streams=5
        self.assertEqual(len(arms),10)
        self.assertEqual(n*streams*len(arms),97550)
        self.assertEqual(n*streams*sum(a['forwards_per_visit'] for a in arms),799910)
        self.assertEqual(2*sum(a['forwards_per_visit'] for a in arms)+4*8,196)
        self.assertEqual(2*len(arms)+4,24)
        self.assertEqual(c['formal_budget']['scoring_records'],97550)
        self.assertEqual(c['formal_budget']['network_forwards'],799910)

if __name__=='__main__':
    unittest.main(verbosity=2)
