"""Independent formulas and common-state gradients; no registered inputs."""
import hashlib,math,unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from dpa_ctta.r6_regional_consistency.loss import weights,objective,scales,permutation_seed,TOLERANCES
from dpa_ctta.host_diagnostic import rng,close


class LossTests(unittest.TestCase):
    def test_partition_edges_double_mean_bounds_mass_and_fallback(self):
        n=262144
        for k in (0,1,73,n//2,n//3,n-1,n):
            q=torch.full((1,2,512,512),.25,dtype=torch.float64);q.view(2,-1)[:,:k]=.75
            w,h,rs=weights(q)
            self.assertTrue(torch.all(w>0));self.assertGreaterEqual(float(w.min()),.125);self.assertLessEqual(float(w.max()),8.)
            torch.testing.assert_close(w.mean((2,3)),torch.ones((1,2),dtype=torch.float64),rtol=0,atol=1e-14)
            for c,r in enumerate(rs):
                self.assertEqual(r['n_fg'],k)
                if k in (0,n):self.assertEqual(r['fallback'],'ONE_PARTITION_EMPTY');self.assertTrue(torch.equal(w,torch.ones_like(w)))
                elif not r['rho_clipped']:self.assertAlmostEqual(float(w[0,c][h[0,c]].sum()),n/2,places=8)
                else:self.assertGreater(abs(float(w[0,c][h[0,c]].sum())-n/2),1.)
    def test_permutation_seed_rng_histogram_and_soft_targets(self):
        q=torch.tensor([.8,.2,.3,.4,.1,.9,.4,.2],dtype=torch.float64).repeat(2).reshape(1,2,2,4);z=torch.zeros_like(q,requires_grad=True);before=q.clone();random=rng()
        _,rs=objective(z,q,'R_SHUFFLE',1);close(rng(),random,exact=True);self.assertTrue(torch.equal(before,q))
        _,again=objective(z,q,'R_SHUFFLE',1);self.assertEqual(rs,again)
        for c,r in enumerate(rs):
            raw=f'R6_WEIGHT_PERM_V1|20260907|1|{c}'.encode('ascii')
            self.assertEqual(r['seed'],int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**63))
            self.assertGreater(r['permutation_changed_positions'],0);self.assertTrue(r['permutation_histogram_preserved'])
        self.assertNotEqual(permutation_seed(1,0),permutation_seed(2,0))
    def test_local_gradients_detached_independent_reference_both_dtypes(self):
        for dtype in (torch.float64,torch.float32):
            q=torch.tensor([.83,.21,.31,.42,.14,.61,.38,.28],dtype=dtype).repeat(2).reshape(1,2,2,4)
            base=torch.linspace(-2.,2.,16,dtype=dtype).reshape_as(q);norms=[]
            h=q>=.5;n=h.sum((2,3),keepdim=True).to(dtype);ratio=((8-n)/n).clamp(.125,8.);bg=8/(ratio*n+8-n)
            w=torch.where(h,ratio*bg,bg).detach();d=base.sigmoid()-q
            reference=((w.double()*d.double())/q.numel()).square().sum((2,3)).sqrt()
            for arm in ('R_BAL','R_SCALE','R_SHUFFLE'):
                z=base.clone().requires_grad_();loss,rs=objective(z,q,arm,3);loss.backward();grad=z.grad
                norms.append(grad.double().square().sum((2,3)).sqrt())
                torch.testing.assert_close(norms[-1],reference,**TOLERANCES[dtype])
                if arm=='R_SCALE':
                    expected=d*torch.tensor([r['a'] for r in rs],dtype=dtype).reshape(1,2,1,1)/q.numel()
                    torch.testing.assert_close(grad,expected,**TOLERANCES[dtype])
    def test_soft_optimum_pos_weight_negative_control_and_zero_energy(self):
        z=torch.tensor([-.8,.8,-1.2,-2.],dtype=torch.float64).repeat(2).reshape(1,2,2,2).requires_grad_();q=z.detach().sigmoid()
        for arm in ('R_BAL','R_SCALE','R_SHUFFLE'):
            v=z.detach().clone().requires_grad_();loss,rows=objective(v,q,arm,1);loss.backward();self.assertEqual(float(v.grad.abs().sum()),0.)
            self.assertTrue(all(r['a']==r['b']==1. and r['S0']==0 for r in rows))
        bad=F.binary_cross_entropy_with_logits(z,q,pos_weight=torch.tensor(8.));bad.backward();self.assertGreater(float(z.grad.abs().sum()),0)
    def test_tiny_residuals_uniform_degeneration_and_hard_failures(self):
        q=torch.full((1,2,2,2),.5,dtype=torch.float64)
        for arm in ('C','R_BAL','R_SCALE','R_SHUFFLE'):
            z=torch.full_like(q,1e-12,requires_grad=True);v,_=objective(z,q,arm,1)
            torch.testing.assert_close(v,F.binary_cross_entropy_with_logits(z,q),rtol=0,atol=0)
        for values in ((0,1,0),(1,0,1),(1,1,0),(float('nan'),1,1),(1,float('inf'),1)):
            with self.assertRaises(ValueError):scales(*values)
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):objective(torch.full_like(q,value),q,'C',1)
            with self.assertRaises(ValueError):objective(torch.zeros_like(q),torch.full_like(q,value),'C',1)
        with patch('torch.autograd.grad',side_effect=AssertionError('VJP forbidden')):objective(torch.zeros_like(q),q,'R_SCALE',1)
