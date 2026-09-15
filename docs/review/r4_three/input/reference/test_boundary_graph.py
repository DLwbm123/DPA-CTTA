"""Procedural tests only; no checkpoint, images, labels, or GPU access."""
import unittest
import torch
from boundary_graph import (reliability, contour_band, edge_weights,
                            anchored_solve, refine_target, degree)


def fixture(size=32):
    xy=torch.linspace(-1,1,size,dtype=torch.float32)
    yy,xx=torch.meshgrid(xy,xy,indexing='ij')
    rr=(xx.square()+yy.square()).sqrt()
    q=torch.stack([torch.sigmoid((.65-rr)*12),torch.sigmoid((.32-rr)*16)])[None]
    views=q[0].repeat(6,1,1,1)
    rgb=torch.stack([(xx+1)/2,(yy+1)/2,((xx*7).sin()*.3+.5)])[None]
    return rgb,q,views


class BoundaryGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_01_full_grid_edge_count(self):
        rgb,_,_=fixture(128)
        wh,wv=edge_weights(rgb[0],'G_BOUND',1)
        self.assertEqual(wh.numel()+wv.numel(),32512)
        self.assertLessEqual(float(degree(wh,wv).max()),4.)

    def test_02_no_artificial_contour_on_uniform_mask(self):
        for val in (False,True):
            self.assertFalse(contour_band(torch.full((2,9,9),val)).any())
        a=torch.zeros(2,9,9,dtype=torch.bool);a[:,:,4:]=True
        b=contour_band(a,0)
        self.assertEqual(int(b.sum()),2*9*2)
        self.assertTrue(b[:,:,3:5].all())
        self.assertFalse(b[:,:,:3].any())

    def test_03_anchors_need_confidence_and_six_view_agreement(self):
        q=torch.tensor([[[.05,.95,.5]]]);v=q[None].repeat(6,1,1,1)
        self.assertEqual(reliability(q,v).tolist(),[[[True,True,False]]])
        v[0,0,0,0]=.7
        self.assertFalse(reliability(q,v)[0,0,0])

    def test_04_control_weight_sums_multisets_and_rng(self):
        rgb,_,_=fixture(16)
        before=torch.random.get_rng_state().clone()
        base=edge_weights(rgb[0],'G_BOUND',3)
        cst=edge_weights(rgb[0],'G_CONST',3)
        sh=edge_weights(rgb[0],'G_SHUFFLE',3)
        sh2=edge_weights(rgb[0],'G_SHUFFLE',3)
        for a,b,c,d in zip(base,cst,sh,sh2):
            self.assertTrue(torch.allclose(a.sum(),b.sum(),atol=1e-10,rtol=1e-12))
            self.assertTrue(torch.equal(a.flatten().sort().values,c.flatten().sort().values))
            self.assertTrue(torch.equal(c,d))
            self.assertFalse(torch.equal(a,c))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_05_dense_system_equivalence_and_anchor_identity(self):
        gen=torch.Generator().manual_seed(27)
        q=torch.rand(2,4,5,generator=gen,dtype=torch.float64)
        free=torch.rand(2,4,5,generator=gen)>.3
        wh=torch.rand(4,4,generator=gen,dtype=torch.float64)
        wv=torch.rand(3,5,generator=gen,dtype=torch.float64)
        out,info=anchored_solve(q,free,wh,wv)
        lap=torch.zeros(20,20,dtype=torch.float64)
        for y in range(4):
            for x in range(5):
                u=y*5+x
                ns=[]
                if x+1<5:ns.append((u+1,wh[y,x]))
                if y+1<4:ns.append((u+5,wv[y,x]))
                for v,w in ns:
                    lap[u,u]+=w;lap[v,v]+=w;lap[u,v]-=w;lap[v,u]-=w
        for c in range(2):
            f=free[c].flatten();a=~f
            lhs=torch.eye(20,dtype=torch.float64)[f][:,f]+lap[f][:,f]
            rhs=q[c].flatten()[f]-lap[f][:,a]@q[c].flatten()[a]
            exact=torch.linalg.solve(lhs,rhs)
            self.assertTrue(torch.allclose(out[c].flatten()[f],exact,atol=1e-8,rtol=1e-8))
        self.assertTrue(info['exact_fixed_nodes'])
        self.assertLessEqual(info['energy_after'],info['energy_before']+1e-10)

    def test_06_probability_range_contraction_and_residual(self):
        gen=torch.Generator().manual_seed(28)
        q=torch.rand(2,20,20,generator=gen,dtype=torch.float64)
        wh=torch.ones(20,19,dtype=torch.float64);wv=torch.ones(19,20,dtype=torch.float64)
        out,info=anchored_solve(q,torch.ones_like(q,dtype=torch.bool),wh,wv)
        self.assertGreaterEqual(float(out.min()),float(q.min()))
        self.assertLessEqual(float(out.max()),float(q.max()))
        self.assertLessEqual(info['max_contraction'],.8)
        self.assertLess(info['max_free_linear_residual'],1e-5)

    def test_07_lambda_zero_and_empty_free_exact_identity(self):
        rgb,q,_=fixture(8);a,b=edge_weights(rgb[0],'G_BOUND',1)
        for free,lam in [(torch.ones_like(q[0],dtype=torch.bool),0.),(torch.zeros_like(q[0],dtype=torch.bool),1.)]:
            x,_=anchored_solve(q[0],free,a,b,lam)
            self.assertTrue(torch.equal(x,q[0].double()))

    def test_08_lifted_support_and_reliable_nodes_exact(self):
        rgb,q,v=fixture(32)
        target,audit,aux=refine_target(rgb,q,v,'G_BOUND',1,grid_size=8)
        self.assertTrue(torch.equal(target[~aux['allowed']],q[~aux['allowed']]))
        self.assertTrue(torch.equal(target[aux['reliable']],q[aux['reliable']]))
        self.assertGreater(int((target!=q).sum()),0)
        self.assertLessEqual(int((target!=q).sum()),int(aux['allowed'].sum()))
        self.assertTrue(audit['exact_outside_allowed'])

    def test_09_unchanged_inputs_and_detached_output(self):
        rgb,q,v=fixture(32)
        saved=[x.clone() for x in (rgb,q,v)]
        q.requires_grad_();v.requires_grad_()
        out,_,_=refine_target(rgb,q,v,'G_BOUND',1,grid_size=8)
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)
        for old,x in zip(saved,(rgb,q,v)):self.assertTrue(torch.equal(old,x))

    def test_10_global_free_superset_and_three_local_controls_share_masks(self):
        rgb,q,v=fixture(64)
        allaux={m:refine_target(rgb,q,v,m,1,grid_size=32,radius=0)[2] for m in ('G_BOUND','G_CONST','G_SHUFFLE','G_GLOBAL')}
        local=allaux['G_BOUND']['allowed']
        self.assertTrue(torch.equal(local,allaux['G_CONST']['allowed']))
        self.assertTrue(torch.equal(local,allaux['G_SHUFFLE']['allowed']))
        self.assertFalse((local & ~allaux['G_GLOBAL']['allowed']).any())

    def test_11_uniform_confident_q_returns_original(self):
        rgb,q,_=fixture(32)
        for val in (.02,.98):
            q=torch.full_like(q,val);v=q[0].repeat(6,1,1,1)
            out,info,_=refine_target(rgb,q,v,'G_BOUND',1,grid_size=8)
            self.assertTrue(torch.equal(out,q));self.assertEqual(info['free_nodes'],0)

    def test_12_uniform_ambiguous_q_is_well_defined(self):
        rgb,q,_=fixture(32);q=torch.full_like(q,.5);v=q[0].repeat(6,1,1,1)
        out,audit,_=refine_target(rgb,q,v,'G_GLOBAL',1,grid_size=8)
        self.assertTrue(torch.allclose(out,q,atol=1e-7,rtol=0))
        self.assertTrue(torch.isfinite(out).all())

    def test_13_rgb_contrast_reduces_cross_edge_weight(self):
        rgb=torch.zeros(3,8,8,dtype=torch.float64);rgb[:,:,4:]=1.
        wh,wv=edge_weights(rgb,'G_BOUND',1)
        self.assertTrue(torch.all(wh[:,3]<.001))
        self.assertTrue(torch.equal(wh[:,:3],torch.ones_like(wh[:,:3])))
        self.assertTrue(torch.equal(wv,torch.ones_like(wv)))

    def test_14_invalid_inputs_rejected(self):
        rgb,q,v=fixture(32)
        with self.assertRaises(ValueError):refine_target(rgb,q+.2,v,'G_BOUND',1,grid_size=8)
        with self.assertRaises(ValueError):refine_target(rgb,q,v,'wrong',1,grid_size=8)
        with self.assertRaises(ValueError):refine_target(rgb,q,v,'G_BOUND',1,grid_size=7)

    def test_15_full_production_shape_without_network(self):
        rgb,q,v=fixture(512)
        target,info,aux=refine_target(rgb,q,v,'G_BOUND',7)
        self.assertEqual(target.shape,(1,2,512,512))
        self.assertEqual(info['potential_undirected_edges'],32512)
        self.assertTrue(info['exact_reliable_full'])
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)
