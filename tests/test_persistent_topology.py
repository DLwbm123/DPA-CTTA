import unittest
import numpy as np
import torch
from dpa_ctta.srg_ctta.persistent_topology import critical_pairs,correction
class TestPersistentTopology(unittest.TestCase):
 def test_fortran_indices_and_lifetimes(self):
  p=np.array([[.91,.22,.73],[.14,.85,.36]])
  import gudhi
  c=gudhi.CubicalComplex(top_dimensional_cells=1-p);diagram=c.persistence(min_persistence=-1)
  value=1-p.max()+sum(b-a for d,(a,b) in diagram if np.isfinite(b))
  self.assertAlmostEqual(float(correction(torch.tensor(p))),value,places=12)
 def test_diagonal_foreground_is_connected(self):
  p=np.array([[.9,.1],[.1,.8]])
  pairs,_=critical_pairs(p)
  self.assertFalse(any(p.ravel()[a]>.5 and p.ravel()[b]<.5 for a,b in pairs[0]))
 def test_hole_and_autograd_finite_difference(self):
  p=torch.tensor([[.91,.82,.93],[.84,.15,.86],[.97,.88,.99]],dtype=torch.float64,requires_grad=True)
  loss=correction(p);loss.backward();g=p.grad.numpy();q=p.detach().numpy();eps=1e-7
  for y,x in np.ndindex(q.shape):
   a=q.copy();b=q.copy();a[y,x]+=eps;b[y,x]-=eps
   numerical=(float(correction(torch.tensor(a)))-float(correction(torch.tensor(b))))/(2*eps)
   self.assertAlmostEqual(g[y,x],numerical,places=6)
  self.assertLess(g[1,1],0) # gradient descent fills the low-probability hole
 def test_ties_retained_and_repeatable_not_certified(self):
  p=np.full((5,4),.5);a=critical_pairs(p);b=critical_pairs(p)
  self.assertTrue(len(a[0][0])>0 and len(a[0][1])>0)
  for x,y in zip(a[0]+[a[1]],b[0]+[b[1]]):np.testing.assert_array_equal(x,y)
  self.assertAlmostEqual(float(correction(torch.tensor(p))),.5)
 def test_cached_indices_recompute_values(self):
  p=torch.tensor([[.9,.1,.8]],dtype=torch.float64);pairs=critical_pairs(p.numpy())
  a=correction(p,pairs);q=p.clone();q[0,2]=.7
  self.assertNotEqual(float(a),float(correction(q,pairs)))
 def test_invalid(self):
  with self.assertRaises(ValueError):critical_pairs(np.array([[np.nan]]))
if __name__=='__main__':unittest.main()
