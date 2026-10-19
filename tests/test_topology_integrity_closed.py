import unittest
import numpy as np
import torch
from dpa_ctta.srg_ctta.topology_integrity_closed import skeleton,weighted_support,integrity,objective,pairs_for_heads
class TestClosedTopologyIntegrity(unittest.TestCase):
 def example(self):
  p=torch.full((2,9,9),.1,dtype=torch.float64);p[:,2:7,2:7]=.9;p[:,4,4]=.2;return p
 def test_skeleton_limit_determinism(self):
  m=np.zeros((11,9),bool);m[1:10,1:8]=True
  a=skeleton(m);self.assertTrue(a.any());self.assertTrue(np.all(~a|m));np.testing.assert_array_equal(a,skeleton(m))
 def test_integrity_gradient_and_identical_is_not_zero(self):
  p=self.example();q=p.clone().requires_grad_();v,_,_=integrity(q,p,weighted_support(p));v.backward()
  self.assertGreater(float(v),0);self.assertTrue(torch.isfinite(q.grad).all());self.assertGreater(float(q.grad.abs().sum()),0)
 def test_finite_difference_fixed_binary_support(self):
  p=self.example();q=(p*.97+.01).requires_grad_();v,_,_=integrity(q,p,weighted_support(p));v.backward();eps=1e-6
  a=q.detach().clone();b=a.clone();a[0,3,3]+=eps;b[0,3,3]-=eps
  n=(integrity(a,p,weighted_support(p))[0]-integrity(b,p,weighted_support(p))[0])/(2*eps)
  self.assertAlmostEqual(float(q.grad[0,3,3]),float(n),places=5)
 def test_empty_extension_is_finite_and_no_blank_reward(self):
  p=torch.full((2,3,3),.2,dtype=torch.float64,requires_grad=True);s=weighted_support(p.detach())
  self.assertEqual(float(s.sum()),0)
  v,_,undefined=integrity(p,p.detach(),s);self.assertEqual(undefined,2);self.assertEqual(float(v),1000.)
  v.backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertEqual(float(p.grad.abs().sum()),0.)
 def test_one_empty_head_keeps_mse_gradient(self):
  previous=self.example();current=torch.full_like(previous,.2,requires_grad=True)
  v,s,undefined=integrity(current,previous,weighted_support(previous));self.assertEqual(undefined,2)
  v.backward();self.assertTrue(torch.isfinite(current.grad).all());self.assertGreater(float(current.grad.abs().sum()),0)
 def test_nonempty_formula_matches_frozen_p3(self):
  from dpa_ctta.srg_ctta.topology_integrity import integrity as frozen
  p=self.example();q=(p*.97+.01).requires_grad_();support=weighted_support(p)
  old,_=frozen(q,p,support);new,_,u=integrity(q,p,support)
  self.assertEqual(u,0);torch.testing.assert_close(old,new,rtol=1e-12,atol=1e-12)
  go=torch.autograd.grad(old,q,retain_graph=True)[0];gn=torch.autograd.grad(new,q)[0];torch.testing.assert_close(go,gn,rtol=1e-12,atol=1e-12)
 def test_permutation_preserves_critical_gradient_norm(self):
  p=self.example();q=p.clone().requires_grad_();z=q.clone().detach().requires_grad_()
  # Correction only, same fixed pairs permuted bijectively.
  from dpa_ctta.srg_ctta.persistent_topology import correction
  a=sum(correction(x,y) for x,y in zip(q,pairs_for_heads(p)));a.backward()
  b=sum(correction(x,y) for x,y in zip(z,pairs_for_heads(p,True)));b.backward()
  self.assertAlmostEqual(float(q.grad.norm()),float(z.grad.norm()),places=12)
 def test_curriculum_and_detached_previous(self):
  p=self.example();prev=p.clone().requires_grad_();p=p.clone().requires_grad_()
  a,c,s,d=objective(p,prev.detach(),weighted_support(prev.detach()),30,'TIB',None)
  self.assertEqual(d['integrity_weight'],1);self.assertTrue(d['ph_refresh']);a.backward();self.assertIsNone(prev.grad)
  _,cc,_,d=objective(p.detach(),prev.detach(),s.detach(),31,'TIB',c)
  self.assertIs(cc,c);self.assertEqual(d['integrity_weight'],.1);self.assertFalse(d['ph_refresh'])
 def test_reset_optimizer_and_full_model_repeat(self):
  from dpa_ctta.srg_ctta.topology_integrity_closed import refine
  class Tiny(torch.nn.Module):
   def __init__(self):
    super().__init__();self.layer=torch.nn.Conv2d(1,2,1)
    with torch.no_grad():self.layer.weight.fill_(.2);self.layer.bias.fill_(1.)
   def forward(self,x):return (self.layer(x),)
  m=Tiny().double();initial={k:v.clone() for k,v in m.state_dict().items()};x=torch.ones((1,1,9,9),dtype=torch.float64)
  events=[]
  z,t,o=refine(m,x,'TIB',steps=2,progress=lambda i,d:events.append((i,d['loss'])))
  self.assertEqual(len(events),2);last={k:v.clone() for k,v in m.state_dict().items()}
  m.load_state_dict(initial);zz,tt,oo=refine(m,x,'TIB',steps=2)
  self.assertTrue(torch.equal(z,zz));self.assertEqual(t,tt)
  for k,v in last.items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
  for k,v in o.items():
   for a,b in v.items():self.assertTrue(torch.equal(b,oo[k][a]) if torch.is_tensor(b) else b==oo[k][a])
 def test_frozen_config_and_static_edges(self):
  from pathlib import Path
  import json
  from dpa_ctta.srg_ctta.topology_closed_run import validate,static_outputs
  c=json.loads((Path(__file__).parents[1]/'configs/topology_integrity_closed_p4.json').read_text());validate(c)
  c['steps']=1
  with self.assertRaises(AssertionError):validate(c)
  z=torch.zeros((1,2,9,9),dtype=torch.float64);z[:,:,2:7,2:7]=1;z[:,:,4,4]=-1
  a=static_outputs(z);self.assertEqual(len(a),5);self.assertTrue(a['FILL_SOURCE'][:,4,4].all())
 def test_original_p3_empty_stop_is_preserved(self):
  from dpa_ctta.srg_ctta.topology_integrity import weighted_support as original
  with self.assertRaises(ValueError):original(torch.full((2,3,3),.2))
if __name__=='__main__':unittest.main()
