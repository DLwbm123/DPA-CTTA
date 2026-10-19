import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dpa_ctta.srg_ctta.masked_calibration import Host,masked_views,soft_target
from dpa_ctta.srg_ctta.masked_run import validate,calibration,JOBS


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm2d(3);self.head=nn.Conv2d(3,2,1)
    def forward(self,x):return self.head(self.bn(x))


class MaskedTests(unittest.TestCase):
    def test_masks_replay_exact_patch_quota(self):
        x=torch.ones(1,3,32,32)
        a=list(masked_views(x,37));b=list(masked_views(x,37))
        self.assertEqual(len(a),2)
        for (view,rho,effective),(other,r2,e2) in zip(a,b):
            self.assertTrue(torch.equal(view,other));self.assertEqual((rho,effective),(r2,e2))
            self.assertTrue(.2<=rho<.4);self.assertEqual(int((view[0,0]==0).sum()),int(64*rho)*16)
            self.assertTrue(torch.equal(view[:,0],view[:,1]))
        self.assertFalse(torch.equal(a[0][0],a[1][0]))

    def test_binary_softening_reverses_overconfident_gradient(self):
        hard=torch.ones(1,2,1,1)
        self.assertTrue(torch.equal(soft_target(hard,.2,'MARS'),torch.full_like(hard,.8)))
        gradients=[]
        for kind in ('MARS','MASK_HARD'):
            z=torch.full_like(hard,2.,requires_grad=True)
            F.binary_cross_entropy_with_logits(z,soft_target(hard,.2,kind)).backward()
            gradients.append(float(z.grad.mean()))
        self.assertGreater(gradients[0],0);self.assertLess(gradients[1],0)
        self.assertTrue(torch.equal(soft_target(hard,.2,'FIXED_RHO'),soft_target(hard,.4,'FIXED_RHO')))

    def test_real_updates_reset_and_frozen_state(self):
        torch.manual_seed(29);model=Model().eval();x=torch.rand(1,3,32,32)
        outputs=[]
        for kind in ('MARS','MASK_HARD','FIXED_RHO','ENT_NORM'):
            h=Host(model,kind);initial=copy.deepcopy(h.model.state_dict())
            pre,post,_,_,d=h.step(x,x,x,None,41)
            self.assertFalse(torch.equal(pre,post));self.assertAlmostEqual(d['steps'][0]['actual_step_norm'],.01,delta=5e-7)
            self.assertGreaterEqual(d['steps'][0]['direction_cosine'],.999999)
            self.assertEqual(d['optimizer_steps'],1);self.assertEqual(d['loss_backward_calls'],1 if kind=='ENT_NORM' else 3)
            affine=[p.clone() for p in h.params];h.reset_optimizer()
            self.assertTrue(all(torch.equal(p,v) for p,v in zip(h.params,affine)))
            self.assertFalse(h.opt.state);h.reset()
            self.assertTrue(all(torch.equal(v,h.model.state_dict()[n]) for n,v in initial.items()))
            _,repeat,_,_,_=h.step(x,x,x,None,41)
            self.assertTrue(torch.equal(post,repeat));outputs.append(post)
        self.assertFalse(torch.equal(outputs[0],outputs[1]))

    def test_frozen_matrix_and_q8_scope(self):
        root=Path(__file__).resolve().parents[1]
        c=json.loads((root/'configs/masked_calibration_ctta_v1.json').read_text());validate(c)
        self.assertEqual(len(JOBS),11);c['masked_views']=8
        with self.assertRaises(AssertionError):validate(c)
        q=np.full((3,3),255,np.uint8);p=np.ones((3,3),bool);roi=p.copy()
        exact=calibration(q,p,p,roi)
        self.assertEqual(exact['brier'],0);self.assertEqual(exact['ece'],0)
        wrong=calibration(q,p,~p,roi)
        self.assertEqual(wrong['brier'],1);self.assertEqual(wrong['ece'],1)
        self.assertIsNone(calibration(q,p,p,~roi))


if __name__=='__main__':unittest.main()
