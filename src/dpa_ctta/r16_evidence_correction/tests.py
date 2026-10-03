"""Qualification tests for identities, candidate conflicts, and stateful two-probe semantics."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from .methods import Engine,ResidualHead,residual,normalize_input,INPUT_CHANNELS,resize,prototypes
from .structure import Tracker,Candidate,hard,largest_filled,containment
from .zero_order import Adapter,directions,losses,update
from ..r15_decision_support.view import supported

class FakeSegmenter:
    def __init__(self):
        self.model=nn.Module();self.model.up3=nn.Identity();self.forwards=0
    def __call__(self,image):
        self.forwards+=1;f=self.model.up3(image)
        return f[:,:2].clone()

class Qualification(unittest.TestCase):
    def test_actual_label_open_denied(self):
        import json,tempfile
        from pathlib import Path
        from .run import forbid_target_labels
        with tempfile.TemporaryDirectory() as tmp:
            label=Path(tmp)/'synthetic-mask.bin';label.write_bytes(b'fake')
            reg=Path(tmp)/'registration.json';reg.write_text(json.dumps({'target':[{'mask_path':str(label)}]}))
            forbid_target_labels({'bindings':{'refs':{'target':{'path':str(reg)}}}})
            with self.assertRaises(PermissionError):label.read_bytes()

    def test_no_label_input_or_state_crossing(self):
        class Spatial:
            def __init__(self):self.model=nn.Module();self.model.up3=nn.Identity()
            def __call__(self,image):
                f=resize(image[:,:1],(128,128)).repeat(1,256,1,1);f=self.model.up3(f)
                return resize(f[:,:2],(512,512))
        e=Engine(Spatial());image=torch.full((1,3,512,512),.7)
        first,_,_=e.outputs(image);second,_,_=e.outputs(image)
        for k in first:self.assertTrue(torch.equal(first[k],second[k]))
        self.assertIsNone(e.feature);e.close()

    def test_support_and_protected_logits(self):
        z=torch.tensor([-1e20,-1e-7,0.,1e-7,1e20]).reshape(1,1,1,5).expand(1,2,1,5).clone()
        q=-z;b,m=supported(z,q)
        self.assertTrue(torch.equal(b.sigmoid()>=.5,q.sigmoid()>=.5))
        self.assertTrue(torch.equal(b[~m],z[~m]))
        mask=np.zeros((2,1,5),bool);r=torch.zeros(1,2,1,5)
        self.assertTrue(torch.equal(residual(b,mask,r),b))

    def test_head_capacity_zero_and_slots(self):
        torch.manual_seed(20261003);a=ResidualHead();b=copy.deepcopy(a)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        x=torch.randn(1,INPUT_CHANNELS,8,8);stat={'mean':torch.randn(1,INPUT_CHANNELS,1,1),'std':torch.ones(1,INPUT_CHANNELS,1,1)}
        self.assertFalse(a(x).any());self.assertFalse(normalize_input(x,stat,'D_LOGIT')[:,8:].any())
        z=torch.randn(1,2,16,16);m=np.ones((2,16,16),bool)
        self.assertTrue(torch.equal(residual(z,m,a(x)),z))

    def test_missing_prototypes_resize_and_flip(self):
        q,d=prototypes(torch.zeros(1,256,128,128),np.zeros((3,512,512),bool))
        self.assertIsNone(q);self.assertEqual(d['fallback'],'missing_seed')
        a=torch.arange(16).reshape(1,1,4,4).float()
        self.assertTrue(torch.equal(torch.flip(torch.flip(a,(-1,)),(-1,)),a))
        self.assertTrue(torch.equal(resize(a,(4,4),'nearest'),a))

    def test_empty_and_conflicting_candidates(self):
        z=torch.full((1,2,16,16),-1.);z[:,:,4:12,4:12]=1
        m=np.ones((2,16,16),bool);protected=np.zeros_like(m);q=torch.full_like(z,.8)
        tracker=Tracker(z,m,protected,q)
        out,d=tracker.select([]);self.assertTrue(torch.equal(out,z));self.assertEqual(d['accepted'],0)
        # A tiny cup addition outside OD must be rejected even if the simple reference permits it.
        c=Candidate(1,1,np.array([0]),torch.ones(1),{},None,[{}, {}, {}],.8)
        out,d=tracker.select([c],'SIMPLE',np.ones_like(m));self.assertEqual(d['rejections']['containment'],1)
        # Ties and overlapping candidates choose the first sorted accepted support once.
        c=Candidate(1,-1,np.array([68]),torch.tensor([-1.]),{},.6,[{}, {}, {}],.8)
        ref=hard(z);ref[1].ravel()[68]=False
        out,d=tracker.select([c,c],'SIMPLE',ref);self.assertEqual(d['accepted'],1);self.assertEqual(d['rejections']['conflict'],1)
        self.assertEqual(containment(hard(z)),0)
        self.assertTrue(np.array_equal(largest_filled(np.zeros_like(m)),np.zeros_like(m)))

    def test_zero_adapter_count_output_and_recovery(self):
        seg=FakeSegmenter();a=Adapter(seg,'Z_CORE',1e-3,1e-4)
        image=torch.linspace(-2,2,256*8*8).reshape(1,256,8,8)
        anchor=seg(image);self.assertTrue(torch.equal(anchor,seg(image)))
        snapshot=a.snapshot();first,trace=a.step(image);self.assertEqual(trace['full_forwards'],2);self.assertTrue(trace['output_before_update'])
        w=a.w.clone();a.restore(snapshot);second,_=a.step(image)
        self.assertTrue(torch.equal(first,second));self.assertTrue(torch.equal(w,a.w));a.close()

    def test_direction_and_shared_random_stream(self):
        w=torch.zeros(512);u=directions(20261003,0,1,w,'Z_CORE')
        self.assertTrue(torch.equal(u,directions(20261003,0,1,w,'Z_BOUND')))
        # Analytic directional finite difference of L(w)=<w,u>, exactly one dimension.
        u=torch.ones(1);w=torch.ones(1);new,_=update(w,u,1.001,.999,.001,.0001,'Z_CORE')
        self.assertLess(float(new),1.)
        zp=torch.tensor([[[[-2.,-1.],[1.,3.]],[[1.,-4.],[2.,-.1]]]])
        lp,lm=losses(1.2*zp,.8*zp)
        self.assertLess(abs(float(lp-lm)),1e-6)
        # A spatial perturbation survives channel-RMS normalization; pixel-sign normalization would erase it.
        changed=zp.clone();changed[0,0,0,0]+=.3
        lp,lm=losses(changed,zp);self.assertGreater(abs(float(lp-lm)),1e-7)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun!=8:raise RuntimeError('test discovery coverage mismatch')
    raise SystemExit(0 if result.wasSuccessful() else 1)
