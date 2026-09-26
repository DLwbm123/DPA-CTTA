import copy
import unittest
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook
from torch import nn
from dpa_ctta.r9_current_first.protocol import graph,SPEC,CAPS,SPEC_SHA
from dpa_ctta.r9_current_first.training import SourceTrainer
from dpa_ctta.r9_current_first.selection import balanced,checkpoint,recipes
from dpa_ctta.r9_current_first.validation import dice
from dpa_ctta.r9_current_first.deployment import Segmenter,Host,prepare
from dpa_ctta.r9_current_first.gradient import GradientHost,STEPS,teacher_views
from dpa_ctta.r8_ba.methods import R8B,R8A,CurrentMLP,R8Segmenter
from dpa_ctta.r8_ba.trainer import SourceTrainer as LegacyTrainer
from dpa_ctta.r8_ba.context import capture,SOURCE_KEYS
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.schedule import CURRICULA
from dpa_ctta.r7_shared.source import SourceData,Record
from dpa_ctta.r7_shared.context import tensor_digest
from dpa_ctta.b1_host import GRATA_COMMIT


class Small(nn.Module):
    def __init__(self):
        super().__init__();self.res=nn.Module();self.res.conv1=nn.Conv2d(3,64,1)
        self.bn=nn.BatchNorm2d(64);self.up1=nn.Conv2d(64,256,1);self.up3=nn.Conv2d(256,256,1);self.head=nn.Conv2d(256,2,1)
    def forward(self,x):
        x=torch.nn.functional.adaptive_avg_pool2d(x,(8,8))
        x=self.up3(self.up1(self.bn(self.res.conv1(x))))
        return torch.nn.functional.interpolate(self.head(x),(512,512))


class API:
    class Rotate_and_Flip:
        def __call__(self,x,i):return torch.rot90(x,i%4,(-2,-1)).flip(-1) if i==4 else torch.rot90(x,i%4,(-2,-1))
        def inverse(self,x,i):return torch.rot90(x.flip(-1) if i==4 else x,-i%4,(-2,-1))
    @staticmethod
    def augmentation_strong_style(d):return d['data']*.8+.1
    @staticmethod
    def normalize_image_to_0_1(x):return (x-x.amin())/(x.amax()-x.amin()).clamp_min(1e-6)


class Tiny:
    amplitude=.3
    def __init__(self):self.calls=[]
    def __call__(self,x,v=None,observe=False):
        self.calls.append((x.clone(),observe))
        logits=x[:,:2]+(torch.zeros(1024) if v is None else v)[:2].view(1,2,1,1)
        return (logits,x.mean().expand(134).clone(),x.mean().expand(64,64).clone()) if observe else logits


def fit_fixture(kind='B'):
    torch.manual_seed(71)
    folds={k:[f'{k}{i}' for i in range(32)] for k in ('fit','cal','val')}
    data=SourceData([Record(g,k,torch.rand(1,3,8,8),torch.randint(2,(1,2,8,8)).float()) for k,gs in folds.items() for g in gs],folds)
    o=Oracles('fit',.3,torch.zeros(1024,512),(tuple(folds['fit'][:2]),)*512,(tuple(folds['fit'][2:4]),)*512,((),)*512)
    basis=torch.eye(1024,dtype=torch.float64)[:,:32 if kind=='A' else 64]
    m=R8A(basis,.3) if kind=='A' else R8B(basis,.3)
    m.observer.fit_scaler(torch.randn(8,134),'fit')
    return Tiny(),m,data,o


def host_fixture(arm=None,alpha=1.,diag=None):
    torch.manual_seed(17)
    s=Segmenter(Small(),.3,alpha=alpha)
    m=R8B(torch.eye(1024,dtype=torch.float64)[:,:64],.3)
    m.observer.fit_scaler(torch.randn(8,134),'fit');m.freeze();m=prepare(m,diag)
    c=dict(id='synthetic',rank=64,film_amplitude=.3,observer='global',aux_multiplier=1.,r9_spec_sha256=SPEC_SHA,output_alpha=alpha,diagnostic=diag)
    source={k:'0'*64 for k in SOURCE_KEYS};source['protocol_sha256']=PROTOCOL_SHA256
    if arm:
        scale=torch.ones(64,dtype=torch.float64);lr=1e-4 if arm=='BN_RESET_G1' else .001
        c.update(gradient_arm=arm,gradient_lr=lr,grata_commit=GRATA_COMMIT,scale_sha256=tensor_digest([('scale',scale)]))
        if arm=='BN_RESET_G1':m=None;c['bn_affine_names']=['bn.bias','bn.weight']
        return GradientHost(s,m,c,source,capture(s,m,c,source),arm,lr,scale,API())
    return Host(s,m,c,source,capture(s,m,c,source),diag)


class Core(unittest.TestCase):
    def test_graph_and_weighted_selection(self):
        g=graph();self.assertEqual(len(g['nodes']),43+4+771*2)
        rows=[dict(episode=i,curriculum=CURRICULA[i//16],visits=32,hard_Dice=.5+i//16*.1) for i in range(64)]
        self.assertAlmostEqual(balanced(rows),.575)
        self.assertEqual(checkpoint({s:rows for s in (4000,8000,12000,16000)})['step'],4000)
        with self.assertRaises(ValueError):balanced(rows[:-1])
        h,s=dice(torch.zeros(1,2,2,2),torch.zeros(1,2,2,2));self.assertEqual(h,[1.,1.])

    def test_legacy_exact_and_self_images_and_snapshot(self):
        torch.set_num_threads(2)
        a=fit_fixture();b=(Tiny(),copy.deepcopy(a[1]),a[2],a[3])
        old=LegacyTrainer(*a,20260924,'same');new=SourceTrainer(*b,20260924,'same',recipe='LEGACY')
        self.assertEqual(old.fit_step(),new.fit_step())
        for k,v in old.method.state_dict().items():self.assertTrue(torch.equal(v,new.method.state_dict()[k]))
        for kind in ('A','B'):
            args=fit_fixture(kind);worker=SourceTrainer(*args,20260924,'self',recipe='SELF')
            worker.fit_step();calls=args[0].calls
            stride=3 if kind=='A' else 2
            for i in range(0,len(calls),stride):self.assertTrue(torch.equal(calls[i+stride-2][0],calls[i+stride-1][0]))
            saved=worker.snapshot();trace=worker.fit_step();weights=copy.deepcopy(worker.method.state_dict())
            worker.restore(saved);self.assertEqual(trace,worker.fit_step())
            for k,v in weights.items():self.assertTrue(torch.equal(v,worker.method.state_dict()[k]))
            self.assertEqual(worker.state['counter'],16)
        args=fit_fixture('A');worker=SourceTrainer(*args,20260924,'task',recipe='SELF_TASK')
        worker.method.fit_loss=lambda *a:(_ for _ in ()).throw(AssertionError('aux called'))
        worker.fit_step();self.assertEqual(len(args[0].calls),16)

    def test_output_only_endpoints_and_ista_reset(self):
        image=torch.rand(1,3,512,512)
        h=host_fixture();r=host_fixture(alpha=.25)
        p,_=h.step(image);q,_=r.step(image)
        self.assertTrue(torch.equal(h.state['z'],r.state['z']));self.assertFalse(torch.equal(p,q))
        z=host_fixture(alpha=0);p0,_=z.step(image)
        self.assertTrue(torch.equal(p0,z.segmenter(image)))
        h=host_fixture(diag='RESET');p,_=h.step(image);q,_=h.step(image);self.assertTrue(torch.equal(p,q))
        h=host_fixture(diag='ISTA20');_,t=h.step(image);self.assertEqual(t['counts']['ISTA_iterations'],20)
        h=host_fixture(diag='PRECAL');h.method.assert_deployment_eta();h.step(image)

    def test_original_three_gradient_arms_numerical_equivalence(self):
        from dpa_ctta.r8_ba.gradient import GradientHost as OldGradient
        image=torch.rand(1,3,512,512)
        for arm in ('B_G1','B_G3','COLD_G3'):
            new=host_fixture(arm);base=host_fixture(arm)
            old=OldGradient(base.segmenter,base.method,base.context['payload']['config'],base.context['payload']['source'],base.context,arm,base.scale,base.lr,API())
            for _ in range(2):
                p,_=new.step(image);q,_=old.step(image)
                self.assertTrue(torch.equal(p,q))
                self.assertTrue(torch.equal(new.state['z'],old.state['z']))

    def test_bn_optimizer_fresh_each_image(self):
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        h=host_fixture('BN_RESET_G1');observed=[]
        hook=register_optimizer_step_pre_hook(lambda opt,*_:observed.append(len(opt.state)))
        try:
            for _ in range(2):h.step(torch.rand(1,3,512,512))
        finally:hook.remove()
        self.assertEqual(observed,[0,0])

    def test_six_gradient_arms_physical_hooks_and_replay(self):
        image=torch.rand(1,3,512,512)
        for arm,k in STEPS.items():
            h=host_fixture(arm);counts=dict(f=0,b=0,o=0)
            def forward(*a):counts['f']+=1
            handle=h.segmenter.model.register_forward_pre_hook(forward)
            original=torch.autograd.backward
            def backward(*a,**kw):counts['b']+=1;return original(*a,**kw)
            torch.autograd.backward=backward
            hook=register_optimizer_step_post_hook(lambda *a:counts.__setitem__('o',counts['o']+1))
            try:p,t=h.step(image)
            finally:torch.autograd.backward=original;hook.remove();handle.remove()
            self.assertEqual(counts,dict(f=7+k,b=k,o=k))
            snap=h.snapshot();q,_=h.step(image);h.restore(snap);replay,_=h.step(image)
            self.assertTrue(torch.equal(q,replay))
            if arm=='BN_RESET_G1':self.assertTrue(all(torch.equal(v,h.bn_source[n]) for n,v in h.bn.items()))
            if arm.startswith('COLD') or arm=='B_RESET_G3':self.assertEqual(float(h.state['z'].norm()),0.)

if __name__=='__main__':unittest.main()
