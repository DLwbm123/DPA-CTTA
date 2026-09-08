import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from dpa_ctta.offline.adaptation_dd import native_adam, FiniteSqrt, OfflineEpisode, history_state
from dpa_ctta.offline.prompt_gradient_matching import cosine_objective
from dpa_ctta.m1_data import select_targets


class TinyPrompt(nn.Module):
    def __init__(self):
        super().__init__();self.data_prompt=nn.Parameter(torch.ones(1,3,1,1))
    def forward(self,x): return x*self.data_prompt,x.mean((2,3),keepdim=True).detach()
    def update(self,x):
        with torch.no_grad(): self.data_prompt.copy_(x)


class TinyBN(nn.BatchNorm2d):
    def __init__(self):
        super().__init__(3);self.sample_num=0;self.new_sample=False
    def forward(self,x):
        mean=x.mean((0,2,3),keepdim=True).detach()
        self.bn_loss=(x.mean((2,3),keepdim=True)-.5*mean).square().mean()
        return (x-.5*mean)/(1+x.var((0,2,3),keepdim=True).detach()).sqrt()


class M1Tests(unittest.TestCase):
    def test_adam_forward_and_moments_empty_existing_and_zero(self):
        for history in (0,4):
            p=nn.Parameter(torch.tensor([.2,-.7,1.],dtype=torch.float32))
            opt=torch.optim.Adam([p],lr=.05,betas=(.9,.99),eps=1e-8,weight_decay=0,foreach=False)
            for i in range(history): p.grad=torch.tensor([.1+i,.2,0.]);opt.step()
            g=torch.tensor([.3,-.4,0.],requires_grad=True)
            actual,state=native_adam(p,g,opt.state.get(p,{}),.05)
            p.grad=g.detach().clone();opt.step()
            torch.testing.assert_close(actual,p,rtol=1e-4,atol=1e-5)
            for key in ('exp_avg','exp_avg_sq'): torch.testing.assert_close(state[key],opt.state[p][key],rtol=1e-4,atol=1e-5)
            self.assertEqual(state['step'],int(opt.state[p]['step']))
            gradient,=torch.autograd.grad(actual.sum(),g)
            self.assertTrue(torch.isfinite(gradient).all())
        x=torch.tensor([0.,4.],requires_grad=True)
        y=FiniteSqrt.apply(x);self.assertTrue(torch.equal(y,torch.sqrt(x)))
        self.assertTrue(torch.equal(torch.autograd.grad(y.sum(),x)[0],torch.tensor([0.,.25])))

    def test_optimizer_primitive_double_reference(self):
        phi=torch.tensor([.8,-.3],dtype=torch.double)
        grad=torch.tensor([.3,.6],dtype=torch.double,requires_grad=True)
        state={'step':7,'exp_avg':torch.tensor([.2,.4],dtype=torch.double),'exp_avg_sq':torch.tensor([.5,.2],dtype=torch.double)}
        self.assertTrue(torch.autograd.gradcheck(lambda g:native_adam(phi,g,state,.01)[0],(grad,)))

    def test_D_O_have_image_gradient_and_history_mask_source_isolation(self):
        torch.manual_seed(31)
        h=SimpleNamespace(prompt=TinyPrompt(),model=nn.Sequential(TinyBN(),nn.Conv2d(3,1,1)),adabn=TinyBN,
            memory_bank=SimpleNamespace(memory={},get_size=lambda:0),neighbor=16)
        h.optimizer=torch.optim.Adam(h.prompt.parameters(),lr=.01,betas=(.9,.99));h.model.eval().requires_grad_(False)
        e=OfflineEpisode.__new__(OfflineEpisode);e.host=h;e.clone=copy.deepcopy(h.model);e.task='fundus';e.device=torch.device('cpu')
        e.counts={k:0 for k in ['live_forwards','proxy_forwards','proxy_images','prompt_forwards','gradient_calls','differentiable_inner']}
        state=history_state(h);original=copy.deepcopy(state);weights=copy.deepcopy(h.model.state_dict())
        pixels=torch.rand(1,3,8,8);mask=(torch.rand(1,1,8,8)>.5).float();masks=mask.expand(4,-1,-1,-1).clone()
        initial=torch.rand(4,3,8,8)
        for method in ('D','O'):
            S=nn.Parameter(initial.clone());opt=torch.optim.Adam([S],lr=.01)
            loss=e.objective(method,S,masks,pixels,mask,state)
            g,=torch.autograd.grad(loss,S);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
            S.grad=g;opt.step();self.assertFalse(torch.equal(S,initial))
            self.assertEqual(state['counters'],original['counters']);self.assertFalse(state['adam']['state'])
            self.assertTrue(torch.equal(state['prompt']['data_prompt'],original['prompt']['data_prompt']))
            for k,v in weights.items(): self.assertTrue(torch.equal(v,h.model.state_dict()[k]))
            self.assertTrue(all(p.grad is None for p in h.model.parameters()));self.assertIsNone(masks.grad)
            e.clear_graphs()

    def test_target_selection_conflict_alias_protected_and_unknown(self):
        def row(i,d='REFUGE',image=None,mask='b',**kw):
            return dict(sample_id=str(i),domain=d,image_sha256=image or str(i),mask_sha256=mask,split='combined',**kw)
        rows=[row(1),row(2,image='1'),row(3,image='conflict'),row(4,image='conflict',mask='c'),
              row(5,image='source'),row(6,d='ORIGA',image='1'),row(7,sealed_final=True)]
        selected,roles,excluded,counts=select_targets(rows,'fundus',{'source'})
        self.assertEqual([r['sample_id'] for r in selected],['1'])
        self.assertEqual(selected[0]['patient_linkage'],'UNKNOWN');self.assertEqual(selected[0]['original_split'],'combined')
        self.assertEqual(len(excluded),4);self.assertEqual(counts['ORIGA']['selected'],0)
        self.assertEqual(selected,select_targets(rows,'fundus',{'source'})[0])

    def test_gm_reference_is_stopped_and_zero_gradient_finite(self):
        s=torch.tensor([.3,-.5],requires_grad=True);r=torch.tensor([.4,.8],requires_grad=True)
        loss=cosine_objective(s,r);gs,gr=torch.autograd.grad(loss,(s,r),allow_unused=True)
        self.assertIsNone(gr);self.assertTrue(torch.isfinite(gs).all())
        z=torch.zeros(3,requires_grad=True);g,=torch.autograd.grad(cosine_objective(z,torch.zeros(3)),z)
        self.assertTrue(torch.isfinite(g).all())

    def test_task_aggregate_is_domain_equal_and_assd_adverse_tail(self):
        from dpa_ctta.m1_analysis import summarize, paired
        def row(domain,dice):
            return dict(domain=domain,metrics=[dict(channel='polyp',dice=dice,assd=2.,gt_empty=False,gt_full=False,pred_empty=False,pred_full=False)],pipeline_elapsed_seconds=1.,host_step_elapsed_seconds=.5,peak_allocated_bytes=8)
        rows=[row('domain1',.2),row('domain2',.8),row('domain2',.8)]
        result=summarize({a:copy.deepcopy(rows) for a in ['N','A','R','D','O']},'polyp')
        self.assertAlmostEqual(result['task_domain_macro_dice_percent']['N'],50.)
        self.assertEqual(result['task_comparisons_pp']['O-N'],0.)
        values=paired([dict(dice=.5,assd=9.),dict(dice=.5,assd=1.)],[dict(dice=.5,assd=2.),dict(dice=.5,assd=None)])
        self.assertEqual(values['assd_common_valid'],1)
        self.assertEqual(values['assd_adverse_upper_decile_mean_px'],7.)
