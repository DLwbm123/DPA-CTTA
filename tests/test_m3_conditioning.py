import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from dpa_ctta.m3_conditioning import condition_proxy,low_frequency_mask,ConditionedHost,ConditionedEpisode,make_host
from dpa_ctta.offline.adaptation_dd import OfflineEpisode,history_state
from test_m1 import TinyPrompt,TinyBN


class M3Tests(unittest.TestCase):
    def test_identity_symmetric_band_and_unchanged_outside_frequencies(self):
        torch.manual_seed(11);S=torch.rand(4,3,16,16);d=torch.rand(1,3,16,16);rng=torch.get_rng_state().clone()
        with patch('torch.fft.fft2',side_effect=AssertionError('identity must not FFT')):self.assertIs(condition_proxy(S,d,0),S)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(condition_proxy(S,S[:1])[:1],S[:1],rtol=1e-4,atol=1e-5)
        for size,radius in [(512,5),(352,3)]:
            mask=low_frequency_mask(size,size,.01,'cpu')[0,0];neg=(-torch.arange(size))%size
            self.assertTrue(torch.equal(mask,mask[neg][:,neg]));self.assertEqual(int(mask.sum()),(2*radius+1)**2)
        captured=[];ifft=torch.fft.ifft2
        def capture(x,**kw):captured.append(x);return ifft(x,**kw)
        with patch('torch.fft.ifft2',side_effect=capture):condition_proxy(S,d,band_fraction=.1)
        mask=low_frequency_mask(16,16,.1,'cpu').expand_as(S)
        self.assertTrue(torch.equal(captured[0][~mask],torch.fft.fft2(S)[~mask]))

    def test_donor_detach_zero_spectrum_finite_gradient_and_shape_validation(self):
        for zero in [False,True]:
            S=(torch.zeros(4,3,8,8) if zero else torch.rand(4,3,8,8)).requires_grad_();d=torch.rand(1,3,8,8,requires_grad=True)
            y,stats=condition_proxy(S,d,statistics=True);gs,gd=torch.autograd.grad(y.square().sum(),(S,d),allow_unused=True)
            self.assertIsNone(gd);self.assertTrue(torch.isfinite(gs).all());self.assertTrue(((y>=0)&(y<=1)).all());self.assertTrue(0<=stats['clipping_fraction']<=1)
        with self.assertRaises(ValueError):condition_proxy(S,torch.rand(2,3,8,8))
        with self.assertRaises(ValueError):condition_proxy(S,d,strength=float('nan'))

    def test_new_donor_nonaccumulation_masks_fixed_preprocessing_once_and_disabled(self):
        S=torch.rand(4,3,8,8);mask=torch.zeros(4,1,8,8);h=ConditionedHost.__new__(ConditionedHost)
        h.extra_weight=.1;h.strength=.5;h.condition_calls=0;h.proxy=SimpleNamespace(pixel_rgb=S.clone(),mask=mask);h.device=torch.device('cpu');h.task='polyp'
        donors=[torch.rand(1,3,8,8),torch.rand(1,3,8,8)];original=S.clone()
        from dpa_ctta.m3_conditioning import preprocess
        with patch('dpa_ctta.m3_conditioning.preprocess',wraps=preprocess) as pre,patch.object(ConditionedHost.__mro__[1],'step',side_effect=lambda x:x):
            for d in donors:
                self.assertIs(h.step(d),d);torch.testing.assert_close(h._proxy_input,preprocess(condition_proxy(original,d),'polyp'))
            self.assertEqual(pre.call_count,2)
        self.assertTrue(torch.equal(h.proxy.pixel_rgb,original));self.assertTrue(torch.equal(mask,torch.zeros_like(mask)))
        for weight,strength in [(0,.5),(.1,0)]:
            h.extra_weight=weight;h.strength=strength
            with patch('dpa_ctta.m3_conditioning.condition_proxy',side_effect=AssertionError('disabled T')),patch.object(ConditionedHost.__mro__[1],'step',return_value='native'):
                self.assertEqual(h.step(donors[0]),'native')

    def test_same_input_new_arms_identical_constructor_and_no_evaluator_argument(self):
        import inspect
        calls=[]
        with patch('dpa_ctta.m3_conditioning.ConditionedHost',side_effect=lambda *a,**k:calls.append((a,k))):
            for arm in ['R3','D3','O3','O2T']:make_host('fundus',arm,{},'same_proxy','cpu')
        for a,k in calls:
            self.assertEqual(a,('fundus',));self.assertEqual(k['proxy_factory'](),'same_proxy');self.assertEqual(k['extra_weight'],.1);self.assertEqual(k['strength'],.5);self.assertEqual(k['beta_boundary'],0)
        self.assertEqual(list(inspect.signature(ConditionedHost.step).parameters),['self','pixel_rgb'])

    def test_conditioned_meta_gradient_and_reused_functional_adam_against_actual(self):
        torch.manual_seed(31)
        h=SimpleNamespace(prompt=TinyPrompt(),model=nn.Sequential(TinyBN(),nn.Conv2d(3,1,1)),adabn=TinyBN,memory_bank=SimpleNamespace(memory={},get_size=lambda:0),neighbor=16)
        h.optimizer=torch.optim.Adam(h.prompt.parameters(),lr=.01,betas=(.9,.99));h.model.eval().requires_grad_(False)
        ep=ConditionedEpisode.__new__(ConditionedEpisode);ep.host=h;ep.clone=copy.deepcopy(h.model);ep.task='fundus';ep.device=torch.device('cpu');ep.inner_result=None
        ep.counts={k:0 for k in ['live_forwards','proxy_forwards','proxy_images','prompt_forwards','gradient_calls','differentiable_inner','condition_transforms']}
        state=history_state(h);initial=copy.deepcopy(state);pixels=torch.rand(1,3,8,8);mask=(torch.rand(1,1,8,8)>.5).float();masks=mask.expand(4,-1,-1,-1).clone()
        for method in ['D','O']:
            S=nn.Parameter(torch.rand(4,3,8,8));loss=ep.objective(method,S,masks,pixels,mask,state,capture_inner=True)
            g,=torch.autograd.grad(loss,S);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
            if method=='O':
                r=ep.inner_result;p=nn.Parameter(r['initial_prompt'].clone());opt=torch.optim.Adam([p],lr=.01,betas=(.9,.99));p.grad=r['gradient'];opt.step()
                torch.testing.assert_close(r['prompt'],p,rtol=1e-4,atol=1e-5)
                for k in ['exp_avg','exp_avg_sq']:torch.testing.assert_close(r['adam'][k],opt.state[p][k],rtol=1e-4,atol=1e-5)
                self.assertEqual(r['adam']['step'],int(opt.state[p]['step']))
            ep.clear_graphs()
        self.assertEqual(state['counters'],initial['counters']);self.assertFalse(state['adam']['state']);self.assertEqual(ep.counts['condition_transforms'],2)

    def test_zero_conditioning_matches_original_offline_math(self):
        # Reuse a tiny source host; only proxy input changes in M3.
        torch.manual_seed(17);h=SimpleNamespace(prompt=TinyPrompt(),model=nn.Sequential(TinyBN(),nn.Conv2d(3,1,1)),adabn=TinyBN,memory_bank=SimpleNamespace(memory={},get_size=lambda:0),neighbor=16)
        h.optimizer=torch.optim.Adam(h.prompt.parameters(),lr=.01,betas=(.9,.99));h.model.eval().requires_grad_(False)
        state=history_state(h);x=torch.rand(1,3,8,8);y=(torch.rand(1,1,8,8)>.5).float();S=nn.Parameter(torch.rand(4,3,8,8))
        ep=ConditionedEpisode.__new__(ConditionedEpisode);ep.host=h;ep.clone=copy.deepcopy(h.model);ep.task='fundus';ep.device=torch.device('cpu');ep.inner_result=None
        ep.counts={k:0 for k in ['live_forwards','proxy_forwards','proxy_images','prompt_forwards','gradient_calls','differentiable_inner','condition_transforms']}
        for method in ['D','O']:
            old=OfflineEpisode.objective(ep,method,S,y.expand(4,-1,-1,-1),x,y,state);go,=torch.autograd.grad(old,S);ep.clear_graphs()
            with patch('dpa_ctta.m3_conditioning.condition_proxy',side_effect=lambda s,*a,**kw:(s,dict(clipping_fraction=0.,change_l2=0.))):
                new=ep.objective(method,S,y.expand(4,-1,-1,-1),x,y,state);gn,=torch.autograd.grad(new,S)
            torch.testing.assert_close(old,new,rtol=0,atol=0);torch.testing.assert_close(go,gn,rtol=0,atol=0);ep.clear_graphs()

    def test_registration_reuses_exact_m2_lists_without_sampler(self):
        from dpa_ctta.m3_registration import load_registered
        from dpa_ctta.source_pilot_release import digest
        from test_m2_episodes import old_fixture
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m2=p/'m2';m3=p/'m3';m2.mkdir();m3.mkdir()
            (m2/'registration.json').write_text('{}');(m2/'receipt.run.json').write_text('{}')
            (m3/'registration.json').write_text(json.dumps(dict(m2_directory=str(m2),m2_registration_sha256=digest(m2/'registration.json'),m2_receipt_sha256=digest(m2/'receipt.run.json'),old_log_identities=[],o2_artifacts={})))
            lists={t:old_fixture(t) for t in ['fundus','polyp']};registered={'tasks':{t:{'episodes':e} for t,e in lists.items()}}
            with patch('dpa_ctta.m3_registration.load_m2',return_value=({},registered)),patch('dpa_ctta.m2_episodes.make_m2_episodes',side_effect=AssertionError('must not regenerate')):
                for method in ['D3','O3']:
                    loaded=load_registered(m3)[1]
                    for task in lists:self.assertIs(loaded['tasks'][task]['episodes'],lists[task])

    def test_four_new_arm_validation_and_nine_pair_report(self):
        from dpa_ctta.m3_analysis import validate_new_results,summarize,render_report
        from test_source_pilot_release import complete_fixture
        base,wanted=complete_fixture();arms={}
        for arm in ['R3','D3','O3','O2T']:
            rows=copy.deepcopy(base['B'])
            for r in rows:r.update(arm=arm,task='polyp',counts={'online_adam':1,'memory_pushes':1,'condition_transforms':1},condition_stats={'clipping_fraction':0.,'change_l2':1.},domain='fixture')
            arms[arm]=rows
        self.assertTrue(validate_new_results(arms,wanted,'polyp'))
        with self.assertRaises(ValueError):validate_new_results({k:v for k,v in arms.items() if k!='O2T'},wanted,'polyp')
        for arm in ['N','A','R','D1','O1','D2','O2']:arms[arm]=copy.deepcopy(arms['R3'])
        summary=summarize(arms,'polyp');self.assertEqual(len(summary['task_comparisons_pp']),9);self.assertEqual(len(summary['task_domain_macro_dice_percent']),11)
        with tempfile.TemporaryDirectory() as tmp:
            public=dict(status='M3_CONDITIONED_PROXY_COMPLETE',source={'polyp':summary},target={'polyp':summary},training={})
            audit=dict(experiment_execution_commit='FIXTURE_ONLY',updates_and_new_records={},reused_M1_M2_records=0,combined_display_records=0,gpu_seconds=0.,private_output_bytes=0,scoring_cost={})
            render_report(Path(tmp),public,audit);self.assertIn('O3-O2T',(Path(tmp)/'M3_EXPERIMENT_REPORT.md').read_text())
