import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.b5_host import Host,H0,make_host,check_policy,policy,SCORE_HEAD_BN,FIELDS
from dpa_ctta.b4_host import PolypC,logits
from dpa_ctta.b4_run import capture
from dpa_ctta.b5_analysis import recompute,validate_rows,COMPLETE
from dpa_ctta.b5_data import SMOKE
from dpa_ctta.integrations.ctta_suite import build_reference_model
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.host_diagnostic import rng,close
from dpa_ctta.hosts.vptta import model_input_from_pixels
from dpa_ctta.p1_analysis import evaluate
from dpa_ctta.p2_data import SUBSETS
from test_vptta_host import pixels


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);seed_all(20260907);m,_=build_reference_model('polyp');cls.state=copy.deepcopy(m.state_dict())
        # Nontrivial source buffers exercise restoration, not only default zero/one.
        for name in SCORE_HEAD_BN:
            cls.state[name+'.running_mean'].fill_(.2);cls.state[name+'.running_var'].fill_(1.3);cls.state[name+'.num_batches_tracked'].fill_(7)

    def test_exact_policy_and_actual_inventory(self):
        h=Host('H',self.state);check_policy(h)
        self.assertEqual((len(h.names),sum(p.numel() for p in h.params)),(300,66212))
        for name,m in h.heads.items():
            for k in FIELDS:close(getattr(m,k),self.state[name+'.'+k],exact=True)
        bad=copy.deepcopy(self.state);del bad[SCORE_HEAD_BN[0]+'.running_var']
        with self.assertRaises((ValueError,KeyError)):Host('H',bad)
        h.model.train()
        with self.assertRaises(ValueError):check_policy(h)

    def test_F_H_only_policy_differs(self):
        seed_all(20260907);f=Host('F',self.state);seed_all(20260907);h=Host('H',self.state)
        self.assertEqual(f.names,h.names);close(f.base.state_dict(),h.base.state_dict(),exact=True);close(f.rng,h.rng,exact=True)
        for (n,p),(k,q) in zip(f.model.named_parameters(),h.model.named_parameters()):self.assertEqual(n,k);close(p,q,exact=True);self.assertEqual(p.requires_grad,q.requires_grad)
        for name in SCORE_HEAD_BN:self.assertIsNone(f.heads[name].running_mean);self.assertTrue(f.heads[name].training);self.assertFalse(h.heads[name].training)
        f.finish(self.state);h.finish(self.state)

    def test_H_fixed_heads_pass_gradient(self):
        seed_all(20260907);h=Host('H',self.state);z,m=h.step(pixels('polyp',0))
        self.assertEqual(z.shape,(1,1,352,352));self.assertEqual(m['counts']['forwards'],8);self.assertEqual(m['adam_step'],1)
        self.assertTrue(all(v>0 for v in h.head_input_gradient.values()));self.assertEqual(set(h.head_input_gradient),set(SCORE_HEAD_BN));self.assertGreater(m['optimizer_update_l2'],0)
        h.finish(self.state)

    def test_H0_stateless_and_fresh_H_first_view(self):
        seed_all(20260907);h=H0(self.state);saved={};before=rng()
        self.assertFalse(hasattr(h,'base'));self.assertTrue(all(not p.requires_grad for p in h.model.parameters()))
        for i in (0,1,1,0):
            z,m=h.step(pixels('polyp',i))
            if i in saved:close(z,saved[i],exact=True)
            else:saved[i]=z
        close(before,rng(),exact=True);h.finish(self.state)
        for i in (0,1):
            fresh=Host('H',self.state)
            with torch.no_grad():z=logits(fresh.model,model_input_from_pixels(pixels('polyp',i),'polyp'))
            close(z,saved[i]);fresh.finish(self.state)

    def test_all_adapt_returns_B4_and_matches(self):
        seed_all(20260907);ref=PolypC(self.state);saved=[]
        for i in (0,1):z,_=ref.step(pixels('polyp',i));saved.append((z,capture(ref)))
        ref.finish(self.state);seed_all(20260907);h=make_host('all-adapt',self.state);self.assertIs(type(h),PolypC)
        for i in (0,1):z,_=h.step(pixels('polyp',i));close(z,saved[i][0]);close(capture(h),saved[i][1]);close(rng(),saved[i][1]['rng'],exact=True)
        h.finish(self.state)

    def test_label_and_RNG_isolation(self):
        seed_all(20260907);h=Host('F',self.state);x=pixels('polyp',0);original=x.clone();z,_=h.step(x);before=capture(h)
        a=evaluate(z.sigmoid(),torch.zeros_like(z),'polyp');b=evaluate(z.sigmoid(),torch.ones_like(z),'polyp');self.assertNotEqual(a,b);close(before,capture(h),exact=True);close(original,x,exact=True)
        # Other code may consume global RNG; Host restores its private trajectory.
        torch.rand(3);h.step(pixels('polyp',1));after=capture(h);h.finish(self.state)
        seed_all(20260907);ref=Host('F',self.state);ref.step(x);ref.step(pixels('polyp',1));close(after,capture(ref),exact=True);ref.finish(self.state)
        with self.assertRaises(TypeError):h.step(x,torch.zeros(1))

    def test_scalar_closeout_rejects_missing_duplicate_order_and_state(self):
        ordered=[dict(group_id=str(i),sample_id=str(i),domain='D',subset=s) for i,s in enumerate(SUBSETS) if s!='all_dev']
        metrics=evaluate(torch.zeros(1,1,3,3),torch.zeros(1,1,3,3),'polyp')
        def rows(arm,order):
            zero=arm=='H0'
            return [dict(r,task='polyp',arm=arm,order=order,visit=i+1,counts=dict(forwards=1 if zero else 8,backwards=int(not zero),base_adam=int(not zero),perturb=0,restore=0),adam_step=0 if zero else i+1,lr=None if zero else 1e-4,layer_policy=policy(arm) if arm in ('F','H','H0') else {},head_state_checked=True,prediction_fixed_before_label=True,frozen_parameters_checked=True,prediction_origin='canonical_stateless' if zero else 'continuous_update',stateless_checked=zero,optimizer_update_l2=0.,prediction_foreground_fraction=0.,score_heads_final_original={n:dict(mean=0.,std=1.) for n in SCORE_HEAD_BN},head_input_gradient_l2={} if zero else {n:0. for n in SCORE_HEAD_BN},diagnostics=None if zero else dict(point_bce=.1,bn_gradient_l2=0.),metrics=copy.deepcopy(metrics),host_seconds=.1,pipeline_seconds=.2,peak_allocated_bytes=1) for i,r in enumerate(ordered)]
        def old(reg,o):return {a:rows(a,o) for a in ('N','A','EA','O2','D4','C','C0')}
        good=rows('F',0)
        for bad in [good[:-1],[good[0]]*3,list(reversed(good))]:
            with self.assertRaises(ValueError):validate_rows(bad,ordered,'F',0,good)
        for key,value in [('adam_step',2),('layer_policy',policy('H')),('counts',dict(forwards=1))]:
            bad=copy.deepcopy(good);bad[0][key]=value
            with self.assertRaises(ValueError):validate_rows(bad,ordered,'F',0,good)
        totals=dict(records=15,forwards=99,backwards=12,base_adam=12,perturb=0,restore=0)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for name,v in {'run.completion.json':dict(status='B5_RUN_COMPLETE',progress=totals,trainable={},gpu_seconds=1),'smoke.completion.json':dict(physical=SMOKE),'CPU_validation.json':dict(exit_code=0)}.items():(out/name).write_text(json.dumps(v))
            (out/'polyp_canonical_H0.jsonl').write_text('\n'.join(json.dumps(r) for r in rows('H0',None)))
            for o in (0,1):
                for a in ('F','H'):(out/f'polyp_{o}_{a}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows(a,o)))
            with patch('dpa_ctta.b5_analysis.stream',side_effect=lambda r,o:ordered),patch('dpa_ctta.b5_analysis.old_records',side_effect=old),patch('dpa_ctta.b5_analysis.BUDGET',totals):
                (out/'polyp_1_H.jsonl').write_text('')
                with self.assertRaises(ValueError):recompute(out,{},dict(commit='CPU_FIXTURE'))
                self.assertFalse((out/'verification.json').exists())
                (out/'polyp_1_H.jsonl').write_text('\n'.join(json.dumps(r) for r in rows('H',1)));recompute(out,{},dict(commit='CPU_FIXTURE'))
            self.assertEqual(json.loads((out/'verification.json').read_text())['status'],COMPLETE)
