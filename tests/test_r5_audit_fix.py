"""External-review regressions: feasible metrics, partial smoke cost, independent rollback."""
import copy,json,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.r5_update_acceptance import analyze,execution,plan,host
from dpa_ctta.r5_update_acceptance.rule import PHYSICAL
from dpa_ctta.r1.host import Host as Old
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.host_diagnostic import close
from test_r5_analysis import fixture_rows,write_fixture
from test_r5_host import Toy,new,state
from test_vptta_host import pixels


def metric(pred,gt,k,n=262144,channel='OD'):
    return dict(channel=channel,total_pixels=n,pred_pixels=pred,gt_pixels=gt,intersection=k,
        dice=2*k/(pred+gt) if pred+gt else 1.,assd=1. if pred and gt else None,
        pred_empty=pred==0,pred_full=pred==n,gt_empty=gt==0,gt_full=gt==n)


class AuditFixTests(unittest.TestCase):
    def test_metric_feasibility_boundaries_and_original_dice_guard(self):
        n=262144
        for pred,gt,k in [(0,0,0),(0,100,0),(100,0,0),(100,100,90),(n,100,100),(100,n,100),(n,n,n),(0,n,0),(n,0,0),(n-10,20,10)]:
            with self.subTest(valid=(pred,gt,k)):analyze.validate_metric(metric(pred,gt,k))
        for pred,gt,k in [(n,100,99),(100,n,99),(n-10,20,9)]:
            with self.subTest(invalid=(pred,gt,k)),self.assertRaises(ValueError):analyze.validate_metric(metric(pred,gt,k))
        bad=metric(100,100,90);bad['dice']-=.1
        with self.assertRaises(ValueError):analyze.validate_metric(bad)
        for key in ('pred_pixels','gt_pixels','intersection','total_pixels'):
            bad=metric(100,100,90);bad[key]=float(bad[key])
            with self.assertRaises(ValueError):analyze.validate_metric(bad)

    def test_33_valid_shadow_rows_impossible_counts_all_predictions(self):
        ts,es,ordered=fixture_rows(n=33);ident={'test':'procedural'};job=dict(arm='C',records=33)
        for r in ts+es:r['binding']=ident
        self.assertEqual(len(analyze.join(ts,es,ordered,job,ident)),33)
        self.assertTrue(ts[-1]['trace']['eligible']);self.assertFalse(ts[-1]['trace']['shadow_accept'])
        for name in ('pre','q','trial','emit'):
            bad=copy.deepcopy(es)
            bad[-1]['evaluation']['metrics'][name]=[metric(262144,100,0,channel=c) for c in ('OD','OC')]
            # Keep emitted-branch equality, flags and reconstructed Dice consistent.
            if name in ('trial','emit'):
                for p in ('trial','emit'):bad[-1]['evaluation']['metrics'][p]=copy.deepcopy(bad[-1]['evaluation']['metrics'][name])
            with self.subTest(prediction=name),self.assertRaises(ValueError):analyze.join(ts,bad,ordered,job,ident)

    def test_complete_A_impossible_metrics_invalidate_without_scientific_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'A';reg,ordered,_=write_fixture(out,'A')
            with patch.object(analyze,'stream',return_value=ordered),patch.object(analyze,'stream_summary',return_value=dict(stream_digest='c'*64)):
                analyze.recompute(out,reg);self.assertTrue(analyze.read(out/'current_result.json')['valid'])
                p=out/'o0a0/evaluation.jsonl';rows=analyze.lines(p)
                for name in ('trial','emit'):rows[32]['evaluation']['metrics'][name]=[metric(262144,100,0,channel=c) for c in ('OD','OC')]
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):analyze.recompute(out,reg)
                current=analyze.read(out/'current_result.json');self.assertFalse(current['valid']);self.assertEqual(current['status'],'INCOMPLETE')
                self.assertFalse((out/'current').exists())

    def smoke_probe(self,target=None,point=None,recipe=None,first_failure=False):
        """Real old/R5 C paths with procedural Toy; only host construction is replaced."""
        recipe=recipe or dict(recipe='B_ALL_ARMS4_OLD_C4_V1',network_forwards=160,loss_backward_calls=20,adam_calls=20,jacobian_vjp_calls=0,seed=20260907,pixel_indices=[0,1,2,3],rtol=1e-4,atol=1e-5)
        made=[];handles=[];error=RuntimeError('injected '+str(point));observed={k:0 for k in PHYSICAL}
        def factory(arm,unused,device,p_accept=None,legacy=False):
            model=Toy();initial=copy.deepcopy(model.state_dict());h=Old('C',model=model,device=device) if legacy else host.Host(arm,model=model,device=device,p_accept=p_accept)
            core=h if legacy else h.core;name='OLD' if legacy else arm;made.append((name,h,core))
            original_finish=h.finish;h.finish=lambda _:original_finish(initial)
            # Independent hooks observe actual model/autograd/optimizer calls.
            def forward(m,x,z):
                observed['network_forwards']+=1
                if name==target and point=='forward' and core.counts['forwards']==11:raise error
            def backward(g):observed['loss_backward_calls']+=1;return g
            def before(opt,args,kwargs):
                if name==target and point=='backward' and core.counts['backwards']==2:raise error
            def after(opt,args,kwargs):
                observed['adam_calls']+=1
                if name==target and point=='adam' and core.counts['base_adam']==2:raise error
            handles.extend([core.model.register_forward_hook(forward),core.params[0].register_hook(backward),core.base.register_step_pre_hook(before),core.base.register_step_post_hook(after)])
            return h
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            out=Path(tmp)
            stack.enter_context(patch('dpa_ctta.r1.host.Host',side_effect=lambda a,s,d:factory(a,s,d,legacy=True)))
            original=host.Host
            # Use the saved real class inside factory while patching smoke's local import.
            def make_new(a,s,d,p_accept=None):
                with patch.object(host,'Host',original):return factory(a,s,d,p_accept)
            stack.enter_context(patch.object(host,'Host',side_effect=make_new))
            if first_failure:(out/'smoke.failure.json').write_text('{"first_failure": true}\n')
            try:
                if target:
                    with self.assertRaises(RuntimeError) as caught:execution.smoke(None,'cpu',out,{},recipe,.5)
                    self.assertIs(caught.exception,error);self.assertFalse((out/'smoke.completion.json').exists())
                    failure=analyze.read(out/'smoke.failure.json')
                    if first_failure:self.assertEqual(failure,{'first_failure':True})
                    else:
                        self.assertEqual(failure['physical'],observed)
                        index=['OLD','C','C_HALF','C_RANDOM','C_VERIFY'].index(target)
                        partial={'forward':(11,1,1),'backward':(15,2,1),'adam':(15,2,2)}[point]
                        self.assertEqual(failure['physical'],dict(network_forwards=index*32+partial[0],loss_backward_calls=index*4+partial[1],adam_calls=index*4+partial[2],jacobian_vjp_calls=0))
                        self.assertEqual(failure['reason'],str(error))
                        self.assertIn('LOWER_BOUND',failure['physical_accounting'])
                    self.assertEqual([a for a,_,_ in made],['OLD','C','C_HALF','C_RANDOM','C_VERIFY'][:index+1] if not first_failure else ['OLD'])
                else:
                    execution.smoke(None,'cpu',out,{},recipe,.5);done=analyze.read(out/'smoke.completion.json')
                    self.assertEqual(done['physical'],{k:recipe[k] for k in PHYSICAL});self.assertEqual(done['physical'],observed)
                    self.assertFalse((out/'smoke.failure.json').exists())
            finally:
                for h in handles:h.remove()

    def test_smoke_partial_forward_all_hosts(self):
        for arm in ('OLD','C','C_HALF','C_RANDOM','C_VERIFY'):
            with self.subTest(arm=arm):self.smoke_probe(arm,'forward')
    def test_smoke_after_backward_before_Adam_all_hosts(self):
        for arm in ('OLD','C','C_HALF','C_RANDOM','C_VERIFY'):
            with self.subTest(arm=arm):self.smoke_probe(arm,'backward')
    def test_smoke_after_Adam_before_return_all_hosts(self):
        for arm in ('OLD','C','C_HALF','C_RANDOM','C_VERIFY'):
            with self.subTest(arm=arm):self.smoke_probe(arm,'adam')
    def test_smoke_success_A_B_exact_budgets(self):
        self.smoke_probe(recipe=plan.science()['A_smoke_proposal']);self.smoke_probe()
    def test_smoke_preserves_first_failure_and_original_exception(self):self.smoke_probe('OLD','forward',first_failure=True)

    def test_rollback_next_step_reference_never_attempted_candidate(self):
        for initialized in (False,True):
            with self.subTest(initialized=initialized):
                h=new('C_VERIFY');seed_all(20260907);reference=Old('C',model=Toy())
                def reference_state():return dict(params=[p.clone() for p in reference.params],adam=copy.deepcopy(reference.base.state_dict()),rng=copy.deepcopy(reference.rng))
                if initialized:
                    h.step(pixels('fundus',0));h.take_evaluation().clear();reference.step(pixels('fundus',0))
                before=reference_state();original=h.rule.decide
                def reject(obs):return dict(original(obs),accept=False)
                with patch.object(h.rule,'decide',side_effect=reject):h.step(pixels('fundus',1))
                h.take_evaluation().clear()
                # Old C never runs the rejected input and never snapshots/restores model or Adam.
                # Only synchronize the consumed augmentation stream for the next input.
                reference.rng=copy.deepcopy(h.core.rng)
                close(before['params'],[p for p in h.params],exact=True);close(before['adam'],h.base.state_dict(),exact=True)
                with patch.object(host,'snapshot',side_effect=AssertionError('reference must not snapshot')),patch.object(host,'rollback',side_effect=AssertionError('reference must not rollback')):
                    expected,_=reference.step(pixels('fundus',2))
                actual,_=h.step(pixels('fundus',2));h.take_evaluation().clear()
                close(actual,expected,exact=True);close(state(h),reference_state(),exact=True)
                self.assertEqual(reference.counts['forwards'],8*(1+int(initialized)))
                self.assertEqual(h.core.counts['forwards'],reference.counts['forwards']+8)
