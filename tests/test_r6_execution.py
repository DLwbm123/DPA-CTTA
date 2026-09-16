import copy,json,tempfile,unittest,runpy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.r6_regional_consistency import analyze,execution,plan,host,run
from dpa_ctta.r6_regional_consistency.loss import PHYSICAL
from dpa_ctta.r1.host import Host as Old
from dpa_ctta.r1.plan import registration_digest
from test_r5_host import Toy


class ExecutionTests(unittest.TestCase):
    def smoke_probe(self,target=None,point=None,recipe=None,first_failure=False):
        """Real old/R6 C paths with procedural Toy; only host construction is replaced."""
        recipe=recipe or plan.SMOKE
        made=[];handles=[];error=RuntimeError('injected '+str(point));observed={k:0 for k in PHYSICAL}
        def factory(arm,unused,device,legacy=False):
            model=Toy();initial=copy.deepcopy(model.state_dict());h=Old('C',model=model,device=device) if legacy else host.Host(arm,model=model,device=device)
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
            def make_new(a,s,d):
                with patch.object(host,'Host',original):return factory(a,s,d)
            stack.enter_context(patch.object(host,'Host',side_effect=make_new))
            if first_failure:(out/'smoke.failure.json').write_text('{"first_failure": true}\n')
            try:
                if target:
                    with self.assertRaises(RuntimeError) as caught:execution.smoke(None,'cpu',out,{},recipe)
                    self.assertIs(caught.exception,error);self.assertFalse((out/'smoke.completion.json').exists())
                    failure=analyze.read(out/'smoke.failure.json')
                    print('EXPECTED_R6_INJECTED_FAILURE '+json.dumps(dict(target=target,point=point,evidence=failure)))
                    if first_failure:self.assertEqual(failure,{'first_failure':True})
                    else:
                        self.assertEqual(failure['physical'],observed)
                        index=['OLD','C','R_BAL','R_SCALE','R_SHUFFLE'].index(target)
                        partial={'forward':(11,1,1),'backward':(15,2,1),'adam':(15,2,2)}[point]
                        self.assertEqual(failure['physical'],dict(network_forwards=index*32+partial[0],loss_backward_calls=index*4+partial[1],adam_calls=index*4+partial[2],jacobian_vjp_calls=0))
                        self.assertEqual(failure['reason'],str(error))
                        self.assertIn('LOWER_BOUND',failure['physical_accounting'])
                    self.assertEqual([a for a,_,_ in made],['OLD','C','R_BAL','R_SCALE','R_SHUFFLE'][:index+1] if not first_failure else ['OLD'])
                else:
                    execution.smoke(None,'cpu',out,{},recipe);done=analyze.read(out/'smoke.completion.json')
                    self.assertEqual(done['physical'],{k:recipe[k] for k in PHYSICAL});self.assertEqual(done['physical'],observed)
                    self.assertFalse((out/'smoke.failure.json').exists())
            finally:
                for h in handles:h.remove()

    def test_smoke_partial_forward_all_hosts(self):
        for arm in ('OLD','C','R_BAL','R_SCALE','R_SHUFFLE'):
            with self.subTest(arm=arm):self.smoke_probe(arm,'forward')
    def test_smoke_after_backward_before_Adam_all_hosts(self):
        for arm in ('OLD','C','R_BAL','R_SCALE','R_SHUFFLE'):
            with self.subTest(arm=arm):self.smoke_probe(arm,'backward')
    def test_smoke_after_Adam_before_return_all_hosts(self):
        for arm in ('OLD','C','R_BAL','R_SCALE','R_SHUFFLE'):
            with self.subTest(arm=arm):self.smoke_probe(arm,'adam')
    def test_smoke_success_A_B_exact_budgets(self):
        self.smoke_probe(recipe=plan.SMOKE);self.smoke_probe()
    def test_smoke_preserves_first_failure_and_original_exception(self):self.smoke_probe('OLD','forward',first_failure=True)

    def test_matrix_defaults_authorization_and_neutral_entry(self):
        a,b,total=[plan.matrix(s) for s in ('A','B_NEW','AB')]
        self.assertEqual([len(a),len(b),len(total)],[12,8,20]);self.assertFalse({j['job_id'] for j in a}&{j['job_id'] for j in b})
        self.assertEqual(sum(j['records'] for j in a),23412);self.assertEqual(sum(j['network_forwards'] for j in b),124864)
        self.assertEqual([j['arm'] for j in a[:4]],['C','R_BAL','R_SCALE','R_SHUFFLE'])
        for workers in (1,2,3):self.assertEqual([a['worker'] for a in plan.allocation(total,workers)['assignments']],[i%workers for i in range(20)])
        with patch('sys.argv',['neutral','--run']),patch('subprocess.check_output',side_effect=AssertionError('no queries')):
            with self.assertRaises(PermissionError):run.main()
            with self.assertRaises(PermissionError):execution.launch({'enabled':False},{'registration':{}},'/unused')
        calls=[]
        with patch('dpa_ctta.b3_runtime.neutral_subprocesses',side_effect=lambda:calls.append('neutral')),patch.object(run,'main',side_effect=lambda:calls.append('main')):runpy.run_path(str(plan.ROOT/'scripts/run_r6.py'),run_name='__main__')
        self.assertEqual(calls,['neutral','main'])
    def test_exact_code_science_data_stream_devices_stage_bindings(self):
        auth=dict(enabled=True,scope='A',approved_code_sha='b'*40,approved_science_sha256=plan.SCIENCE_SHA,approved_registration_digest=registration_digest({}),approved_stream_digest='c'*64,approved_production_fingerprint=plan.fingerprint(),approved_trajectory_count=12,allowed_physical_gpu_ids=[5,6,7],max_workers=3,background_allowed=False,caps=plan.CAPS,smoke_recipe=plan.SMOKE,external_review=dict(status='NOT_RUN'),user_waiver=dict(explicit=True,reference='PROCEDURAL_FIXTURE_ONLY',code_sha='b'*40,scope='A'))
        with patch.object(plan,'science',return_value={}),patch.object(plan,'stream_summary',return_value=dict(stream_digest='c'*64)):
            self.assertEqual(plan.authorize(auth,{},'b'*40),[5,6,7])
            mutations=[{'scope':'AB'},{'scope':'B_NEW'},{'caps':None},{'smoke_recipe':None},{'reuse_A':'history'},{'allowed_physical_gpu_ids':None},{'allowed_physical_gpu_ids':[True]},{'max_workers':4},{'user_waiver':{'explicit':True,'reference':'old R5 waiver','code_sha':'b'*40,'scope':'R5'}}]
            mutations += [{k:'incorrect'} for k in ('approved_code_sha','approved_science_sha256','approved_registration_digest','approved_stream_digest','approved_production_fingerprint','approved_trajectory_count')]
            for change in mutations:
                with self.subTest(change=change),self.assertRaises(PermissionError):plan.authorize(dict(auth,**change),{},'b'*40)
        with tempfile.TemporaryDirectory() as temp,patch.object(plan,'SCIENCE',Path(temp)/'missing.json'):
            with self.assertRaises(FileNotFoundError):plan.authorize(auth,{},'b'*40)
