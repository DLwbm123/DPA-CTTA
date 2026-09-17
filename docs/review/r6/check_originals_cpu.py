"""Fresh CPU comparison; importing the original never executes its file-writing main."""
import hashlib, importlib.util, json, os, platform, subprocess, time, unittest
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from dpa_ctta.r6_regional_consistency import loss as production, plan
from dpa_ctta.r6_regional_consistency.analyze import gate

ROOT=plan.ROOT
HERE=ROOT/'docs/review/r6'
ORIGINAL=HERE/'originals/r6_plan'
spec=importlib.util.spec_from_file_location('original_math',ORIGINAL/'R6_MATH_REFERENCE.py')
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
DETAILS=[]


class OriginalComparison(unittest.TestCase):
    def test_original_bytes_and_configuration_binding(self):
        manifest=json.loads((ORIGINAL/'MANIFEST.json').read_bytes())
        for name,entry in manifest.items():
            b=(ORIGINAL/name).read_bytes()
            self.assertEqual(len(b),entry['bytes'],name)
            self.assertEqual(hashlib.sha256(b).hexdigest(),entry['sha256'],name)
        self.assertEqual(plan.SCIENCE.read_bytes(),(ORIGINAL/'R6_SCIENCE_PROPOSAL.json').read_bytes())
        s=plan.science()
        self.assertEqual(s['status'],'DESIGN_PROPOSAL_NOT_IMPLEMENTED')
        self.assertEqual(s['base_publication_sha'],plan.BASE)
        self.assertEqual(s['registration_digest'],plan.REGISTRATION)
        self.assertEqual(s['stream_digest'],plan.RECURRENCE)
        self.assertEqual(s['arms'],list(production.ARMS))
        dry=plan.dry_run()
        self.assertEqual(dry['budgets'],s['budgets'])
        for scope,stage in s['stages'].items():
            jobs=plan.matrix(scope)
            self.assertEqual(len(jobs),stage['jobs'])
            self.assertEqual(sorted({j['order'] for j in jobs}),stage['orders'])
            self.assertEqual(sum(j['reused_from_A'] for j in jobs),12 if scope=='AB' else 0)
        smoke=s['smoke_proposal']; actual=plan.SMOKE
        self.assertEqual({k:actual[k] for k in smoke['per_device']},smoke['per_device'])
        self.assertEqual(actual['seed'],smoke['seed'])
        self.assertEqual(actual['pixel_indices'],smoke['pixel_indices'])
        self.assertEqual(actual['rtol'],smoke['C_parity']['GPU_rtol'])
        self.assertEqual(actual['atol'],smoke['C_parity']['GPU_atol'])
        caps=dict(s['resources']['proposed_hard_caps'])
        caps['bytes']=caps.pop('private_output_bytes')
        self.assertEqual(plan.CAPS,caps)
        defaults=json.loads((ROOT/'configs/r6_execution.defaults.json').read_bytes())
        self.assertFalse(defaults['enabled'])
        self.assertEqual(defaults['max_workers'],s['resources']['max_workers'])
        self.assertEqual(defaults['threads_per_worker'],s['resources']['threads_per_worker'])
        self.assertIsNone(defaults['allowed_physical_gpu_ids'])
        self.assertIsNone(defaults['caps'])
        with self.assertRaises(PermissionError):plan.authorize(defaults,{})
        with patch.object(plan,'digest',return_value='0'*64):
            with self.assertRaises(ValueError):plan.science()

    def test_common_inputs_losses_gradients_weights_and_scales(self):
        for dtype in (torch.float64,torch.float32):
            g=torch.Generator().manual_seed(771)
            z=torch.randn((1,2,11,13),dtype=dtype,generator=g)
            cases={}
            for n in (0,1,17,71,126,142,143):
                q=torch.full_like(z,.2)
                q[0,0].flatten()[:n]=.8
                q[0,1].flatten()[:143-n]=.7
                cases['foreground_'+str(n)]=(z,q)
            cases['random_soft']=(z,torch.rand(z.shape,dtype=dtype,generator=g))
            cases['zero_residual']=(torch.zeros_like(z),torch.full_like(z,.5))
            cases['tiny_residual']=(torch.full_like(z,1e-6),torch.full_like(z,.5))
            for name,(z,q) in cases.items():
                for visit in (1,17,1951):
                    before=torch.get_rng_state().clone()
                    w,wp,a,b,meta=reference.coefficients(z,q,visit)
                    ideal,_,_=production.weights(q)
                    self.assertTrue(torch.equal(ideal.to(dtype),w))
                    self.assertTrue(all(not t.requires_grad for t in (w,wp,a,b)))
                    for arm in production.ARMS:
                        x=z.clone().requires_grad_(True);y=z.clone().requires_grad_(True)
                        expected,_=reference.loss(x,q,arm,visit)
                        actual,rows=production.objective(y,q,arm,visit)
                        gx,=torch.autograd.grad(expected,x);gy,=torch.autograd.grad(actual,y)
                        tol=production.TOLERANCES[dtype]
                        torch.testing.assert_close(actual,expected,**tol)
                        torch.testing.assert_close(gy,gx,**tol)
                        for c,row in enumerate(rows):
                            self.assertEqual(row['seed'],reference.local_seed(visit,c))
                            for key in ('S0','Sw','Sperm'):
                                self.assertEqual(row[key],meta[key][c])
                            self.assertAlmostEqual(row['a'],meta['scale'][c],places=14)
                            self.assertAlmostEqual(row['b'],meta['shuffle_scale'][c],places=14)
                            self.assertEqual(row['shuffled_weight_sha256'],hashlib.sha256(wp[0,c].contiguous().numpy().tobytes()).hexdigest())
                        DETAILS.append(dict(dtype=str(dtype),case=name,visit=visit,arm=arm,
                            loss_abs_error=float((actual-expected).abs().detach()),
                            gradient_max_abs_error=float((gy-gx).abs().max())))
                    self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_original_gate_thresholds_against_production(self):
        s=plan.science();g=torch.Generator().manual_seed(903)
        for stage,key,n in [('A','A_gate',2),('B_NEW','B_gate',4)]:
            p=s[key];controls=('C','R_SCALE','R_SHUFFLE')
            thresholds={c:p['mean_R_BAL_minus_'+c+'_pp'] for c in controls}
            cases=[({c:[v]*n for c,v in thresholds.items()},[-2.]*4,{c:-.1 for c in controls})]
            for _ in range(200):
                cases.append(({c:torch.randn(n,generator=g).tolist() for c in controls},
                              (torch.randn(4,generator=g)*2).tolist(),dict(zip(controls,(torch.randn(3,generator=g)*.2).tolist()))))
            for values,domains,recurrence in cases:
                valid=all(sum(values[c])/n>=thresholds[c] for c in controls)
                valid &= all(v>=p['recurrence_delta_vs_each_control_min_pp'] for v in recurrence.values())
                if stage=='A':
                    valid &= all(v>=p['each_primary_delta_vs_each_control_min_pp'] for vs in values.values() for v in vs)
                    valid &= min(domains)>=p['each_domain_two_order_mean_R_BAL_minus_C_min_pp']
                else:
                    valid &= all(sum(v>0 for v in vs)>=p['positive_main_orders_vs_each_control_min'] for vs in values.values())
                    valid &= min(values['C'])>=p['worst_main_order_R_BAL_minus_C_min_pp']
                    valid &= min(domains)>=p['each_domain_four_order_mean_R_BAL_minus_C_min_pp']
                result=gate(values,domains,recurrence,stage)
                self.assertEqual(result['passed'],valid)
                self.assertFalse(result['next_execution_authorized'])


if __name__=='__main__':
    torch.set_num_threads(2)
    start=time.monotonic()
    with patch('torch.cuda._lazy_init',side_effect=AssertionError('Stage I GPU forbidden')):
        historical=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(reference.ReferenceTests))
        current=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OriginalComparison))
    count=lambda r:dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped))
    success=historical.wasSuccessful() and current.wasSuccessful()
    result=dict(implementation_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='NEW CPU checks; archived historical math logs untouched; does not replace 166-test implementation suite',
        python=platform.python_version(),torch=torch.__version__,seconds=time.monotonic()-start,
        original_reference_fresh_rerun=count(historical),new_comparison=count(current),
        tolerances={str(k):v for k,v in production.TOLERANCES.items()},cases=DETAILS,
        cuda_initialized=torch.cuda.is_initialized(),real_asset_reads=0,model_forwards=0,optimizer_calls=0,
        autograd_calls_for_reference_comparison=len(DETAILS)*2,reference_autograd_calls=8,
        exit_code=0 if success else 1)
    print(json.dumps(result,indent=2))
    if os.environ.get('CHECK_OUTPUT'):Path(os.environ['CHECK_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(result['exit_code'])
