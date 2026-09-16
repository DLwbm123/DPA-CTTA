import copy,json,random,runpy,tempfile,unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from dpa_ctta.r5_update_acceptance import analyze,plan,run,execution
from dpa_ctta.r5_update_acceptance.rule import Rule,PHYSICAL
from dpa_ctta.r1.plan import binding,registration_digest
from dpa_ctta.r1.evidence import write
from test_r5_rule import observation


def metric(channel,intersection=90):
    return dict(channel=channel,dice=intersection/100,assd=1.,gt_empty=False,gt_full=False,pred_empty=False,pred_full=False,pred_pixels=100,gt_pixels=100,intersection=intersection,total_pixels=262144)


def fixture_rows(arm='C',n=1951,p_accept=None):
    rule=Rule(arm,p_accept);traces=[];labels=[];ordered=[];committed=forced=eligible=0
    for i in range(1,n+1):
        entry=dict(sample_id='s'+str(i),group_id='g'+str(i),domain='D'+str(i%4),subset='remaining_dev' if i<=1695 else 'legacy_dev' if i<=1823 else 'p1_extension_dev');ordered.append(entry)
        z=rule.decide(observation(.01,trial=.3 if i%3==0 else .1));rule.append(z['r']);committed+=z['accept'];forced+=z['forced'];eligible+=z['eligible']
        z.update(arm=arm,counts=PHYSICAL.copy(),totals=dict(n_visits=i,n_candidate_adam=i,n_committed=committed,n_rejected=i-committed,n_forced=forced,n_eligible=eligible,parameter_restorations=i-committed),adam_committed_step=committed,buffer_count_after=len(rule.history),source_unchanged=True,transaction_complete=True,host_seconds=0.)
        ident=dict(visit=i,**entry);traces.append(dict(**ident,trace=z))
        pre=[metric(c) for c in ('OD','OC')];trial=[metric(c,89 if not z['shadow_accept'] else 90) for c in ('OD','OC')]
        labels.append(dict(**ident,evaluation=dict(transaction_before_GT=True,metrics=dict(pre=pre,q=copy.deepcopy(pre),trial=trial,emit=copy.deepcopy(trial if z['accept'] else pre)))))
    return traces,labels,ordered


def write_fixture(out,scope,cal=None,reuse=None):
    out.mkdir();reg=dict(checkpoint_only=True,fixture='PROCEDURAL_ONLY');ident=dict(run_id='a'*32,code_sha='b'*40,science_sha256=plan.SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest='c'*64,c_fingerprint=plan.fingerprint(),scope=scope)
    jobs=plan.matrix(scope);recipe=plan.science()['A_smoke_proposal'] if scope=='A' else dict(recipe='B_ALL_ARMS4_OLD_C4_V1',network_forwards=160,loss_backward_calls=20,adam_calls=20,jacobian_vjp_calls=0,seed=20260907,pixel_indices=[0,1,2,3],rtol=1e-4,atol=1e-5)
    receipt=dict(binding=ident,jobs=jobs,devices=[dict(index=0,uuid='CPU_FIXTURE')],schedule=plan.allocation(jobs,1),caps=dict(trajectory_seconds=1e5,wall_seconds=1e6,active_seconds=1e6,bytes=1024**3),smoke_recipe=recipe,calibration=cal,reuse_A=str(reuse) if reuse else None)
    write(out/'receipt.json',receipt);processes=[]
    write(out/'R5_SCOPE.json',dict(schema='R5_UPDATE_ACCEPTANCE_V1',scope=scope,run_id=ident['run_id']))
    for key,phase,identity in [('device0','smoke',binding(receipt,0))]+[(j['job_id'],'formal',binding(receipt,0,j)) for j in jobs]:
        pid=100+len(processes);processes.append(dict(key=key,phase=phase,binding=identity,pid=pid,pgid=pid,status='EXITED',exit_code=0))
    (out/'device0').mkdir();write(out/'device0/smoke.completion.json',dict(binding=binding(receipt,0),status='MECHANICAL_SMOKE_COMPLETE',C_parity_valid=True,physical={k:recipe[k] for k in PHYSICAL},recipe=recipe))
    ordered=None
    for j in jobs:
        t,e,ordered=fixture_rows(j['arm'],p_accept=cal['p_accept'] if j['arm']=='C_RANDOM' else None);p=out/j['job_id'];p.mkdir();b=binding(receipt,0,j)
        for name,rows in [('unlabeled.jsonl',t),('evaluation.jsonl',e)]:
            with (p/name).open('w') as f:
                for row in rows:row['binding']=b;f.write(json.dumps(row)+'\n')
        write(p/'completion.json',dict(binding=b,status='TRAJECTORY_COMPLETE',records=1951,physical={k:v*1951 for k,v in PHYSICAL.items()},totals=t[-1]['trace']['totals'],seconds=1.))
    write(out/'matrix.processes.json',dict(binding=ident,status='COMPUTE_COMPLETE',processes=processes,exit_codes=[0]*len(processes),unstarted_jobs=[],wall_seconds=1.,active_seconds=1.))
    write(out/'processes.started.json',dict(binding=ident,processes=[{k:p[k] for k in ('pid','pgid','binding','phase','key')} for p in processes]))
    return reg,ordered,receipt


class AnalysisTests(unittest.TestCase):
    def test_scalar_replay_all_arms_corruption_and_join(self):
        for arm in ('C','C_HALF','C_RANDOM','C_VERIFY'):
            ts,es,order=fixture_rows(arm,150,.6 if arm=='C_RANDOM' else None);job=dict(arm=arm,records=150);identity={'test':'procedural'}
            for row in ts+es:row['binding']=identity
            result=analyze.join(ts,es,order,job,identity,.6 if arm=='C_RANDOM' else None);self.assertEqual(len(result),150)
            for change in ('duplicate','missing','order','SSE','window','Q90','accept','step','causal','label'):
                t=copy.deepcopy(ts);e=copy.deepcopy(es)
                if change=='duplicate':t[90]=t[89]
                elif change=='missing':t.pop()
                elif change=='order':t[80],t[81]=t[81],t[80]
                elif change=='SSE':t[40]['trace']['e_pre']['sse']+=1
                elif change=='window':t[40]['trace']['past_count']+=1
                elif change=='Q90':t[40]['trace']['q90_past']+=.001
                elif change=='accept':t[40]['trace']['accept']=not t[40]['trace']['accept']
                elif change=='step':t[40]['trace']['adam_committed_step']+=1
                elif change=='causal':e[40]['evaluation']['transaction_before_GT']=False
                else:e[40]['evaluation']['metrics']['emit'][0]['dice']+=.1
                with self.assertRaises((ValueError,KeyError)):analyze.join(t,e,order,job,identity,.6 if arm=='C_RANDOM' else None)
    def test_label_free_calibration_and_analytical_random_reference(self):
        ts,es,ordered=fixture_rows();sets={o:copy.deepcopy(ts) for o in (0,1,4)}
        a=analyze.calibration(sets,{});self.assertEqual(a['denominator'],3*(1951-32));self.assertFalse(a['uses_labels'])
        for e in es:e['evaluation']['metrics']['pre'][0]['dice']=0
        self.assertEqual(a,analyze.calibration(sets,{}))
        for rows in sets.values():
            for r in rows:r['trace']['eligible']=False
        zero=analyze.calibration(sets,{});self.assertIsNone(zero['p_accept']);self.assertEqual(zero['status'],'INVALID_ZERO_ELIGIBLE')
        rows=[dict(**t,metrics=e['evaluation']['metrics'],L=[-1 if not t['trace']['shadow_accept'] else 0]*2) for t,e in zip(ts,es)]
        d=analyze.diagnostic(rows);self.assertGreater(d['G'],d['G_random']);self.assertAlmostEqual(d['G_random'],(1-d['eligible_acceptance_rate'])*d['G'])
        uneven=[dict(domain='D0',x=0)]*100+[dict(domain='D'+str(i),x=4) for i in (1,2,3)]
        self.assertEqual(analyze.domain_mean(uneven,lambda r:r['x']),3.)
    def test_A_gate_boundaries_and_mechanical_failures(self):
        base={o:dict(eligible=100,eligible_rejected=50,eligible_acceptance_rate=.5,G=.2,G_random=.19) for o in ('0','1','4')}
        self.assertTrue(analyze.gate_A(base)['eligible_for_review'])
        for order,key,value in [('0','G',-.001),('1','G',.199),('4','G',-.001),('0','G_random',.2),('0','eligible_acceptance_rate',.499),('1','eligible_rejected',0),('4','eligible',0)]:
            x=copy.deepcopy(base);x[order][key]=value;self.assertFalse(analyze.gate_A(x)['eligible_for_review'])
        with self.assertRaises(ValueError):analyze.gate_A(base,False)
        self.assertFalse(analyze.gate_A(base)['next_execution_authorized'])
    def test_B_gate_boundaries(self):
        v=dict(C=[.5]*4,C_HALF=[.2]*4,C_RANDOM=[.2]*4)
        self.assertTrue(analyze.gate_B(v,[-2]*4,-.1)['passed'])
        for k in v:
            bad=copy.deepcopy(v);bad[k][0]-=.001;self.assertFalse(analyze.gate_B(bad,[-2]*4,-.1)['passed'])
        self.assertFalse(analyze.gate_B(v,[-2.001]*4,-.1)['passed']);self.assertFalse(analyze.gate_B(v,[-2]*4,-.101)['passed'])
    def test_3_17_20_budget_scopes_default_entry_and_neutral_protection(self):
        a,b,alljobs=[plan.matrix(s) for s in ('A','B_NEW','AB')]
        self.assertEqual([len(a),len(b),len(alljobs)],[3,17,20]);self.assertFalse({j['job_id'] for j in a}&{j['job_id'] for j in b})
        self.assertEqual([x['worker'] for x in plan.allocation(a,3)['assignments']],[0,1,2])
        self.assertEqual(sum(j['records'] for j in alljobs),39020);self.assertEqual(sum(j['network_forwards'] for j in alljobs),312160)
        with patch('sys.argv',['neutral','--run']),patch('subprocess.check_output',side_effect=AssertionError('no device/asset queries')):
            with self.assertRaises(PermissionError):run.main()
            with self.assertRaises(PermissionError):execution.launch({'enabled':False},{'registration':{}},'/unused')
        calls=[]
        with patch('dpa_ctta.b3_runtime.neutral_subprocesses',side_effect=lambda:calls.append('neutral')),patch.object(run,'main',side_effect=lambda:calls.append('main')):runpy.run_path(str(plan.ROOT/'scripts/run_r5.py'),run_name='__main__')
        self.assertEqual(calls,['neutral','main'])
    def test_new_scope_review_waiver_caps_and_calibration_bindings(self):
        auth=dict(enabled=True,scope='A',approved_code_sha='b'*40,approved_science_sha256=plan.SCIENCE_SHA,approved_registration_digest=registration_digest({}),approved_stream_digest='c'*64,approved_c_fingerprint=plan.fingerprint(),approved_trajectory_count=3,allowed_physical_gpu_ids=[5,6,7],max_workers=3,background_allowed=False,caps=dict(trajectory_seconds=3600,wall_seconds=10000,active_seconds=30000,bytes=1024**3),smoke_recipe=plan.science()['A_smoke_proposal'],external_review=dict(status='NOT_RUN'),user_waiver=dict(explicit=True,reference='PROGRAMMATIC_TEST_ONLY',code_sha='b'*40,scope='A'))
        with patch.object(plan,'stream_summary',return_value=dict(stream_digest='c'*64)):
            self.assertEqual(plan.authorize(auth,{},'b'*40),[5,6,7])
            for mutation in ({'scope':'AB'},{'scope':'B_NEW'},{'approved_code_sha':'c'*40},{'caps':None},{'smoke_recipe':None},{'calibration':{'p_accept':.7}},{'user_waiver':{'explicit':True,'reference':'R4 old waiver','code_sha':'b'*40,'scope':'R4'}}):
                with self.assertRaises(PermissionError):plan.authorize(dict(auth,**mutation),{},'b'*40)
    def test_complete_A_then_separate_17_B_reuse_and_atomic_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);A=root/'A';reg,ordered,receipt=write_fixture(A,'A')
            with patch.object(analyze,'stream',return_value=ordered),patch.object(analyze,'stream_summary',return_value=dict(stream_digest='c'*64)):
                result=analyze.recompute(A,reg);self.assertEqual(result['status'],'R5A_DIAGNOSTIC_COMPLETE_ELIGIBLE_FOR_REVIEW');self.assertFalse(result['next_execution_authorized']);self.assertIsNone(result['H_t'])
                self.assertFalse((root/'B').exists());cal=analyze.read(A/'label_free_calibration.json')
                old,rows=analyze.reuse_A(A,reg,cal);self.assertEqual(len(rows),3)
                bad=copy.deepcopy(cal);bad['numerator']+=1
                with self.assertRaises(ValueError):analyze.reuse_A(A,reg,bad)
                B=root/'B';write_fixture(B,'B_NEW',cal,A);resultB=analyze.recompute(B,reg)
                self.assertEqual(resultB['status'],'R5B_EXPERIMENT_COMPLETE');self.assertEqual(len(resultB['trajectories']),20);self.assertEqual(resultB['physical']['adam_calls'],33167);self.assertEqual(resultB['physical_total']['adam_calls'],39020)
                self.assertEqual(resultB['reuse_A_binding']['code_sha'],'b'*40)
                (A/'o0a0/unlabeled.jsonl').write_text('')
                with self.assertRaises(ValueError):analyze.recompute(A,reg)
                self.assertFalse(analyze.read(A/'current_result.json')['valid'])
                with self.assertRaises(ValueError):analyze.recompute(B,reg)
                self.assertFalse(analyze.read(B/'current_result.json')['valid'])
    def test_foreign_historical_results_are_never_invalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old={'status':'R4T_EXPERIMENT_COMPLETE','valid':True};write(p/'current_result.json',old)
            with self.assertRaises(FileNotFoundError):analyze.recompute(p,{})
            self.assertEqual(analyze.read(p/'current_result.json'),old)
            write(p/'R5_SCOPE.json',{'schema':'NOT_R5'})
            with self.assertRaises(ValueError):analyze.recompute(p,{})
            self.assertEqual(analyze.read(p/'current_result.json'),old)
