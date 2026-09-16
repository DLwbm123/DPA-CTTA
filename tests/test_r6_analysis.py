"""Complete procedural ledgers are test fixtures, never registered-metadata evidence."""
import copy,hashlib,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dpa_ctta.r6_regional_consistency import analyze,plan
from dpa_ctta.r6_regional_consistency.loss import ARMS,PHYSICAL
from dpa_ctta.r1.plan import binding,registration_digest
from dpa_ctta.r1.evidence import write
from test_r5_audit_fix import metric


def fixture_rows(arm='C',n=1951):
    ts=[];es=[];ordered=[];N=262144
    for i in range(1,n+1):
        entry=dict(sample_id='s'+str(i),group_id='g'+str(i),domain='D'+str(i%4),subset='remaining_dev' if i<=1695 else 'legacy_dev' if i<=1823 else 'p1_extension_dev');ordered.append(entry)
        cs=[]
        for c,ch in enumerate(('OD','OC')):
            seed=int.from_bytes(hashlib.sha256(f'R6_WEIGHT_PERM_V1|20260907|{i}|{c}'.encode()).digest()[:8],'big')%(2**63)
            cs.append(dict(channel=ch,n_fg=N//2,n_bg=N//2,rho=1.,rho_clipped=False,w_fg=1.,w_bg=1.,fallback=None,applied_w_fg=1.,applied_w_bg=1.,mean_weight=1.,S0=0.,Sw=0.,Sperm=0.,a=1.,b=1.,seed=seed,bce_sum=N/2,bce_fg_sum=N/4,bce_bg_sum=N/4,weighted_bce_sum=N/2,shuffled_bce_sum=N/2,residual_fg_sse=0.,residual_bg_sse=0.,weighted_fg_sse=0.,weighted_bg_sse=0.,residual_weighted_dot=0.,residual_shuffled_dot=0.,weight_residual_dot=0.,weight_squared_sum=N,permutation_histogram_preserved=True,permutation_changed_positions=0,base_weight_sha256='0'*64,shuffled_weight_sha256='0'*64,expected_logit_gradient_l2=0.,actual_logit_gradient_l2=0.))
        trace=dict(arm=arm,visit=i,counts=PHYSICAL.copy(),cumulative={k:v*i for k,v in PHYSICAL.items()},adam_step=i,channels=cs,actual_loss=.5,bn_gradient_l2=0.,adam_affine_displacement_l2=0.,source_unchanged=True,transaction_complete=True,host_seconds=0.)
        ident=dict(visit=i,**entry);ts.append(dict(**ident,trace=trace))
        pre=[metric(100,100,90,channel=c) for c in ('OD','OC')];post=[metric(100,100,91 if arm=='R_BAL' else 90,channel=c) for c in ('OD','OC')]
        es.append(dict(**ident,evaluation=dict(transaction_before_GT=True,metrics=dict(pre=pre,q=[metric(N//2,100,90,channel=c) for c in ('OD','OC')],post=post),evaluator_seconds=0.,pipeline_seconds=0.)))
    return ts,es,ordered


def write_fixture(out,scope,reuse=None):
    out.mkdir();reg=dict(fixture='PROCEDURAL_ONLY');ident=dict(run_id=('a' if scope=='A' else 'b')*32,code_sha=('b' if scope=='A' else 'd')*40,science_sha256=plan.SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest='c'*64,production_fingerprint=plan.fingerprint(),scope=scope)
    jobs=plan.matrix(scope);recipe=plan.SMOKE
    receipt=dict(binding=ident,jobs=jobs,devices=[dict(index=0,uuid='CPU_FIXTURE')],schedule=plan.allocation(jobs,1),caps=plan.CAPS,smoke_recipe=recipe,reuse_A=str(reuse) if reuse else None,reuse_A_binding=analyze.read(reuse/'receipt.json')['binding'] if reuse else None)
    write(out/'receipt.json',receipt);write(out/'R6_SCOPE.json',dict(schema='R6_REGIONAL_CONSISTENCY_V1',scope=scope,run_id=ident['run_id']));processes=[]
    for key,phase,identity in [('device0','smoke',binding(receipt,0))]+[(j['job_id'],'formal',binding(receipt,0,j)) for j in jobs]:
        pid=100+len(processes);processes.append(dict(key=key,phase=phase,binding=identity,pid=pid,pgid=pid,status='EXITED',exit_code=0))
    (out/'device0').mkdir();write(out/'device0/smoke.completion.json',dict(binding=binding(receipt,0),status='MECHANICAL_SMOKE_COMPLETE',C_parity_valid=True,physical={k:recipe[k] for k in PHYSICAL},recipe=recipe))
    for j in jobs:
        ts,es,ordered=fixture_rows(j['arm']);p=out/j['job_id'];p.mkdir();b=binding(receipt,0,j)
        for name,rows in [('unlabeled.jsonl',ts),('evaluation.jsonl',es)]:
            with (p/name).open('w') as f:
                for row in rows:row['binding']=b;f.write(json.dumps(row)+'\n')
        write(p/'completion.json',dict(binding=b,status='TRAJECTORY_COMPLETE',records=1951,physical={k:v*1951 for k,v in PHYSICAL.items()},adam_step=1951,peak_allocated_bytes=0,seconds=1.))
    write(out/'matrix.processes.json',dict(binding=ident,status='COMPUTE_COMPLETE',processes=processes,exit_codes=[0]*len(processes),unstarted_jobs=[],wall_seconds=1.,active_seconds=1.))
    write(out/'processes.started.json',dict(binding=ident,processes=[{k:p[k] for k in ('pid','pgid','binding','phase','key')} for p in processes]))
    return reg,ordered,receipt


class AnalysisTests(unittest.TestCase):
    def test_all_arms_ledger_corruptions_and_metrics(self):
        for arm in ARMS:
            ts,es,ordered=fixture_rows(arm,8);identity={'fixture':True};job=dict(arm=arm,records=8)
            for r in ts+es:r['binding']=identity
            self.assertEqual(len(analyze.join(ts,es,ordered,job,identity)),8)
            for change in ('duplicate','missing','order','weight','cap','a','b','seed','S0','step','count','causal','dice','pre','q','post','GT','q_partition'):
                t,e=copy.deepcopy(ts),copy.deepcopy(es);z=t[1]['trace'];c=z['channels'][0]
                if change=='duplicate':t[1]=t[0]
                elif change=='missing':t.pop()
                elif change=='order':t[1],t[2]=t[2],t[1]
                elif change=='weight':c['w_fg']=1.5
                elif change=='cap':c['rho_clipped']=True
                elif change in ('a','b','seed','S0'):c[change]+=1
                elif change=='step':z['adam_step']-=1
                elif change=='count':z['counts']['network_forwards']+=1
                elif change=='causal':e[1]['evaluation']['transaction_before_GT']=False
                elif change=='dice':e[1]['evaluation']['metrics']['post'][0]['dice']+=.1
                elif change=='q_partition':e[1]['evaluation']['metrics']['q'][0]=metric(100,100,90)
                elif change=='GT':e[1]['evaluation']['metrics']['post'][0]=metric(100,99,90)
                else:e[1]['evaluation']['metrics'][change]=[metric(262144,100,99,channel=ch) for ch in ('OD','OC')]
                with self.subTest(arm=arm,change=change),self.assertRaises(ValueError):analyze.join(t,e,ordered,job,identity)
    def test_gate_equal_boundaries_and_each_independent_failure(self):
        for stage,n in [('A',2),('B_NEW',4)]:
            values=dict(C=[.5]*n,R_SCALE=[.2]*n,R_SHUFFLE=[.2]*n);ds=[-2.]*4;rec={c:-.1 for c in analyze.CONTROLS}
            self.assertTrue(analyze.gate(values,ds,rec,stage)['passed'])
            self.assertFalse(analyze.gate(values,ds,rec,stage)['next_execution_authorized'])
            for c in values:
                bad=copy.deepcopy(values);bad[c][0]-=.001;self.assertFalse(analyze.gate(bad,ds,rec,stage)['passed'])
                bad=copy.deepcopy(values);bad[c]=[-.01,2.] if n==2 else [0.,0.,2.,2.];self.assertFalse(analyze.gate(bad,ds,rec,stage)['passed'])
                self.assertFalse(analyze.gate(values,ds,dict(rec,**{c:-.1001}),stage)['passed'])
            for i in range(4):
                bad=ds.copy();bad[i]-=.001;self.assertFalse(analyze.gate(values,bad,rec,stage)['passed'])
            if n==4:
                bad=copy.deepcopy(values);bad['C']=[-.501,1.,1.,1.];self.assertFalse(analyze.gate(bad,ds,rec,stage)['passed'])
            with self.assertRaises(ValueError):analyze.gate({},ds,rec,stage)
    def test_full_A_then_B_reuse_original_bindings_and_invalidated_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);A=root/'A';reg,ordered,receipt=write_fixture(A,'A')
            with patch.object(analyze,'stream',return_value=ordered),patch.object(analyze,'stream_summary',return_value=dict(stream_digest='c'*64)):
                result=analyze.recompute(A,reg);self.assertEqual(result['status'],'R6A_COMPLETE_ELIGIBLE_FOR_REVIEW');self.assertFalse(result['next_execution_authorized']);self.assertEqual(len(result['trajectories']),12)
                self.assertFalse((root/'B').exists());old,rows=analyze.reuse_A(A,reg,receipt['binding']);self.assertEqual(len(rows),12)
                with self.assertRaises(PermissionError):analyze.reuse_A(A,reg,dict(receipt['binding'],code_sha='f'*40))
                B=root/'B';write_fixture(B,'B_NEW',A);r=analyze.recompute(B,reg)
                self.assertEqual(r['status'],'R6_EXPERIMENT_COMPLETE');self.assertEqual(len(r['trajectories']),20);self.assertEqual(r['physical_formal']['adam_calls'],15608);self.assertEqual(r['physical_total_formal']['adam_calls'],39020)
                self.assertEqual(r['reuse_A_binding'],receipt['binding'])
                p=A/'o0a0/evaluation.jsonl';bad=analyze.lines(p);bad[0]['evaluation']['metrics']['q'][0]=metric(262144,100,99)
                p.write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaises(ValueError):analyze.recompute(A,reg)
                self.assertFalse(analyze.read(A/'current_result.json')['valid']);self.assertEqual(analyze.read(A/'current_result.json')['status'],'INCOMPLETE')
                with self.assertRaises(ValueError):analyze.recompute(B,reg)
                self.assertFalse(analyze.read(B/'current_result.json')['valid'])
    def test_foreign_marker_run_scope_never_invalidate_history(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);old=dict(valid=True,status='R5A_DIAGNOSTIC_COMPLETE_NO_ADVANCE');write(p/'current_result.json',old)
            with self.assertRaises(FileNotFoundError):analyze.recompute(p,{})
            write(p/'receipt.json',dict(binding=dict(scope='A',run_id='original')))
            for marker in [dict(schema='R5_UPDATE_ACCEPTANCE_V1',scope='A',run_id='original'),dict(schema='R6_REGIONAL_CONSISTENCY_V1',scope='B_NEW',run_id='original'),dict(schema='R6_REGIONAL_CONSISTENCY_V1',scope='A',run_id='other')]:
                write(p/'R6_SCOPE.json',marker,replace=True)
                with self.assertRaises(ValueError):analyze.recompute(p,{})
                self.assertEqual(analyze.read(p/'current_result.json'),old)
                with self.assertRaises(ValueError):analyze.reuse_A(p,{}, {})
