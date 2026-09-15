"""Finite-matrix and scalar-closeout tests; all identities/metrics are synthetic."""
import copy,json,os,tempfile,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from dpa_ctta.r4_three import analyze,plan,execution,run
from dpa_ctta.r1.evidence import write
from dpa_ctta.r1.plan import binding,registration_digest,science as primary_science
from test_r1_fixes import scalar_row


@contextmanager
def fixture(out):
    cfg=copy.deepcopy(plan.science());jobs=plan.matrix()['jobs'];orders=primary_science()['orders']
    rows=[dict(group_id=str(i*3+j),sample_id=str(i*3+j),domain=d,subset=s) for i,d in enumerate(orders[0]) for j,s in enumerate(('remaining_dev','legacy_dev','p1_extension_dev'))]
    reg=dict(checkpoint=dict(bytes=7),target=rows,fixture='PROGRAMMATIC_SCALARS')
    def ordered(reg,index):
        ds=orders[index] if index<4 else orders[0][1:]+orders[0][:1]
        return [r for d in ds for r in rows if r['domain']==d]
    for j in jobs:
        j.update(records=12,network_forwards=12*(9 if j['arm'] in ('MT_RP','FT_RP') else 8),loss_backward_calls=12,adam_calls=12,jacobian_vjp_upper=0)
    for key,col in [('scoring_records','records'),('network_forwards','network_forwards'),('loss_backward_calls','loss_backward_calls'),('adam_calls','adam_calls')]:cfg['formal_budget'][key]=sum(j[col] for j in jobs)
    identity=dict(run_id='a'*32,code_sha='b'*40,science_sha256=plan.SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest='c'*64)
    packet=dict(binding=identity,jobs=jobs,devices=[dict(index=i,uuid='CPU_SLOT_'+str(i),model='PROCEDURAL') for i in range(3)],schedule=plan.allocation(jobs,3))
    write(out/'receipt.json',packet);entries=[];backend=dict(seed=20260907,deterministic_algorithms=False,warn_only=False,cublas_workspace_config=None)
    for i in range(3):
        p=out/('device'+str(i));p.mkdir();ident=binding(packet,i)
        write(p/'smoke.completion.json',dict(binding=ident,status='MECHANICAL_SMOKE_COMPLETE',physical=dict(network_forwards=260,loss_backward_calls=32,jacobian_vjp_calls=0,adam_calls=32,actual_parameter_replacements=0),backend=backend,checkpoint_io=dict(bytes=7)))
        entries.append(dict(binding=ident,phase='smoke',key=p.name,pid=100+i,pgid=100+i,status='EXITED',exit_code=0))
    for j,assignment in zip(jobs,packet['schedule']['assignments']):
        p=out/j['job_id'];p.mkdir();ident=binding(packet,assignment['worker'],j);records=[];arm=j['arm'];rp=arm in ('RP','MT_RP','FT_RP')
        for t,e in enumerate(ordered(reg,j['order']),1):
            r=scalar_row(e,'C_PCA_REGION' if rp else 'C',j['order'],t);r.update(arm=arm,binding=ident,auxiliary_after_state_commit=True)
            a=dict(arm=arm,global_visit=t,counts=dict(network_forwards=9 if arm in ('MT_RP','FT_RP') else 8,loss_backward_calls=1,jacobian_vjp_calls=0,adam_calls=1,actual_parameter_replacements=0),pca=r['pca'],teacher_update=arm.startswith('MT'),source_unchanged=True)
            q=[dict(channel=c,total_pixels=9,gt_foreground=0,gt_background=9,pred_foreground=0,intersection=0,dice=1.,brier_sum=0.,brier=0.,foreground_brier_sum=0.,background_brier_sum=0.,foreground_brier=None,background_brier=0.) for c in ('OD','OC')]
            aux=dict(q=q,memory=dict(status='ACTUAL_COMMITTED_STUDENT_PRE_TOKENS',regions=[dict(region=i,selected_tokens=0,correct_tokens=0,incorrect_tokens=0) for i in range(4)]) if rp else dict(status='NOT_APPLICABLE'))
            if arm.startswith('G_'):
                a['graph']=dict(steps=64,grid=128,potential_undirected_edges=32512,exact_outside_allowed=True,exact_reliable_full=True,exact_fixed_nodes=True)
                aux.update(qstar=copy.deepcopy(q),graph=dict(outside_allowed_max_change=0.,reliable_max_change=0.,allowed_counts=[0,0],reliable_counts=[9,9],reliable_correct=[9,9],changed_counts=[0,0],
                    transitions=[dict(channel=c,scope=s,denominator=9 if s=='all' else 0,wrong_to_correct=0,correct_to_wrong=0,correct_unchanged=9 if s=='all' else 0,wrong_unchanged=0) for c in ('OD','OC') for s in ('all','allowed','hard_flip')]))
            if arm.startswith('K'):
                a.update(kernels={'res.conv1':{},'res.layer1.2.conv2':{}},trainable_scalars=dict(KDG=21283,K_ALL=21283,K_MAG=19264,K_FREE=23424)[arm])
            r.update(r4t=a,auxiliary=aux);records.append(r)
        (p/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        counts={k:sum(r['r4t']['counts'][k] for r in records) for k in analyze.COUNT_KEYS}
        write(p/'completion.json',dict(binding=ident,status='TRAJECTORY_COMPLETE',records=12,physical=counts,backend=backend,checkpoint_io=dict(bytes=7),seconds=1.))
        pid=100+len(entries);entries.append(dict(binding=ident,phase='formal',key=p.name,pid=pid,pgid=pid,status='EXITED',exit_code=0))
    write(out/'matrix.processes.json',dict(binding=identity,status='COMPUTE_COMPLETE',exit_codes=[0]*len(entries),processes=entries,active_seconds=1.,wall_seconds=1.,unstarted_jobs=[]))
    write(out/'processes.started.json',dict(binding=identity,processes=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]))
    with patch.object(analyze,'science',return_value=cfg),patch.object(analyze,'matrix',return_value=dict(jobs=jobs)),patch.object(analyze,'stream',side_effect=ordered),patch.object(analyze,'stream_summary',return_value=dict(stream_digest='c'*64)):
        yield reg,jobs,packet,ordered


class ExecutionTests(unittest.TestCase):
    def test_entry_installs_neutral_subprocesses_before_dispatch(self):
        import runpy
        calls=[]
        with patch('dpa_ctta.b3_runtime.neutral_subprocesses',side_effect=lambda:calls.append('neutral')),patch.object(run,'main',side_effect=lambda:calls.append('dispatch')):
            runpy.run_path(str(plan.ROOT/'scripts/run_r4t.py'),run_name='__main__')
        self.assertEqual(calls,['neutral','dispatch'])

    def test_seventy_complete_and_atomic_invalid_after_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with fixture(out) as (reg,jobs,packet,ordered):
                value=analyze.recompute(out,reg)
                self.assertEqual(value['status'],'R4T_EXPERIMENT_COMPLETE');self.assertEqual(value['physical']['records'],840)
                self.assertEqual(len(value['mechanism']),70);self.assertEqual(value['secondary_recurrence'],value['target']['4'])
                self.assertFalse(value['next_execution_authorized']);self.assertTrue((out/'R4T_EXPERIMENT_REPORT.md').exists())
                (out/'o4a13/records.jsonl').write_text('')
                with self.assertRaises(ValueError):analyze.recompute(out,reg)
                self.assertFalse(json.loads((out/'current_result.json').read_text())['valid'])

    def test_auxiliary_counter_corruption_and_causal_marker_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with fixture(out) as (reg,jobs,packet,ordered):
                job=jobs[10];rows=[json.loads(s) for s in (out/job['job_id']/'records.jsonl').read_text().splitlines()]
                ident=binding(packet,packet['schedule']['assignments'][10]['worker'],job)
                for change in ('transition','brier','causal','calls'):
                    bad=copy.deepcopy(rows)
                    if change=='transition':bad[0]['auxiliary']['graph']['transitions'][0]['wrong_to_correct']=1
                    if change=='brier':bad[0]['auxiliary']['q'][0]['brier']=.4
                    if change=='causal':bad[0]['auxiliary_after_state_commit']=False
                    if change=='calls':bad[0]['r4t']['counts']['network_forwards']=9
                    with self.assertRaises(ValueError):analyze.validate(bad,ordered(reg,0),job,ident)
                (out/'device1/smoke.completion.json').unlink()
                with self.assertRaises(FileNotFoundError):analyze.recompute(out,reg)

    def test_frozen_AB_preservation_matrix_and_caps(self):
        cfg=plan.science();old=json.loads((plan.ROOT/'docs/review/r4_three/input/inherited/R4D_SCIENCE_PROPOSAL.json').read_text())
        self.assertEqual(cfg['arms'][:10],old['arms'])
        for key in ('teacher','student','RP','kernel_geometry','resources','direction_A_preservation'):self.assertEqual(cfg[key],old[key])
        jobs=plan.matrix()['jobs'];self.assertEqual(len(jobs),70)
        self.assertEqual(sum(j['network_forwards'] for j in jobs),1112070)
        for w in (1,2,3):self.assertEqual([a['worker'] for a in plan.allocation(jobs,w)['assignments']],[(i%14+i//14)%w for i in range(70)])
        self.assertEqual(execution.caps(),dict(trajectory_seconds=21600,wall_seconds=259200,active_seconds=345600,bytes=6442450944))

    def test_disabled_authorization_before_model_or_device_and_review_mode(self):
        with patch('subprocess.check_output',side_effect=AssertionError('no device query')),patch('sys.argv',['neutral','--run']):
            with self.assertRaises(PermissionError):run.main()
            with self.assertRaises(PermissionError):execution.launch(dict(enabled=False),dict(registration={}),'/unused')
        auth=dict(enabled=True,approved_code_sha='b'*40,approved_science_sha256=plan.SCIENCE_SHA,approved_registration_digest=registration_digest({}),approved_stream_digest='c'*64,approved_trajectory_count=70,allowed_physical_gpu_ids=[5,6,7],max_workers=3,background_allowed=True)
        with patch.object(execution,'stream_summary',return_value=dict(stream_digest='c'*64)):
            with self.assertRaises(PermissionError):execution.authorize(auth,{},'b'*40)
            for status in ('EXTERNAL_PASS','EXPLICIT_USER_WAIVER'):
                test=dict(auth,review=dict(status=status,reference='PROCEDURAL_AUTHORIZATION_TEST_ONLY'))
                self.assertEqual(execution.authorize(test,{},'b'*40),[5,6,7])
            auth['approved_trajectory_count']=50
            with self.assertRaises(PermissionError):execution.authorize(auth,{},'b'*40)
