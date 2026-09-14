"""Synthetic scalar fixtures exercise every arm/stream; no model or real assets."""
import copy,json,os,tempfile,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from dpa_ctta.r3 import analyze,plan,execution
from dpa_ctta.r1.evidence import write
from dpa_ctta.r1.plan import binding,registration_digest
from test_r1_fixes import scalar_row


@contextmanager
def fixture(out):
    cfg=copy.deepcopy(plan.science());jobs=plan.matrix()['jobs']
    rows=[dict(group_id=str(i*3+j),sample_id=str(i*3+j),domain=d,subset=s) for i,d in enumerate(cfg['orders'][0]) for j,s in enumerate(('remaining_dev','legacy_dev','p1_extension_dev'))]
    reg=dict(checkpoint=dict(bytes=7),target=rows,fixture='PROGRAMMATIC_ONLY')
    def ordered(reg,index):
        ds=cfg['orders'][index] if index<4 else cfg['orders'][0][1:]+cfg['orders'][0][:1]
        return [r for d in ds for r in rows if r['domain']==d]
    for j in jobs:
        j.update(records=12,network_forwards=12*(9 if j['arm'].startswith(('T_','S_')) else 8),loss_backward_calls=12,adam_calls=12,jacobian_vjp_upper=96 if j['arm'].startswith('U_') else 0)
    for key,col in [('scoring_records','records'),('network_forwards','network_forwards'),('loss_backward_calls','loss_backward_calls'),('adam_calls','adam_calls'),('jacobian_vjp_upper','jacobian_vjp_upper')]:cfg['formal_budget'][key]=sum(j[col] for j in jobs)
    identity=dict(run_id='a'*32,code_sha='b'*40,science_sha256=plan.SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest='c'*64)
    packet=dict(binding=identity,jobs=jobs,devices=[dict(index=i,uuid='CPU_SLOT_'+str(i),model='PROCEDURAL') for i in range(2)],schedule=plan.allocation(jobs,2))
    write(out/'receipt.json',packet);entries=[];backend=dict(seed=20260907,deterministic_algorithms=False,warn_only=False,cublas_workspace_config=None)
    for i in range(2):
        p=out/('device'+str(i));p.mkdir();ident=binding(packet,i)
        write(p/'smoke.completion.json',dict(binding=ident,status='MECHANICAL_SMOKE_COMPLETE',physical=dict(network_forwards=316,loss_backward_calls=38,jacobian_vjp_calls=0,adam_calls=38,actual_parameter_replacements=0),backend=dict(backend,cublas_workspace_config=':4096:8'),paired_comparison_backend=dict(deterministic_algorithms=True,warn_only=False,cublas_workspace_config=':4096:8'),checkpoint_io=dict(bytes=7)))
        entries.append(dict(binding=ident,phase='smoke',key=p.name,pid=100+i,pgid=100+i,status='EXITED',exit_code=0))
    for j,assignment in zip(jobs,packet['schedule']['assignments']):
        p=out/j['job_id'];p.mkdir();ident=binding(packet,assignment['worker'],j);records=[]
        for t,e in enumerate(ordered(reg,j['order']),1):
            r=scalar_row(e,'C' if j['arm']=='C' else 'C_PCA_REGION',j['order'],t)
            r.update(arm=j['arm'],binding=ident)
            a=dict(arm=j['arm'],global_visit=t,counts=dict(network_forwards=9 if j['arm'].startswith(('S_','T_')) else 8,loss_backward_calls=1,jacobian_vjp_calls=0,adam_calls=1,actual_parameter_replacements=0))
            if j['arm'] not in ('C','RP'):
                a.update(memory_input=r['pca']['input'],banks=r['pca']['banks'],basis_versions_used=[None]*4,density_versions_used=[None]*4)
                for b in a['banks']:b['frame_version']=t if j['arm'].startswith('M_') else 0
            if j['arm'].startswith('S_'):a['context']=dict(slot=0,created=t==1,active_slots=1,slot_assigned_counts=[t],slot_updates=[t],optimizer_steps=[t])
            r['r3']=a;records.append(r)
        (p/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        counts={k:sum(r['r3']['counts'][k] for r in records) for k in analyze.COUNT_KEYS}
        write(p/'completion.json',dict(binding=ident,status='TRAJECTORY_COMPLETE',records=12,physical=counts,backend=backend,checkpoint_io=dict(bytes=7),seconds=1.))
        pid=100+len(entries);entries.append(dict(binding=ident,phase='formal',key=p.name,pid=pid,pgid=pid,status='EXITED',exit_code=0))
    write(out/'matrix.processes.json',dict(binding=identity,status='COMPUTE_COMPLETE',exit_codes=[0]*len(entries),processes=entries,active_seconds=1.,wall_seconds=1.,unstarted_jobs=[]))
    write(out/'processes.started.json',dict(binding=identity,processes=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]))
    with patch.object(analyze,'science',return_value=cfg),patch.object(analyze,'matrix',return_value=dict(jobs=jobs)),patch.object(analyze,'stream',side_effect=ordered),patch.object(analyze,'stream_summary',return_value=dict(stream_digest='c'*64)):
        yield reg,jobs,packet,ordered


class ExecutionTests(unittest.TestCase):
    def test_phase_environment_matrix_preserves_parent_and_identity(self):
        for value in (None,'invalid',':16:8',':4096:8'):
            for phase in ('smoke','formal'):
                with self.subTest(parent=value,phase=phase):
                    parent=dict(CUDA_VISIBLE_DEVICES='CPU_SLOT',RUN_MODE=phase,RUN_WORKER='1',RUN_JOB='o4a16',RUN_PACKET='/procedural/packet',RUN_FILE='/procedural/entry',PYTHONPATH='/procedural/src',EXTRA='preserved')
                    if value is not None:parent['CUBLAS_WORKSPACE_CONFIG']=value
                    saved=parent.copy();child=execution.worker_environment(parent,phase)
                    self.assertEqual(parent,saved);self.assertIsNot(child,parent)
                    self.assertEqual({k:v for k,v in child.items() if k!='CUBLAS_WORKSPACE_CONFIG'},{k:v for k,v in parent.items() if k!='CUBLAS_WORKSPACE_CONFIG'})
                    if phase=='smoke':self.assertEqual(child['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
                    else:self.assertNotIn('CUBLAS_WORKSPACE_CONFIG',child)
                    execution.check_worker_environment(phase,child)
                    print('PHASE_ENV '+json.dumps(dict(parent=value,phase=phase,child=child.get('CUBLAS_WORKSPACE_CONFIG'),parent_unchanged=parent==saved)))

    def test_unknown_phase_rejected(self):
        with self.assertRaisesRegex(ValueError,'unknown execution phase'):execution.worker_environment({},'other')
        with self.assertRaisesRegex(ValueError,'unknown execution phase'):execution.check_worker_environment('other',{})

    def test_direct_worker_mismatch_rejected_before_device_or_assets(self):
        with patch.object(execution,'context',return_value=({}, {},0,Path('/unused'))),patch.object(execution,'checkpoint',side_effect=AssertionError('no assets')),patch('dpa_ctta.source_pilot_release.environment',side_effect=AssertionError('no device query')),patch('torch.cuda._lazy_init',side_effect=AssertionError('no CUDA init')):
            for phase,values in (('smoke',(None,'invalid',':16:8','')),('formal',('invalid',':16:8',':4096:8',''))):
                for value in values:
                    with self.subTest(phase=phase,value=value),patch.dict(os.environ,RUN_MODE=phase):
                        os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
                        if value is not None:os.environ['CUBLAS_WORKSPACE_CONFIG']=value
                        with self.assertRaisesRegex(PermissionError,'CUBLAS_WORKSPACE_CONFIG mismatch'):execution.worker()

    def test_direct_worker_disabled_authorization_precedes_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet=Path(tmp)/'packet.json';packet.write_text(json.dumps(dict(authorization=dict(enabled=False),assets=dict(registration={}))))
            with patch.dict(os.environ,RUN_PACKET=str(packet),RUN_MODE='smoke',CUBLAS_WORKSPACE_CONFIG='invalid'),patch.object(execution,'check_worker_environment',side_effect=AssertionError('authorization must run first')),patch('subprocess.check_output',side_effect=AssertionError('no device query')):
                with self.assertRaisesRegex(PermissionError,'execution disabled'):execution.worker()

    def test_formal_worker_records_actual_policy_without_changing_it(self):
        import torch
        original=execution.backend_policy()
        try:
            torch.use_deterministic_algorithms(True,warn_only=True)
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,RUN_MODE='formal',RUN_JOB='o0a0'):
                os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
                out=Path(tmp)
                with fixture(out) as (reg,jobs,packet,ordered),patch.object(execution,'context',return_value=(packet,reg,0,out)),patch.object(execution,'checkpoint',return_value=({},dict(bytes=7))),patch('dpa_ctta.source_pilot_release.environment',return_value=dict(seed=20260907)),patch.object(execution,'trajectory') as trajectory,patch('torch.cuda._lazy_init',side_effect=AssertionError('no CUDA init')):
                    before=execution.backend_policy();execution.worker()
                    self.assertEqual(trajectory.call_args.args[5],dict(seed=20260907,**before))
                    self.assertEqual(execution.backend_policy(),before)
                    self.assertEqual(before,dict(deterministic_algorithms=True,warn_only=True,cublas_workspace_config=None))
        finally:torch.use_deterministic_algorithms(original['deterministic_algorithms'],warn_only=original['warn_only'])

    def test_smoke_failure_restores_both_flags_and_workspace(self):
        import torch
        original=execution.backend_policy()
        try:
            torch.use_deterministic_algorithms(False,warn_only=True)
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'):
                before=execution.backend_policy()
                def fail_before_model(seed):
                    self.assertEqual(execution.backend_policy(),dict(deterministic_algorithms=True,warn_only=False,cublas_workspace_config=':4096:8'))
                    raise RuntimeError('procedural failure before model')
                with patch('dpa_ctta.source_pilot.seed_all',side_effect=fail_before_model):
                    with self.assertRaisesRegex(RuntimeError,'procedural failure before model'):execution.smoke({},'cpu',Path(tmp),{},before,{})
                self.assertEqual(execution.backend_policy(),before)
                self.assertFalse((Path(tmp)/'smoke.completion.json').exists())
        finally:torch.use_deterministic_algorithms(original['deterministic_algorithms'],warn_only=original['warn_only'])

    def test_85_fixture_closeout_separate_secondary_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with fixture(out) as (reg,jobs,packet,ordered):
                result=analyze.recompute(out,reg)
                self.assertEqual(result['physical']['records'],1020);self.assertEqual(len(result['mechanism']),85)
                self.assertEqual(set(result['target']),{'0','1','2','3','4'})
                self.assertEqual(result['secondary_recurrence'],result['target']['4']);self.assertFalse(result['next_execution_authorized'])
                self.assertTrue((out/'R3_EXPERIMENT_REPORT.md').exists())
                (out/'o4a16/records.jsonl').write_text('')
                with self.assertRaises(ValueError):analyze.recompute(out,reg)
                self.assertFalse(json.loads((out/'current_result.json').read_text())['valid']);self.assertFalse((out/'R3_EXPERIMENT_REPORT.md').exists())

    def test_corrupt_counts_and_missing_evidence_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with fixture(out) as (reg,jobs,packet,ordered):
                j=jobs[5];p=out/j['job_id']/'records.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
                rows[0]['r3']['counts']['jacobian_vjp_calls']=9
                with self.assertRaises(ValueError):analyze.validate(rows,ordered(reg,0),j,binding(packet,1,j))
                (out/'device0/smoke.completion.json').unlink()
                with self.assertRaises(FileNotFoundError):analyze.recompute(out,reg)
                self.assertFalse(json.loads((out/'current_result.json').read_text())['valid'])

    def test_authorization_rejects_before_model_or_device(self):
        with patch('subprocess.check_output',side_effect=AssertionError('no device query')):
            with self.assertRaises(PermissionError):execution.launch(dict(enabled=False),dict(registration={}),'/unused')
        self.assertEqual(execution.caps(),dict(trajectory_seconds=21600,wall_seconds=345600,active_seconds=576000,bytes=6*1024**3))
