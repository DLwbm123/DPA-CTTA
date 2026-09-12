"""Engineering regressions: procedural assets/scalars and owned short CPU children."""
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from dpa_ctta.r1 import run,analyze
from dpa_ctta.r1.assets import checkpoint,target,AssetMismatch,verified_bytes
from dpa_ctta.r1.plan import science,matrix,allocation,binding,registration_digest,digest,SCIENCE,from_b3
from dpa_ctta.r1.recovery import TrendController
from dpa_ctta.r1.streaming_pca import StreamingSubspace,bank_audit
from dpa_ctta.r1.region_memory import Memory
from dpa_ctta.r1.evidence import write
from dpa_ctta.r1.supervise import supervise,Interrupted
from dpa_ctta.p1_analysis import evaluate


def scalar_row(entry,arm,order,visit,controller=None):
    pca=None
    if arm.startswith('C_PCA_'):
        count=1 if arm=='C_PCA_GLOBAL' else 4
        banks=[dict(bank_audit(StreamingSubspace()),last_visit=visit) for _ in range(count)]
        pca=dict(input=dict(region_token_counts=[1024,0,1024,0],sampled_region_counts=[0]*4,selected_region_counts=[0]*4,zero_vectors=[0]*4,assigned_bank_counts=[0]*count,missing_foreground=[True,True]),basis_versions_used=[None]*count,banks=banks,subloss=0.,base_loss=.1)
    return dict(entry,arm=arm,order=order,global_visit=visit,reset_count=0,total_adam_calls=visit,optimizer_steps_since_reset=visit,segment_age=visit,reset_before_current=False,controller=None if controller is None else dict(sensitivity=.1,trend=controller.observe(.1)),counts=dict(forwards=11 if arm=='C_SENS' else 8,backwards=1,base_adam=1,perturb=0,restore=0),prediction_fixed_before_label=True,metrics=evaluate(torch.zeros(1,2,3,3),torch.zeros(1,2,3,3),'fundus'),pca=pca,host_seconds=.1,pipeline_seconds=.2,peak_allocated_bytes=1,asset_io={k:dict(bytes=1,read_verify_decode_seconds=.01) for k in ('image','mask')})


@contextmanager
def complete_fixture(out,workers=3):
    """Twelve procedural identities per job, all 24 arms/orders, no real scores."""
    out=Path(out);cfg=copy.deepcopy(science());jobs=matrix()['jobs']
    canonical=[dict(group_id=str(i*3+j),sample_id=str(i*3+j),domain=d,subset=s) for i,d in enumerate(cfg['orders'][0]) for j,s in enumerate(('remaining_dev','legacy_dev','p1_extension_dev'))]
    def ordered(reg,o):return [r for d in cfg['orders'][o] for r in canonical if r['domain']==d]
    for j in jobs:j.update(records=12,forwards=12*(11 if j['arm']=='C_SENS' else 8),backwards=12,adam=12)
    cfg['formal_budget'].update(records=288,forwards=2448,backwards=288,adam=288)
    reg=dict(checkpoint=dict(bytes=7),target=canonical,fixture='PROGRAMMATIC_SCALARS_ONLY')
    packet=dict(binding=dict(run_id='a'*32,code_sha='b'*40,science_sha256=digest(SCIENCE),registration_digest=registration_digest(reg)),devices=[dict(index=i,uuid='CPU_FIXTURE_SLOT_'+str(i),model='PROGRAMMATIC_CPU') for i in range(workers)],jobs=jobs,schedule=allocation(jobs,workers),formal_budget=cfg['formal_budget'],smoke_budget=cfg['per_gpu_smoke_budget'])
    write(out/'receipt.json',packet)
    entries=[];backend=dict(device_name='PROGRAMMATIC_CPU',torch=torch.__version__)
    for i in range(workers):
        p=out/('device'+str(i));p.mkdir()
        write(p/'smoke.completion.json',dict(binding=binding(packet,i),status='MECHANICAL_SMOKE_COMPLETE',physical=dict(forwards=118,backwards=14,base_adam=14,perturb=0,restore=0),backend=backend,checkpoint_io=dict(bytes=7)))
        entries.append(dict(binding=binding(packet,i),phase='smoke',key=p.name,pid=100+i,pgid=100+i,status='EXITED',exit_code=0))
    for job,a in zip(jobs,packet['schedule']['assignments']):
        p=out/job['job_id'];p.mkdir();identity=binding(packet,a['worker'],job)
        controller=TrendController() if job['arm']=='C_SENS' else None
        rows=[dict(scalar_row(e,job['arm'],job['order'],i,controller),binding=identity) for i,e in enumerate(ordered(reg,job['order']),1)]
        (p/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
        write(p/'completion.json',dict(binding=identity,status='TRAJECTORY_COMPLETE',records=12,physical=counts,backend=backend,checkpoint_io=dict(bytes=7)))
        pid=100+len(entries);entries.append(dict(binding=identity,phase='formal',key=p.name,pid=pid,pgid=pid,status='EXITED',exit_code=0))
    write(out/'matrix.processes.json',dict(binding=binding(packet),status='COMPUTE_COMPLETE',exit_codes=[0]*len(entries),processes=entries,active_seconds=1.,wall_seconds=1.))
    write(out/'processes.started.json',dict(binding=binding(packet),processes=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]))
    with patch('dpa_ctta.r1.analyze.stream',side_effect=ordered),patch('dpa_ctta.r1.analyze.science',return_value=cfg),patch('dpa_ctta.r1.analyze.matrix',return_value=dict(jobs=jobs)):
        yield reg,packet,ordered


class AssetChecks(unittest.TestCase):
    def test_checkpoint_digest_precedes_smoke_and_formal_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'weights.bin';torch.save({'weight':torch.arange(3)},cp);stat=cp.stat()
            reg=dict(checkpoint=dict(path=str(cp),sha256='0'*64,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
            out=root/'out';out.mkdir();job=matrix()['jobs'][0]
            packet=dict(binding=dict(run_id='a'*32,code_sha='b'*40,science_sha256=digest(SCIENCE),registration_digest=registration_digest(reg)),devices=[dict(uuid='CPU_FIXTURE_ONLY')],jobs=[job],schedule=dict(assignments=[dict(job_id=job['job_id'],worker=0)]))
            with patch.object(run,'context',return_value=(packet,reg,0,out)),patch('dpa_ctta.source_pilot_release.environment',side_effect=AssertionError('GPU environment')),patch('torch.load',side_effect=AssertionError('deserialization')) as loader:
                with self.assertRaises(AssetMismatch):run.worker()
                loader.assert_not_called()
                # The failed smoke remains evidence; a separate procedural proof only tests the formal loader.
                (out/'device0/smoke.failure.json').unlink()
                write(out/'device0/smoke.completion.json',dict(binding=binding(packet,0),status='MECHANICAL_SMOKE_COMPLETE',physical=dict(forwards=118,backwards=14,base_adam=14,perturb=0,restore=0)))
                with patch.dict(os.environ,RUN_JOB=job['job_id']):
                    with self.assertRaises(AssetMismatch):run.formal_worker()
                loader.assert_not_called()
            self.assertEqual(cp.stat().st_size,stat.st_size);self.assertEqual(cp.stat().st_mtime_ns,stat.st_mtime_ns)
            reg['checkpoint']['sha256']=digest(cp);state,info=checkpoint(reg)
            self.assertTrue(torch.equal(state['weight'],torch.arange(3)));self.assertEqual(info['bytes'],stat.st_size)

    def test_RGB_mask_same_verified_bytes_and_prediction_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rgb=root/'input.png';mask=root/'label.png'
            Image.new('RGB',(9,7),(12,34,56)).save(rgb);Image.new('L',(9,7),255).save(mask)
            row=dict(image_path=str(rgb),mask_path=str(mask),image_size=[9,7],image_sha256=digest(rgb),mask_sha256=digest(mask))
            from dpa_ctta.source_io import read_pixels,read_mask
            self.assertTrue(torch.equal(target(row,'image')[0],read_pixels(rgb,'fundus',[9,7])))
            self.assertTrue(torch.equal(target(row,'mask')[0],read_mask(mask,'fundus',[9,7])))
            for kind in ('image','mask'):
                with self.assertRaises(AssetMismatch):target(dict(row,**{kind+'_sha256':'0'*64}),kind)
            events=[]
            def verified(path,expected):
                if path==str(mask):self.assertIn('host returned',events)
                events.append('mask' if path==str(mask) else 'RGB');return verified_bytes(path,expected)
            class ImageOnly:
                def step(self,x):events.append('host returned');return torch.zeros(1,2,512,512),{}
            with patch('dpa_ctta.r1.assets.verified_bytes',side_effect=verified),patch('dpa_ctta.p1_run.sync'):
                result=run.current(ImageOnly(),row)
            self.assertEqual(events,['RGB','host returned','mask']);self.assertEqual(set(result['asset_io']),{'image','mask'})
            events.clear()
            with patch('dpa_ctta.r1.assets.verified_bytes',side_effect=verified),patch('dpa_ctta.p1_run.sync'),patch.object(ImageOnly,'step',side_effect=RuntimeError('programmatic host failure')):
                with self.assertRaises(RuntimeError):run.current(ImageOnly(),row)
            self.assertEqual(events,['RGB'])

    def test_from_metadata_preserves_existing_hashes_without_readers(self):
        from dpa_ctta.r1.plan import COUNTS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            subsets=['remaining_dev']*1695+['legacy_dev']*128+['p1_extension_dev']*128
            for domain,count in COUNTS.items():
                for i in range(count):
                    n=len(rows);rows.append(dict(group_id=str(n),sample_id=str(n),domain=domain,subset=subsets[n],manifest_index=i,image_size=[9,7],image_path='UNREAD_RGB',mask_path='UNREAD_MASK',image_sha256='1'*64,mask_sha256='2'*64))
            source=root/'registration.json';source.write_text(json.dumps(dict(tasks=dict(fundus=dict(checkpoint=dict(path='UNREAD_WEIGHT',sha256='3'*64,bytes=1,mtime_ns=1),target=rows)),p2_directory=str(root/'absent'))))
            with patch('dpa_ctta.source_io.read_pixels',side_effect=AssertionError('real reader')),patch('dpa_ctta.source_io.read_mask',side_effect=AssertionError('real mask')),patch('dpa_ctta.source_io.source_proxy',side_effect=AssertionError('proxy')):
                reg=from_b3(source)
            self.assertEqual([(r['image_sha256'],r['mask_sha256']) for r in reg['target']],[('1'*64,'2'*64)]*1951)
            self.assertEqual([r['group_id'] for r in reg['target']],[r['group_id'] for r in rows]);self.assertNotIn('proxy',reg)


class EvidenceChecks(unittest.TestCase):
    def test_first_visit_reset_and_controller_log_contradictions(self):
        e=dict(group_id='toy',sample_id='toy',domain='D',subset='remaining_dev')
        row=scalar_row(e,'C_SENS',0,1,TrendController());analyze.validate([row],[e],'C_SENS',0)
        bad=dict(row,reset_before_current=True,reset_count=1)
        with self.assertRaisesRegex(ValueError,'causality'):analyze.validate([bad],[e],'C_SENS',0)
        bad=copy.deepcopy(row);bad['controller']['trend']['trigger']=True
        with self.assertRaisesRegex(ValueError,'replay'):analyze.validate([bad],[e],'C_SENS',0)
        rows=[];c=TrendController();age=0;resets=0
        for i in range(1,54):
            r=scalar_row(dict(e,group_id=str(i),sample_id=str(i)),'C_SENS',0,i)
            s=0. if i<50 else 1.;t=c.observe(s)
            if t['trigger']:resets+=1;age=0
            age+=1;r.update(controller=dict(sensitivity=s,trend=t),reset_before_current=t['trigger'],reset_count=resets,segment_age=age,optimizer_steps_since_reset=age);rows.append(r)
        analyze.validate(rows,rows,'C_SENS',0)
        self.assertEqual(rows[49]['reset_count'],1)
        for field in ('age','ema','best','resets'):
            bad=copy.deepcopy(rows);bad[51]['controller']['trend'][field]+=1
            with self.assertRaises(ValueError):analyze.validate(bad,rows,'C_SENS',0)

    def test_PCA_live_scalar_refresh_and_invalid_banks_quotas(self):
        for mode in ('GLOBAL','REGION','SHUFFLED'):
            memory=Memory(mode);q=torch.zeros(1024,2);q[:512]=1;views=q.expand(6,-1,-1);previous=None
            for i in range(1,34):
                snaps=memory.snapshots(i);vectors,ids,inputs=memory.prepare(torch.randn(1024,32),q,views,i);memory.merge(vectors,i)
                evidence=dict(input=inputs,basis_versions_used=[s[2] if s is not None else None for s in snaps],banks=memory.audit(),subloss=.1 if any(s is not None for s in snaps) else 0.)
                old=copy.deepcopy(previous);previous=analyze.pca_scalars(evidence,previous,'C_PCA_'+mode,i)
                if i==17:
                    changes=[('banks',[]),('basis_versions_used',[])]
                    for key,value in changes:
                        bad=copy.deepcopy(evidence);bad[key]=value
                        with self.assertRaises(ValueError):analyze.pca_scalars(bad,old,'C_PCA_'+mode,i)
                    for key,value in [('n',-1),('contributing_images',99),('version',i),('rank',9),('eigh_calls',999)]:
                        bad=copy.deepcopy(evidence);bad['banks'][0][key]=value
                        with self.assertRaises(ValueError):analyze.pca_scalars(bad,old,'C_PCA_'+mode,i)
                    bad=copy.deepcopy(evidence);bad['input']['assigned_bank_counts'][0]=-1
                    with self.assertRaises(ValueError):analyze.pca_scalars(bad,old,'C_PCA_'+mode,i)

    def test_completion_and_run_smoke_process_contradictions(self):
        mutations=[('o0a0/completion.json',lambda x:x.update(records=11)),('o0a0/completion.json',lambda x:x['physical'].update(forwards=95)),('o0a0/completion.json',lambda x:x['binding'].update(order=1)),('device0/smoke.completion.json',lambda x:x['physical'].update(forwards=119)),('device0/smoke.completion.json',lambda x:x['binding'].update(code_sha='c'*40)),('matrix.processes.json',lambda x:x['exit_codes'].__setitem__(0,1)),('receipt.json',lambda x:x['binding'].update(registration_digest='0'*64))]
        for name,change in mutations:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                with complete_fixture(out) as (reg,packet,ordered):
                    path=out/name;value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):analyze.recompute(out,reg)
                    self.assertFalse(json.loads((out/'current_result.json').read_text())['valid']);self.assertFalse((out/'public_aggregate.json').exists())

    def test_failure_with_success_and_stale_complete_invalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with complete_fixture(out) as (reg,packet,ordered):
                result=analyze.recompute(out,reg);self.assertEqual(result['physical']['records'],288)
                saved=json.loads((out/'current_result.json').read_text())['result_directory']
                self.assertTrue((out/'public_aggregate.json').is_file())
                (out/'o0a0/failure.json').write_text('{"status":"INCOMPLETE"}')
                with self.assertRaises(ValueError):analyze.recompute(out,reg)
                self.assertFalse((out/'public_aggregate.json').exists());self.assertFalse((out/'R1_EXPERIMENT_REPORT.md').exists())
                self.assertTrue((out/saved/'public_aggregate.json').exists());self.assertFalse(json.loads((out/'current_result.json').read_text())['valid'])
                self.assertEqual(out.stat().st_mode&0o777,0o700)

    def test_history_method_order_receipt_and_C0_canonical_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new';out.mkdir()
            with complete_fixture(out) as (reg,packet,ordered):
                reference=ordered(reg,1);rows=[scalar_row(e,'C',1,i) for i,e in enumerate(reference,1)];all_rows={(1,'C'):rows}
                historic=Path(tmp)/'old';historic.mkdir();old_target=ordered(reg,0)
                old_reg=historic/'registration.json';old_reg.write_text(json.dumps(dict(tasks=dict(fundus=dict(target=old_target)))))
                receipt=historic/'receipt.run.json';value=dict(commit='c'*40,config_sha256='d'*64,registration_sha256=digest(old_reg),gpu_uuid='CPU_FIXTURE_ONLY',output_directory=str(historic));receipt.write_text(json.dumps(value))
                entry=dict(receipt_path=str(receipt),receipt_sha256=digest(receipt),registration_path=str(old_reg),receipt_binding={k:value[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')})
                path=historic/'fundus_1_A.jsonl';old=[dict(r,task='fundus',arm='A',order=1,visit=i) for i,r in enumerate(rows,1)]
                path.write_text('\n'.join(json.dumps(r) for r in old));(historic/'run.completion.json').write_text('{"status":"P2_RUN_COMPLETE","exit_code":0}')
                hreg=dict(historical_scalars={'1':{'A':str(path)}},historical_bindings={'1':{'A':entry}})
                self.assertEqual(analyze.secondary_controls(all_rows,hreg)['1']['A']['status'],'AVAILABLE')
                for field,value in [('arm','NOT_A'),('order',0),('visit',77)]:
                    bad=copy.deepcopy(old);bad[0][field]=value;path.write_text('\n'.join(json.dumps(r) for r in bad))
                    result=analyze.secondary_controls(all_rows,hreg)['1']['A'];self.assertEqual(result['status'],'UNVERIFIED');self.assertIn('method/order/visit',result['error'])
                path=historic/'fundus_canonical_C0.jsonl'
                canonical=[dict(scalar_row(e,'C',0,i),task='fundus',arm='C0',order=None,visit=i,prediction_origin='canonical_stateless',stateless_checked=True,counts=dict(forwards=1,backwards=0,base_adam=0,perturb=0,restore=0)) for i,e in enumerate(old_target,1)]
                path.write_text('\n'.join(json.dumps(r) for r in canonical));(historic/'run.completion.json').write_text('{"status":"B4_RUN_COMPLETE","exit_code":0}')
                hreg=dict(historical_scalars={'1':{'C0':str(path)}},historical_bindings={'1':{'C0':entry}})
                self.assertNotEqual([r['group_id'] for r in canonical],[r['group_id'] for r in rows]);self.assertEqual(analyze.secondary_controls(all_rows,hreg)['1']['C0']['status'],'AVAILABLE')
                canonical[0]['metrics']=canonical[0]['metrics'][:1];path.write_text('\n'.join(json.dumps(r) for r in canonical))
                self.assertEqual(analyze.secondary_controls(all_rows,hreg)['1']['C0']['status'],'UNVERIFIED')
                hreg['historical_bindings']['1']['C0']=dict(entry,receipt_sha256='0'*64)
                self.assertEqual(analyze.secondary_controls(all_rows,hreg)['1']['C0']['status'],'UNVERIFIED')


CHILD='''import json,os,time,sys,subprocess,signal
from pathlib import Path
s=json.loads(os.environ['SPEC']);p=Path(os.environ['OUT'])/s['key'];p.mkdir(exist_ok=True)
(p/'records.jsonl').write_text('{"procedural_prefix":1}\\n')
if s.get('grandchild'):
    c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    (p/'descendant.pid').write_text(str(c.pid))
if s.get('signal'):
    time.sleep(.07);os.kill(os.getppid(),s['signal'])
if s.get('scope'):(p/'failure.json').write_text(json.dumps({'scope':s['scope'],'status':'INCOMPLETE'}))
time.sleep(s.get('delay',.01));sys.exit(s.get('exit',0))
'''


class ProcessChecks(unittest.TestCase):
    def packet(self,workers=2):
        jobs=matrix()['jobs'][:6]
        return dict(binding=dict(run_id='a'*32,code_sha='b'*40,science_sha256=digest(SCIENCE),registration_digest='c'*64),devices=[dict(uuid='CPU_FIXTURE_SLOT_'+str(i)) for i in range(workers)],jobs=jobs,schedule=allocation(jobs,workers))
    def spawner(self,out,options):
        created=[]
        def start(spec,log):
            extra=options(spec,len(created))
            if extra.get('spawn_error'):raise OSError('procedural spawn failure')
            env=os.environ.copy();env.update(OUT=str(out),SPEC=json.dumps(dict(spec,**extra)),CUDA_VISIBLE_DEVICES='')
            p=subprocess.Popen([sys.executable,'-c',CHILD],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            created.append((copy.deepcopy(spec),p));return p
        return start,created
    def caps(self,**kw):return dict(dict(trajectory_seconds=4.,wall_seconds=10.,active_seconds=20.,bytes=1024**2),**kw)
    def test_rotation_one_two_three_slots_has_no_arm_device_lock(self):
        jobs=matrix()['jobs']
        for k in (1,2,3):
            plan=allocation(jobs,k);a=plan['assignments'];self.assertEqual(len({v['job_id'] for v in a}),24)
            self.assertEqual(sum(v['forwards'] for v in plan['loads']),398004)
            for arm in science()['arms']:
                slots={x['worker'] for j,x in zip(jobs,a) if j['arm']==arm};self.assertEqual(len(slots),k)
    def test_finite_success_records_all_created_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);start,created=self.spawner(out,lambda s,n:{})
            result=supervise(out,self.packet(),start,self.caps(),.01)
            self.assertEqual(result['status'],'COMPUTE_COMPLETE');self.assertEqual(len(created),8);self.assertTrue(all(p.poll()==0 for s,p in created))
    def test_later_worker_failure_stops_dispatch_allows_active_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            start,created=self.spawner(out,lambda s,n:{} if s['phase']=='smoke' else dict(delay=.3 if s['worker']==0 else .02,exit=0 if s['worker']==0 else 1))
            with self.assertRaises(RuntimeError):supervise(out,self.packet(),start,self.caps(),.01)
            formal=[(s,p) for s,p in created if s['phase']=='formal'];self.assertEqual(len(formal),2);self.assertEqual(formal[0][1].returncode,0)
            self.assertTrue((out/'dispatch.stopped.json').exists());self.assertTrue((out/formal[0][0]['key']/'records.jsonl').read_text())
    def test_trajectory_timeout_prefix_and_unrelated_process_survives(self):
        outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);start,created=self.spawner(out,lambda s,n:{} if s['phase']=='smoke' else dict(delay=5,grandchild=True))
                with self.assertRaises(RuntimeError):supervise(out,self.packet(1),start,self.caps(trajectory_seconds=.15),.01)
                self.assertIsNone(outsider.poll());self.assertTrue(all(p.poll() is not None for s,p in created))
                self.assertEqual(json.loads((out/'o0a0/supervisor.failure.json').read_text())['status'],'TIMEOUT');self.assertTrue((out/'o0a0/records.jsonl').read_text())
                pid=int((out/'o0a0/descendant.pid').read_text())
                state=subprocess.run(['ps','-p',str(pid),'-o','stat='],text=True,capture_output=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'),state)
        finally:os.killpg(outsider.pid,signal.SIGTERM);outsider.wait(timeout=2)
    def test_wall_and_active_caps_reach_stuck_smoke(self):
        for key in ('wall_seconds','active_seconds'):
            with self.subTest(cap=key),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);start,created=self.spawner(out,lambda s,n:dict(delay=5))
                with self.assertRaises(RuntimeError):supervise(out,self.packet(1),start,self.caps(**{key:.12}),.01)
                self.assertTrue(all(p.poll() is not None for s,p in created));self.assertFalse(any(s['phase']=='formal' for s,p in created))
    def test_partial_spawn_failure_cleans_created_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);start,created=self.spawner(out,lambda s,n:dict(spawn_error=n==1,delay=5))
            with self.assertRaises(OSError):supervise(out,self.packet(),start,self.caps(),.01)
            self.assertEqual(len(created),1);self.assertIsNotNone(created[0][1].poll())
    def test_parent_signal_during_spawn_is_deferred_until_child_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);start,created=self.spawner(out,lambda s,n:dict(delay=5))
            def interrupted_start(spec,log):
                process=start(spec,log);os.kill(os.getpid(),signal.SIGTERM);return process
            with self.assertRaises(Interrupted):supervise(out,self.packet(),interrupted_start,self.caps(),.01)
            self.assertEqual(len(created),1);self.assertIsNotNone(created[0][1].poll())
    def test_parent_SIGINT_SIGTERM_clean_up_own_groups(self):
        for sig in (signal.SIGINT,signal.SIGTERM):
            with self.subTest(signal=sig),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);start,created=self.spawner(out,lambda s,n:dict(delay=5,signal=sig,grandchild=True))
                with self.assertRaises(Interrupted):supervise(out,self.packet(1),start,self.caps(),.01)
                self.assertTrue(all(p.poll() is not None for s,p in created));self.assertEqual(json.loads((out/'matrix.processes.json').read_text())['status'],'INCOMPLETE')
    def test_shared_failure_cancels_only_this_batch_and_output_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);start,created=self.spawner(out,lambda s,n:{} if s['phase']=='smoke' else dict(delay=5 if s['worker']==0 else .02,exit=0 if s['worker']==0 else 1,scope='shared_assets' if s['worker']==1 else None))
            with self.assertRaises(RuntimeError):supervise(out,self.packet(),start,self.caps(),.01)
            formal=[p for s,p in created if s['phase']=='formal'];self.assertEqual(len(formal),2);self.assertNotEqual(formal[0].returncode,0)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'programmatic_large.bin').write_bytes(b'x'*4096);start,created=self.spawner(out,lambda s,n:{})
            with self.assertRaises(RuntimeError):supervise(out,self.packet(1),start,self.caps(bytes=1024),.01)
            self.assertEqual(created,[])
