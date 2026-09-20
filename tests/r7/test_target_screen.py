"""Only temporary procedural files/tensors. No real target registrations used."""
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from common import segmenter, method, pixels
from dpa_ctta.r7_shared.context import provenance, json_digest
from dpa_ctta.r7_shared.preparation import prepared_artifact
from dpa_ctta.r7_shared.host import OnlineHost
from dpa_ctta.r7_target_screen import runner as r


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='screen-procedural-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in ('target', 'artifacts', 'storage'): (self.root/name).mkdir()
        self.cfg = dict(device='cpu', physical_GPU_ids=None, runtime=r.runtime(),
            dtype_policy='R7_CPU_FP32_MODEL_FP64_LATENT_V1', workers=3, threads=2,
            retry=False, resume=False, wall_seconds=100, output_bytes=24*1024**2+2*r.TERMINAL_RESERVE,
            job_wall_seconds=20, job_output_bytes=1024**2, max_asset_bytes=32*1024**2,
            max_image_pixels=1024**2, storage_root=str(self.root/'storage'),
            resource_authorization=dict(scope=r.SCOPE, receipt_id='procedural'))

    def put(self, name, raw):
        p = self.root/name; p.write_bytes(raw)
        return dict(path=str(p), sha256=r.digest(raw), bytes=len(raw))

    def fixture(self):
        img = io.BytesIO(); Image.new('RGB', (8, 8), (80, 100, 120)).save(img, format='PNG')
        mask = io.BytesIO(); Image.new('L', (8, 8), 255).save(mask, format='PNG')
        image = self.put('target/image.png', img.getvalue()); label = self.put('target/mask.png', mask.getvalue())
        cp = self.put('checkpoint.pt', b'procedural checkpoint metadata only')
        row = dict(image_path=image['path'], mask_path=label['path'], image_sha256=image['sha256'], mask_sha256=label['sha256'], image_size=[8,8], domain='REFUGE', subset='remaining_dev', group_id='procedural')
        reg = dict(checkpoint=cp, target=[row]); metadata = self.put('registration.json', json.dumps(reg).encode())
        s = segmenter(); self.addCleanup(s.close)
        src = provenance(); src.update(source_binding_status='BOUND', checkpoint_file_sha256=cp['sha256'], source_manifest_sha256='a'*64, source_split_sha256='b'*64)
        inventory = {}
        for name in sorted(r.METHODS):
            m = method(name[0], name.endswith('STATIC')); m.freeze()
            artifact = prepared_artifact(s, m, src)
            data = io.BytesIO(); torch.save(dict(schema=artifact['schema'], weights=artifact['weights'], method_digest=artifact['method_digest']), data)
            asset = self.put('artifacts/'+name+'.pt', data.getvalue())
            context = self.put('artifacts/'+name+'.context.json', json.dumps(artifact['binding']).encode())
            inventory[name] = dict(file=name+'.pt', bytes=asset['bytes'], training_asset_file_sha256=asset['sha256'], context_file_sha256=context['sha256'], context_sha256=artifact['binding']['sha256'])
        inv = self.put('inventory.json', json.dumps(inventory).encode())
        binding = dict(code_sha='f'*40, science_sha256=r.SCIENCE, jobs=r.matrix(r.SCOPE), seed=r.SEED,
            resources=self.cfg, registration=metadata, inventory=inv, registration_digest='c'*64, recurrence_digest='d'*64,
            source_release=dict(status='REVIEWED', artifact_identity=inv['sha256'], external_review_reference='procedural-only', trusted_loader_verified=True, manifest_sha256='a'*64, split_sha256='b'*64),
            checkpoint=cp, target_root=str(self.root/'target'), artifact_root=str(self.root/'artifacts'), output_dir=str(self.root/'storage/run'))
        receipt = dict(schema='R7_TARGET_SCREEN_AUTH_V1', scope=r.SCOPE, enabled=True,
            user_authorization=dict(granted=True, scope=r.SCOPE, receipt_id='procedural'), binding=binding)
        self.rebind(receipt)
        return receipt, reg, s, inventory

    def rebind(self, receipt):
        receipt['execution_layer_review'] = dict(status='PASS', scope=r.SCOPE, reference='TEST_ONLY_NOT_AUTHORITY', binding_sha256=json_digest(receipt['binding']))

    def approve_fixture(self, receipt):
        with patch.object(r, 'code_identity', return_value='f'*40), patch.object(r, 'dry_run', return_value=dict(recurrence=dict(registration_digest='c'*64, stream_digest='d'*64))):
            return r.preflight(receipt)

    def test_disabled_all_scopes_before_reads(self):
        with patch.object(r, 'verified', side_effect=AssertionError('no reads')), patch.object(r, 'code_identity', side_effect=AssertionError('no probes')):
            for scope in ('TARGET_SCREEN','SOURCE_PREP','TARGET_MECHANISM','TARGET_EXTENSION','GPU_SMOKE'):
                with self.assertRaises(PermissionError): r.preflight(dict(scope=scope, enabled=False))
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(PermissionError): r.main()

    def test_exact_authority_code_resource_and_matrix_binding(self):
        receipt, *_ = self.fixture(); self.approve_fixture(receipt)
        mutations = [lambda x:x.update(scope='TARGET_MECHANISM'),
            lambda x:x['execution_layer_review'].update(status='PENDING'),
            lambda x:x['user_authorization'].update(granted=False),
            lambda x:x['binding'].update(code_sha='0'*40),
            lambda x:x['binding'].update(seed=0),
            lambda x:x['binding']['jobs'].pop(),
            lambda x:x['binding']['resources'].update(workers=2),
            lambda x:x['binding']['resources'].update(retry=True),
            lambda x:x['binding']['resources'].update(resume=True),
            lambda x:x['binding']['source_release'].update(status='PENDING')]
        for change in mutations:
            candidate = copy.deepcopy(receipt); change(candidate)
            # Code/config/matrix checks must reject even a synthetically matching review hash.
            if candidate['scope']==r.SCOPE and candidate['execution_layer_review']['status']=='PASS': self.rebind(candidate)
            with self.subTest(change=change), self.assertRaises((PermissionError, ValueError)): self.approve_fixture(candidate)
        candidate=copy.deepcopy(receipt); candidate['binding']['resources']['wall_seconds']+=1
        with self.assertRaises(PermissionError): self.approve_fixture(candidate)

    def test_wrong_registration_recurrence_and_inventory_hash(self):
        receipt, *_ = self.fixture()
        for key in ('registration_digest','recurrence_digest'):
            c=copy.deepcopy(receipt); c['binding'][key]='0'*64; self.rebind(c)
            with self.assertRaises(ValueError): self.approve_fixture(c)
        for key in ('registration','inventory'):
            c=copy.deepcopy(receipt); c['binding'][key]['sha256']='0'*64; self.rebind(c)
            with self.assertRaises(ValueError): self.approve_fixture(c)
        # Actual frozen metadata validator rejects a synthetic replacement, without pixels.
        with self.assertRaises(ValueError): r.dry_run({'target': []})

    def test_trusted_loader_and_context_hashes(self):
        receipt, _, s, inventory = self.fixture(); self.approve_fixture(receipt)
        for name in sorted(r.METHODS):
            host=r.load_artifact(s, name, self.root/'artifacts', inventory)
            self.assertEqual(host.method.group, name[0]); self.assertEqual(host.method.static, name.endswith('STATIC'))
        bad=copy.deepcopy(inventory); bad['A_FULL']['training_asset_file_sha256']='0'*64
        with self.assertRaises(ValueError): r.load_artifact(s, 'A_FULL', self.root/'artifacts', bad)
        bad=copy.deepcopy(inventory); bad['A_FULL']['context_sha256']='0'*64
        with self.assertRaises(ValueError): r.load_artifact(s, 'A_FULL', self.root/'artifacts', bad)
        changed=segmenter(); changed.model.seg_head.weight.data.add_(1)
        with self.assertRaises(ValueError): r.load_artifact(changed, 'A_FULL', self.root/'artifacts', inventory)
        changed.close()

    def test_target_reader_paths_after_read_and_label_capability(self):
        _, reg, _, _ = self.fixture(); row=reg['target'][0]
        image=r.TargetReader(self.root/'target', 1024**2, 'image')
        mask=r.TargetReader(self.root/'target', 1024**2, 'mask')
        only=r.image_records([row])[0]; self.assertEqual(set(only), {'image_path','image_sha256','image_size'})
        self.assertEqual(tuple(image.read(only).shape), (1,3,512,512))
        with self.assertRaises(KeyError): mask.read(only)
        image.after_check()
        Path(row['image_path']).write_bytes(b'mutated')
        with self.assertRaises(ValueError): image.after_check()

    def test_traversal_symlink_hardlink_and_output_isolation(self):
        p=self.root/'target/a'; p.write_bytes(b'x')
        link=self.root/'target/link'; link.symlink_to(p)
        for path in (link, self.root/'target/../target/a', self.root/'checkpoint'):
            with self.assertRaises((ValueError,OSError)): r.confined(self.root/'target', path)
        hard=self.root/'target/hard'; os.link(p, hard)
        for path in (p,hard):
            with self.assertRaises(ValueError): r.confined(self.root/'target',path)
        receipt,*_=self.fixture()
        for path in (self.root/'target/out', self.root/'artifacts/out', self.root/'storage'):
            c=copy.deepcopy(receipt); c['binding']['output_dir']=str(path); self.rebind(c)
            with self.assertRaises(ValueError): self.approve_fixture(c)
        out=r.BudgetOutput(self.root/'storage/new', {}, 1024**2)
        with self.assertRaises(FileExistsError): r.BudgetOutput(out.path, {}, 1024**2)
        with self.assertRaises(ValueError): out.bytes('../escape',b'x')

    def test_full_persistence_static_reset_and_instance_order_isolation(self):
        for group in 'ABC':
            s=segmenter(); full=method(group); full.freeze(); h=OnlineHost(s,full)
            h.step(pixels()); before=copy.deepcopy(h.state); h.step(pixels(1))
            self.assertEqual(h.state['counter'],2)
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in h.state.items() if isinstance(v,torch.Tensor)))
            m=method(group,True); m.freeze(); hs=OnlineHost(s,m); hs.step(pixels()); a,_=hs.step(pixels(1))
            independent=method(group,True); independent.freeze(); fresh=OnlineHost(s,independent); b,_=fresh.step(pixels(1))
            self.assertTrue(torch.equal(a,b)); self.assertIsNot(hs.method,fresh.method)
            self.assertIsNot(hs.state,fresh.state); self.assertEqual(h.visits,2); self.assertEqual(fresh.visits,1)
            s.close()

    def test_exact_nominal_counts_and_schedule(self):
        jobs=r.matrix(r.SCOPE)
        self.assertEqual((len(jobs),sum(j['arrivals'] for j in jobs),sum(j['network_forwards'] for j in jobs),sum(j['backwards'] for j in jobs),sum(j['Adam'] for j in jobs)),(24,46824,122913,5853,5853))
        self.assertEqual(r.schedule(1),[jobs])
        self.assertEqual([sum(j['network_forwards'] for j in lane) for lane in r.schedule(3)],[40971]*3)
        self.assertEqual([set(j['order'] for j in lane) for lane in r.schedule(3)],[{0},{1},{4}])
        self.assertEqual(r.physical(dict(counts=dict(forwards=8,backwards=1,base_adam=1)),'C_BASE'),dict(forwards=8,backwards=1,Adam=1))
        with self.assertRaises(ValueError): r.physical(dict(counts=dict(forwards=7,backwards=1,base_adam=1)),'C_BASE')
        for arm in r.ARMS:
            f=8 if arm=='C_BASE' else 1 if arm=='C0' else 2
            c=r.physical(dict(counts=dict(forwards=f,backwards=int(arm=='C_BASE'),base_adam=int(arm=='C_BASE'))),arm)
            self.assertEqual(c['forwards']*1951,next(j for j in jobs if j['arm']==arm)['network_forwards'])

    def test_online_and_posthoc_separate_and_evaluator_equivalence(self):
        _,reg,_,_=self.fixture(); row=reg['target'][0]
        s=segmenter(); h=OnlineHost(s); out=r.BudgetOutput(self.root/'storage/job',{},1024**2)
        image=r.TargetReader(self.root/'target',1024**2,'image')
        with patch('dpa_ctta.source_io.read_mask',side_effect=AssertionError('no label in online')):
            counts=r.online(h,r.image_records([row]),image,'C0',out,lambda:None)
        self.assertEqual(counts,dict(forwards=1,backwards=0,Adam=0))
        h._check_frozen(boundary=True); s.close(); del h
        scored=r.posthoc([row],r.TargetReader(self.root/'target',1024**2,'mask'),'C0',0,out,lambda:None)
        self.assertEqual(len(scored),1); self.assertEqual(len(scored[0]['dice']),2)
        self.assertEqual((out.path/'prediction_0000.bits').stat().st_size,65536)

    def test_finite_caps_gpu_unqualified(self):
        r.device_policy(self.cfg)
        for key,value in [('wall_seconds',0),('job_wall_seconds',float('inf')),('output_bytes',1),('physical_GPU_ids',[0]),('device','cuda:0')]:
            c=copy.deepcopy(self.cfg);c[key]=value
            with self.assertRaises((PermissionError,ValueError)):r.device_policy(c)
        out=r.BudgetOutput(self.root/'storage/job',{},1024**2)
        with self.assertRaises(ValueError):out.bytes('large',b'x'*1024**2)
        with self.assertRaises(TimeoutError):r.resource_check(out,dict(wall_seconds=1,output_bytes=1024**2),time.monotonic()-2)

    def exited(self, out, code=0):
        p=subprocess.Popen([sys.executable,'-c',f'raise SystemExit({code})'],start_new_session=True)
        p.wait();return dict(process=p,out=out,started=time.monotonic())

    def test_already_exited_worker_terminal_resource_audit(self):
        out=r.BudgetOutput(self.root/'storage/job',{},1024**2);rec=self.exited(out)
        (out.path/'rogue').write_bytes(b'x'*1024**2)
        with self.assertRaises(ValueError):r.terminal(rec,self.cfg)
        self.assertTrue(rec['terminal_done']);self.assertTrue(rec['cleaned'])
        out2=r.BudgetOutput(self.root/'storage/job2',{},1024**2);rec=self.exited(out2);rec['started']-=30
        with self.assertRaises(TimeoutError):r.terminal(rec,self.cfg)

    def test_execution_first_survives_cleanup_and_evidence_no_retry(self):
        approved=dict(config=self.cfg,binding={'procedural':True})
        out=r.BudgetOutput(self.root/'storage/run',{},self.cfg['output_bytes']);calls=[]
        def start(slot,job,job_out):
            calls.append(job['job_id']);p=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],start_new_session=True);p.wait();return p
        with patch.object(r,'cleanup_owned',side_effect=RuntimeError('cleanup failure')), patch.object(out,'evidence',side_effect=OSError('evidence failure')):
            with self.assertRaisesRegex(RuntimeError,'worker nonzero exit 7'):r.supervise(approved,out,start)
        self.assertLessEqual(len(calls),3);self.assertEqual(len(set(calls)),len(calls))
        self.assertFalse((out.path/'completion.json').exists())

    def test_scheduler_all_24_fresh_jobs_three_lanes_and_report(self):
        approved=dict(config=self.cfg,binding={'procedural':True})
        out=r.BudgetOutput(self.root/'storage/run',{},self.cfg['output_bytes']);seen=[]
        def start(slot,job,job_out):
            seen.append((slot,job['job_id'],job_out.owner,job_out.path))
            job_out.write('worker.json',dict(status='JOB_PENDING_TERMINAL_AUDIT',target_after_check='UNCHANGED'))
            job_out.write('scalars.private.json',[dict(arm=job['arm'],order=job['order'],content='one',domain='REFUGE',subset='remaining_dev',dice=[.5,.6],assd=[None,1])])
            return subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
        r.supervise(approved,out,start)
        self.assertEqual(len(seen),24);self.assertEqual(len({x[2] for x in seen}),24);self.assertEqual(len({x[3] for x in seen}),24)
        for slot in range(3):self.assertEqual([x[1] for x in seen if x[0]==slot],[j['job_id'] for j in r.schedule(3)[slot]])
        result=json.loads((out.path/'report.json').read_text())
        self.assertFalse(result['automatic_nomination']);self.assertFalse(result['next_execution_authorized'])
        self.assertEqual({v['reference'] for v in result['paired']},{'C_BASE','A_STATIC','B_STATIC','C_STATIC'})
        self.assertTrue(result['absolute'])
        for _,_,_,p in seen:self.assertTrue((p/'terminal.json').exists());self.assertTrue((p/'completion.json').exists())

    def test_worker_first_error_survives_after_check_failure(self):
        receipt,reg,_,_=self.fixture();approved=self.approve_fixture(receipt)
        out=r.BudgetOutput(self.root/'storage/job',{},1024**2)
        job=copy.deepcopy(r.matrix(r.SCOPE)[0]);job.update(arrivals=1,scored_contents=1)
        with patch.object(r,'stream',return_value=reg['target']),patch.object(r,'make_host',side_effect=RuntimeError('first online error')),patch.object(r.TargetReader,'after_check',side_effect=ValueError('after error')):
            with self.assertRaisesRegex(RuntimeError,'first online error'):r.execute_job(approved,job,out)
        self.assertEqual(json.loads((out.path/'first_error.json').read_text())['message'],'first online error')
        self.assertEqual(json.loads((out.path/'worker.json').read_text())['status'],'FAILED')

    def test_procedural_job_completes_online_before_any_mask(self):
        receipt,reg,s,_=self.fixture();approved=self.approve_fixture(receipt)
        out=r.BudgetOutput(self.root/'storage/job',{},1024**2)
        job=copy.deepcopy(r.matrix(r.SCOPE)[1]);job.update(arrivals=1,scored_contents=1,network_forwards=1)
        from dpa_ctta.source_io import read_mask
        def label(*args,**kwargs):
            self.assertEqual(s.handles,[])
            return read_mask(*args,**kwargs)
        with patch.object(r,'stream',return_value=reg['target']),patch.object(r,'make_host',return_value=(OnlineHost(s),None)),patch('dpa_ctta.source_io.read_mask',side_effect=label):
            r.execute_job(approved,job,out)
        result=json.loads((out.path/'worker.json').read_text())
        self.assertEqual(result['status'],'JOB_PENDING_TERMINAL_AUDIT')
        self.assertEqual(result['target_after_check'],'UNCHANGED')
        self.assertFalse((out.path/'completion.json').exists())

    def test_real_factory_trusted_loader_fresh_instances(self):
        receipt,_,_,_=self.fixture();approved=self.approve_fixture(receipt)
        with patch.object(r,'load_model',side_effect=lambda raw:segmenter()):
            for arm in r.ARMS[1:]:
                a,_=r.make_host(approved,arm);b,_=r.make_host(approved,arm)
                self.assertIsNot(a.segmenter,b.segmenter)
                if arm!='C0':
                    self.assertIsNot(a.method,b.method);self.assertIsNot(a.state,b.state)
                a.step(pixels());self.assertEqual(b.visits,0)
                a.segmenter.close();b.segmenter.close()

    def test_serial_scheduler_preserves_frozen_job_list(self):
        cfg=copy.deepcopy(self.cfg);cfg['workers']=1
        approved=dict(config=cfg,binding={'procedural':True});out=r.BudgetOutput(self.root/'storage/run',{},cfg['output_bytes']);seen=[]
        def start(slot,job,job_out):
            seen.append(job['job_id']);self.assertEqual(slot,0)
            job_out.write('worker.json',dict(status='JOB_PENDING_TERMINAL_AUDIT',target_after_check='UNCHANGED'))
            job_out.write('scalars.private.json',[dict(arm=job['arm'],order=job['order'],content='one',domain='REFUGE',subset='remaining_dev',dice=[.5,.6],assd=[None,1])])
            return subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
        r.supervise(approved,out,start)
        self.assertEqual(seen,[j['job_id'] for j in r.matrix(r.SCOPE)])

    def test_running_worker_wall_cap_kills_only_owned_process(self):
        cfg=copy.deepcopy(self.cfg);cfg['workers']=1;cfg['job_wall_seconds']=.01
        approved=dict(config=cfg,binding={'procedural':True});out=r.BudgetOutput(self.root/'storage/run',{},cfg['output_bytes']);started=[]
        def start(slot,job,job_out):
            p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True);started.append(p);return p
        with self.assertRaises(TimeoutError):r.supervise(approved,out,start)
        self.assertEqual(len(started),1);self.assertIsNotNone(started[0].poll())
        self.assertFalse((out.path/'completion.json').exists())

    def test_same_content_rewrite_is_not_immutable(self):
        _,reg,_,_=self.fixture();row=reg['target'][0]
        reader=r.TargetReader(self.root/'target',1024**2,'image');reader.read(r.image_records([row])[0])
        p=Path(row['image_path']);raw=p.read_bytes();p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'changed after read'):reader.after_check()

    def test_child_rechecks_exact_authority_without_freshness_bypass(self):
        receipt,_,_,_=self.fixture();approved=self.approve_fixture(receipt)
        out=r.BudgetOutput(receipt['binding']['output_dir'],dict(binding_sha256=json_digest(receipt['binding'])),self.cfg['output_bytes'])
        with self.assertRaises(ValueError):self.approve_fixture(receipt)
        with patch.object(r,'code_identity',return_value='f'*40),patch.object(r,'dry_run',return_value=dict(recurrence=dict(registration_digest='c'*64,stream_digest='d'*64))):
            r.preflight(receipt,owned_output=out.owner)
            with self.assertRaises(ValueError):r.preflight(receipt,owned_output='foreign')

    def test_prediction_quantization_matches_original_evaluator(self):
        import numpy as np
        from dpa_ctta.p1_analysis import evaluate
        logits=torch.tensor([-1e-9,0.,1e-9,-10.,10.,-1.,1.,2.]).reshape(1,2,2,2)
        probabilities=logits.sigmoid(); mask=(logits>0).float()
        bits=np.packbits((probabilities>=.5).numpy().reshape(-1))
        decoded=torch.from_numpy(np.unpackbits(bits).copy()).float().reshape(1,2,2,2)
        self.assertEqual(evaluate(probabilities,mask,'fundus'),evaluate(decoded,mask,'fundus'))

    def test_output_directory_replacement_and_links_fail(self):
        out=r.BudgetOutput(self.root/'storage/job',{},1024**2)
        moved=self.root/'storage/moved';out.path.rename(moved);out.path.symlink_to(moved,target_is_directory=True)
        with self.assertRaises(ValueError):out.write('x.json',{})
        with self.assertRaises(ValueError):r.resource_check(out,dict(wall_seconds=10,output_bytes=1024**2),time.monotonic())
        out.path.unlink();out.path.mkdir();(out.path/'owner.json').write_bytes((moved/'owner.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'replaced'):out.write('x.json',{})

    def test_cleanup_failure_is_preserved_separately(self):
        out=r.BudgetOutput(self.root/'storage/job',{},1024**2);rec=self.exited(out,7)
        with patch.object(r,'cleanup_owned',side_effect=RuntimeError('cleanup-specific')):
            with self.assertRaisesRegex(RuntimeError,'nonzero exit 7'):r.terminal(rec,self.cfg)
        self.assertEqual(rec['terminal_errors'][0]['message'],'cleanup-specific')

    def test_cost_survives_posthoc_and_prediction_failure(self):
        for failure in ('posthoc','prediction'):
            with self.subTest(failure=failure):
                receipt,reg,s,_=self.fixture();approved=self.approve_fixture(receipt)
                out=r.BudgetOutput(self.root/('storage/'+failure),{},1024**2)
                job=copy.deepcopy(r.matrix(r.SCOPE)[1]);job.update(arrivals=1,scored_contents=1,network_forwards=1)
                original=out.bytes
                def write(name,raw):
                    if failure=='prediction' and name.startswith('prediction_'): raise OSError('prediction failure')
                    return original(name,raw)
                with patch.object(r,'stream',return_value=reg['target']),patch.object(r,'make_host',return_value=(OnlineHost(s),None)),patch.object(out,'bytes',side_effect=write),patch.object(r,'posthoc',side_effect=ValueError('posthoc failure')):
                    with self.assertRaisesRegex(Exception,failure+' failure'):r.execute_job(approved,job,out)
                cost=json.loads((out.path/'worker.json').read_text())['model_cost']
                self.assertEqual(cost['physical'],dict(forwards=1,backwards=0,Adam=0))
                self.assertEqual(cost['committed_visits'],1)
                self.assertEqual(cost['prediction_files'],int(failure=='posthoc'))
                self.assertEqual(cost['completeness'],'EXACT_OBSERVED')
                self.assertEqual((out.path/'counts.json').exists(),failure=='posthoc')

    def test_partial_step_and_cost_evidence_failure_preserve_first(self):
        receipt,reg,s,_=self.fixture();approved=self.approve_fixture(receipt)
        out=r.BudgetOutput(self.root/'storage/partial',{},1024**2)
        job=copy.deepcopy(r.matrix(r.SCOPE)[1]);job.update(arrivals=1,scored_contents=1,network_forwards=1)
        h=OnlineHost(s)
        def failed_step(pixels):
            s(pixels)
            raise RuntimeError('original model failure')
        original=out.write
        def write(name,value):
            if name.startswith('cost_0000'):raise OSError('cost persistence failure')
            return original(name,value)
        with patch.object(r,'stream',return_value=reg['target']),patch.object(r,'make_host',return_value=(h,None)),patch.object(h,'step',side_effect=failed_step),patch.object(out,'write',side_effect=write):
            with self.assertRaisesRegex(RuntimeError,'original model failure'):r.execute_job(approved,job,out)
        self.assertEqual(json.loads((out.path/'first_error.json').read_text())['message'],'original model failure')
        cost=json.loads((out.path/'worker.json').read_text())['model_cost']
        self.assertEqual(cost['physical']['forwards'],1);self.assertEqual(cost['committed_visits'],0)
        self.assertEqual(cost['completeness'],'LOWER_BOUND');self.assertIsNotNone(cost['unobserved_tail'])
        self.assertEqual(cost['evidence_errors'][0]['message'],'cost persistence failure')

    def test_base_partial_physical_counts_are_not_visit_estimates(self):
        h=SimpleNamespace(counts=dict(forwards=7,backwards=1,base_adam=1),steps=0)
        cost=r.cost_state();r.capture_cost(h,'C_BASE',Counter(),cost)
        self.assertEqual(cost['physical'],dict(forwards=7,backwards=1,Adam=1));self.assertEqual(cost['committed_visits'],0)

    def test_gpu_context_cpu_resources_rejected_before_online(self):
        receipt,*_=self.fixture()
        inv=json.loads(Path(receipt['binding']['inventory']['path']).read_text())
        for name,row in inv.items():
            p=self.root/'artifacts'/(name+'.context.json');c=json.loads(p.read_text())
            c['payload']['environment']['execution_backend']={'schema':'R7_CUDA_FP32_BACKBONE_CPU_METHOD_V1'}
            c['sha256']=json_digest(c['payload']);raw=json.dumps(c).encode();p.write_bytes(raw)
            row.update(context_file_sha256=r.digest(raw),context_sha256=c['sha256'])
        raw=json.dumps(inv).encode();Path(receipt['binding']['inventory']['path']).write_bytes(raw)
        receipt['binding']['inventory']['sha256']=r.digest(raw);receipt['binding']['source_release']['artifact_identity']=r.digest(raw);self.rebind(receipt)
        with patch.object(r.TargetReader,'read',side_effect=AssertionError('no pixels')):
            with self.assertRaisesRegex(ValueError,'GPU context'):self.approve_fixture(receipt)

    def test_matrix_readiness_covers_all_eight_without_pixels(self):
        receipt,*_=self.fixture();approved=self.approve_fixture(receipt);seen=[]
        def make(a,arm):
            seen.append(arm)
            if arm=='C_BASE':return SimpleNamespace(finish=lambda state:None),{}
            return SimpleNamespace(_check_frozen=lambda **kw:None,segmenter=SimpleNamespace(close=lambda:None)),None
        with patch.object(r,'make_host',side_effect=make),patch.object(r.TargetReader,'read',side_effect=AssertionError('no pixels')):
            evidence=r.matrix_readiness(approved)
        self.assertEqual(seen,r.ARMS);self.assertEqual(evidence['model_forwards'],0)

    def test_GPU_route_requires_all_arm_gpu_binding(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(device='cuda:0',physical_GPU_ids=[6],workers=1,dtype_policy='R7_CUDA_FP32_BACKBONE_CPU_METHOD_V1',qualification={'path':'procedural'})
        # The GPU-first amendment binds every arm, including C_BASE, to CUDA.
        self.assertIsNone(r.device_policy(cfg))

    def test_real_baseline_factory_uses_GPU_in_GPU_route(self):
        receipt,*_=self.fixture();approved=self.approve_fixture(receipt);approved['config'].update(device='cuda:0',baseline_device='cpu')
        with patch.object(r.torch,'load',return_value={}),patch('dpa_ctta.b1_host.Host') as factory:
            r.make_host(approved,'C_BASE')
            factory.assert_called_once_with('C',{},'cuda:0')

    def test_explicit_user_waiver_keeps_exact_binding(self):
        receipt,*_=self.fixture()
        receipt['execution_layer_review']['status']='USER_WAIVED'
        with self.assertRaises(PermissionError):self.approve_fixture(receipt)
        receipt['user_authorization']['external_review_waiver']=dict(scope=r.SCOPE,binding_sha256=json_digest(receipt['binding']),explicit=True)
        self.approve_fixture(receipt)
        receipt['binding']['source_release']['status']='USER_ACCEPTED_VERIFIED_ARTIFACTS'
        receipt['execution_layer_review']['binding_sha256']=json_digest(receipt['binding'])
        receipt['user_authorization']['external_review_waiver']['binding_sha256']=json_digest(receipt['binding'])
        self.approve_fixture(receipt)
        receipt['binding']['seed']+=1
        with self.assertRaises(PermissionError):self.approve_fixture(receipt)
