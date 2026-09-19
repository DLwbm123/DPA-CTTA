"""Execution-layer checks on temporary synthetic files; no real source assets."""
import copy,io,json,os,subprocess,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from common import Small,segmenter,method,source_fixture,pixels
from dpa_ctta.r7_source_prep import registry as reg,runner as run
from dpa_ctta.r7_shared.context import SCIENCE
from dpa_ctta.r7_shared.preparation import prepared_artifact
from dpa_ctta.r7_shared.numerics import COUNTS

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.data=self.root/'data';self.data.mkdir();self.storage=self.root/'output';self.storage.mkdir()
        self.checkpoint=self.root/'random.pt';torch.save(Small().state_dict(),self.checkpoint);raw=self.checkpoint.read_bytes()
        self.target=dict(checkpoint=dict(sha256=reg.digest(raw),bytes=len(raw)),target=[])
        self.tdigest=reg.registration_digest(self.target);self.patch=patch.object(reg,'TARGET_DIGEST',self.tdigest);self.patch.start()
        rows=[]
        for i in range(48):
            image=np.full((8,8,3),i,dtype=np.uint8);mask=np.full((8,8),128 if i%2 else 0,dtype=np.uint8)
            paths={};hashes={}
            for k,a in [('image',image),('mask',mask)]:
                relative='RIM_ONE_r3/'+k+'/'+str(i)+'.png';p=self.data/relative;p.parent.mkdir(exist_ok=True,parents=True);Image.fromarray(a).save(p);paths[k+'_relative']=relative;hashes[k+'_sha256']=reg.digest(p.read_bytes())
            rows.append(dict(sample_id='fixture_'+str(i),group_id=hashes['image_sha256'],domain='RIM_ONE_r3',image_size=[8,8],**paths,**hashes))
        self.manifest=dict(schema='R7_SOURCE_MANIFEST_V1',source='RIM_ONE_r3',target_registration_digest=self.tdigest,grouping=dict(kind='CONTENT',evidence='procedurally encoded distinct synthetic pixels; not patients',dependency_risk='PATIENT_DEPENDENCE_UNKNOWN'),checkpoint=dict(sha256=reg.digest(raw),bytes=len(raw),provenance='random Small fixture',pretraining_exposure='PROCEDURAL_ONLY'),records=rows)
        self.split=reg.freeze(self.manifest,self.target)
        self.config=json.loads(run.DEFAULTS.read_text());self.config.update(enabled=True,storage_root=str(self.storage),resource_authorization=dict(scope='SOURCE_PREP',receipt_id='SYNTHETIC_TEST_ONLY_NOT_REAL_AUTH'))
        self.receipt=dict(schema='R7_SOURCE_PREP_AUTH_V1',scope='SOURCE_PREP',enabled=True,code_sha='synthetic-code',science_sha256=SCIENCE,source_binding_status='BOUND',user_authorization=dict(granted=True,scope='SOURCE_PREP',receipt_id='SYNTHETIC_TEST_ONLY_NOT_REAL_AUTH'),execution_layer_review=dict(status='PASS',scope='SOURCE_PREP',code_sha='synthetic-code',checkpoint_sha256=self.manifest['checkpoint']['sha256']),source_root=str(self.data),checkpoint_path=str(self.checkpoint),output_dir=str(self.storage/'run'))
        for key,value in [('manifest',self.manifest),('split',self.split),('target',self.target),('config',self.config)]:
            p=self.root/(key+'.json');p.write_text(json.dumps(value));self.receipt[key]=dict(path=str(p),sha256=reg.digest(p.read_bytes()));self.receipt['execution_layer_review'][key+'_sha256']=self.receipt[key]['sha256']
        self.identity=patch.object(run,'code_identity',return_value='synthetic-code');self.identity.start()
    def tearDown(self):self.identity.stop();self.patch.stop();self.temp.cleanup()
    def approved(self):return run.preflight(self.receipt)
    def output(self):return run.BudgetOutput(self.storage/'run',{},self.config['output_bytes'])
    def test_metadata_split_freezes_before_decode_and_preserves_existing(self):
        with patch('PIL.Image.open',side_effect=AssertionError('metadata must not decode')):
            a=reg.audit(self.manifest,self.split,self.target);self.assertEqual(a['fold_groups'],dict(fit=33,cal=7,val=8));self.assertFalse(a['patient_independence']);self.assertEqual(self.split,reg.freeze(self.manifest,self.target,self.split['folds']))
    def test_overlap_minimum_duplicate_and_filename_group_rejected(self):
        for change in ('fold','minimum','duplicate','filename','patient'):
            m=copy.deepcopy(self.manifest);s=copy.deepcopy(self.split)
            if change=='fold':s['folds']['cal'][0]=s['folds']['fit'][0]
            elif change=='minimum':m['records']=m['records'][:10]
            elif change=='duplicate':m['records'].append(m['records'][0])
            elif change=='filename':m['records'][0]['group_id']='filename_0'
            else:m['grouping']['kind']='PATIENT'
            with self.assertRaises(ValueError):reg.audit(m,s,self.target)
    def test_target_overlap_and_bad_domain_rejected(self):
        t=copy.deepcopy(self.target);t['target']=[dict(image_sha256=self.manifest['records'][0]['image_sha256'])];m=copy.deepcopy(self.manifest);m['target_registration_digest']=reg.registration_digest(t)
        with patch.object(reg,'TARGET_DIGEST',m['target_registration_digest']):
            with self.assertRaises(ValueError):reg.freeze(m,t)
        m=copy.deepcopy(self.manifest);m['records'][0]['domain']='REFUGE'
        with self.assertRaises(ValueError):reg.freeze(m,self.target)
    def test_source_path_escape_and_symlink_hardlink_rejected(self):
        with self.assertRaises(ValueError):reg.asset_path(self.data,'../target.png')
        p=self.data/self.manifest['records'][0]['image_relative'];link=self.root/'link';link.symlink_to(p)
        with self.assertRaises(ValueError):reg.verified(link,self.manifest['records'][0]['image_sha256'],100000)
        hard=self.root/'hard';os.link(p,hard)
        with self.assertRaises(ValueError):reg.verified(p,self.manifest['records'][0]['image_sha256'],100000)
    def test_same_byte_decode_real_reader_on_synthetic_PNG_and_source_after_check(self):
        count=Counter();reader=reg.Reader(self.manifest,self.split,self.target,self.data,count,1024**2);row=self.manifest['records'][0]
        x=reader.decode(row,'image');y=reader.decode(row,'mask');self.assertEqual(tuple(x.shape),(1,3,512,512));self.assertEqual(tuple(y.shape),(1,2,512,512));self.assertTrue((y==1).all());self.assertEqual(count['RGB_decodes'],1);reader.after_check()
        (self.data/row['image_relative']).write_bytes(b'changed')
        with self.assertRaises(ValueError):reader.after_check()
    def test_file_digest_mismatch_rejects_before_decode(self):
        reader=reg.Reader(self.manifest,self.split,self.target,self.data,Counter(),1024**2);row=copy.deepcopy(self.manifest['records'][0]);row['image_sha256']='0'*64
        with patch('PIL.Image.open',side_effect=AssertionError('must reject before decoder')):
            with self.assertRaises(ValueError):reader.decode(row,'image')
    def test_disabled_wrong_scope_and_old_review_reject_before_assets(self):
        for key,val in [('enabled',False),('scope','TARGET_SCREEN'),('user_authorization',dict(granted=False)),('execution_layer_review',dict(status='PASS',scope='STAGE_I_IMPLEMENTATION',code_sha='synthetic-code'))]:
            r=copy.deepcopy(self.receipt);r[key]=val
            with patch.object(run,'verified',side_effect=AssertionError('must reject before metadata/assets')):
                with self.assertRaises(PermissionError):run.preflight(r)
    def test_preflight_real_synthetic_metadata_and_wrong_code_science_resource_hash(self):
        self.assertEqual(self.approved()['audit']['groups'],48)
        for which in ('code','science','config'):
            r=copy.deepcopy(self.receipt)
            if which=='code':r['code_sha']='wrong'
            elif which=='science':r['science_sha256']={}
            else:r['config']['sha256']='0'*64
            with self.assertRaises((ValueError,PermissionError)):run.preflight(r)
    def test_CPU_device_route_rejects_cuda_and_unbounded_resources_without_GPU_query(self):
        for key,val in [('device','cuda:0'),('physical_GPU_ids',[5]),('wall_seconds',0),('output_bytes',float('inf')),('dtype_policy','float16'),('workers',3)]:
            c=dict(self.config);c[key]=val
            with self.assertRaises(ValueError):run.device_policy(c)
        run.device_policy(self.config)
    def test_output_source_overlap_and_memory_overflow_rejected(self):
        r=copy.deepcopy(self.receipt);r['output_dir']=str(self.data/'bad')
        with self.assertRaises(ValueError):run.preflight(r)
        c=dict(self.config,max_decoded_bytes=1);p=Path(r['config']['path']);p.write_text(json.dumps(c));r=copy.deepcopy(self.receipt);r['config']['sha256']=reg.digest(p.read_bytes());r['execution_layer_review']['config_sha256']=r['config']['sha256']
        with self.assertRaises(ValueError):run.preflight(r)

    def test_GPU_requires_new_grant_and_exact_qualification_without_initializing(self):
        self.config.update(device='cuda:0',physical_GPU_ids=[5],dtype_policy='R7_CUDA_FP32_BACKBONE_CPU_FP64_LATENT_V1')
        p=Path(self.receipt['config']['path']);p.write_text(json.dumps(self.config))
        self.receipt['config']['sha256']=reg.digest(p.read_bytes());self.receipt['execution_layer_review']['config_sha256']=reg.digest(p.read_bytes())
        q=self.root/'qualification.json';q.write_text(json.dumps(dict(status='PASSED',code_sha='synthetic-code',physical_GPU_ids=[5],failures=0,checks=['procedural fixture'])))
        self.receipt['gpu_qualification']=dict(path=str(q),sha256=reg.digest(q.read_bytes()))
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='5'),patch('torch.cuda.is_available',side_effect=AssertionError('preflight must not initialize GPU')):
            with self.assertRaises(PermissionError):self.approved()
            self.receipt['execution_layer_review']['status']='USER_AUTHORIZED_GPU_QUALIFIED'
            with self.assertRaises(PermissionError):self.approved()
            self.receipt['user_authorization'].update(gpu_transition_authorized=True,base_cpu_code_sha='f719c703087b38c07bdfbe7ce9dcfa62d88a12d9')
            self.assertEqual(self.approved()['config']['device'],'cuda:0')
            bad=json.loads(q.read_bytes());bad['code_sha']='wrong';q.write_text(json.dumps(bad));self.receipt['gpu_qualification']['sha256']=reg.digest(q.read_bytes())
            with self.assertRaises(ValueError):self.approved()

    def test_GPU_resource_mapping_and_backend_settings_fail_closed(self):
        c=dict(self.config,device='cuda:0',physical_GPU_ids=[5],dtype_policy='R7_CUDA_FP32_BACKBONE_CPU_FP64_LATENT_V1')
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='5'):
            run.device_policy(c)
            for ids in ([4],[5,6],[True],[],None):
                with self.assertRaises(ValueError):run.device_policy(dict(c,physical_GPU_ids=ids))
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='6'):
            with self.assertRaises(ValueError):run.device_policy(c)
        with patch.dict(os.environ,CUBLAS_WORKSPACE_CONFIG=''),patch('torch.cuda.is_available',side_effect=AssertionError('configuration must reject first')):
            with self.assertRaises(ValueError):run.configure_backend(c)
    def test_budget_phases_full_counts_no_silent_truncation(self):
        expected=run.expected_counts(self.split['folds']);self.assertEqual(expected['A_basis']['source_VJP'],1024);self.assertEqual(expected['A_FULL/fit']['backbone_forwards'],12000);self.assertEqual(expected['C_STATIC/constant_variance']['backbone_forwards'],224)
        c=Counter();m=run.Meter({'fit':dict(backbone_forwards=2)},30,lambda row:None,counter=c);m.mark('fit');m.before_forward();c['backbone_forwards']=2
        with self.assertRaises(ValueError):m.before_forward()
        m.mark(None);m.complete()
        m=run.Meter({'fit':dict(backbone_forwards=2)},30,lambda row:None,counter=Counter());m.mark('fit')
        with self.assertRaises(ValueError):m.mark(None)
    def test_synthetic_checkpoint_CPU_load_and_wrong_state_reject(self):
        with patch('dpa_ctta.integrations.ctta_suite.build_reference_model',side_effect=lambda task:(Small(),None)):
            s=run.load_model(self.checkpoint.read_bytes());self.assertTrue(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in s.model.parameters()));s.close()
            raw=io.BytesIO();torch.save({'wrong':torch.ones(1)},raw)
            with self.assertRaises(RuntimeError):run.load_model(raw.getvalue())
    def test_six_artifact_release_and_independent_expected_loader(self):
        s=segmenter();models={}
        for g in 'ABC':
            for static in (False,True):
                m=method(g,static);m.freeze();name=g+('_STATIC' if static else '_FULL')
                models[name]=dict(**prepared_artifact(s,m),fit_steps=1000,cal_steps=256,validation=[],projection_audit={})
        # Synthetic artifact fixtures carry step markers only; no source fitting.
        out=self.output();inventory=run.release(dict(models=models),s,out)
        h=run.load_artifact(s,'B_FULL',out.path,inventory);h.step(pixels())
        b=segmenter()
        with torch.no_grad():b.model.seg_head.bias.add_(1)
        before=COUNTS['backbone_forwards']
        with self.assertRaises(ValueError):run.load_artifact(b,'B_FULL',out.path,inventory)
        self.assertEqual(COUNTS['backbone_forwards'],before)
        s.close();b.close()
    def test_output_budget_preserves_first_evidence_and_no_overwrite(self):
        out=run.BudgetOutput(self.storage/'run',{},70000)
        with self.assertRaises(ValueError):out.bytes('large.bin',b'x'*70000)
        with self.assertRaises(ValueError):out.evidence('../escape.json',{})
        out.evidence('first_error.json',dict(type='first'))
        with self.assertRaises(FileExistsError):out.evidence('first_error.json',dict(type='second'))
        self.assertEqual(json.loads((out.path/'first_error.json').read_text())['type'],'first')
    def test_worker_failure_after_check_failure_are_separate_no_retry(self):
        approved=self.approved();out=self.output();data,_=source_fixture()
        with patch.object(reg.Reader,'data',return_value=data),patch.object(reg.Reader,'after_check',side_effect=ValueError('second_after_check')),patch('dpa_ctta.integrations.ctta_suite.build_reference_model',side_effect=lambda task:(Small(),None)),patch.object(run,'prepare_tensors',side_effect=RuntimeError('first_training_failure')) as train:
            with self.assertRaisesRegex(RuntimeError,'first_training_failure'):run.execute(approved,out)
            self.assertEqual(train.call_count,1)
        self.assertEqual(json.loads((out.path/'first_error.json').read_text())['message'],'first_training_failure');self.assertEqual(json.loads((out.path/'after_check_error.json').read_text())['message'],'second_after_check')
    def test_supervisor_CPU_success_failure_timeout_cleanup(self):
        for i,code in enumerate(('pass','raise SystemExit(3)','import time;time.sleep(10)')):
            out=run.BudgetOutput(self.storage/str(i),{},1000000);processes=[]
            def start():
                p=subprocess.Popen([sys.executable,'-c',code],start_new_session=True);processes.append(p);return p
            if i:
                with self.assertRaises((RuntimeError,TimeoutError)):run.supervise_one(start,dict(wall_seconds=.2,output_bytes=1000000),out)
            else:run.supervise_one(start,dict(wall_seconds=3,output_bytes=1000000),out)
            self.assertIsNotNone(processes[0].poll());self.assertTrue(json.loads((out.path/'supervisor.json').read_text())['cleaned'],{p.name:p.read_text() for p in out.path.glob('supervisor*.json')})
    def test_default_entry_refuses_without_receipt(self):
        with patch.dict(os.environ,{},clear=True),patch.object(run,'preflight',side_effect=AssertionError('no preflight without receipt')):
            with self.assertRaises(PermissionError):run.main()

    def fixture_process(self,code=0):
        return SimpleNamespace(pid=123456789,returncode=code,poll=lambda:code)
    def test_SP1_noncanonical_paths_reject_before_assets_or_output(self):
        cases=[('output_dir',self.storage/'..'/'data'/'new'),
               ('output_dir',self.storage/'..'/'outside'),
               ('output_dir',self.storage/'..'/'config.json'/'new'),
               ('output_dir',self.storage/'..'/run.ROOT.relative_to('/')/'new'),
               ('source_root',self.storage/'..'/'data'),
               ('checkpoint_path',self.storage/'..'/'random.pt')]
        for key,path in cases:
            with self.subTest(key=key,path=str(path)):
                r=copy.deepcopy(self.receipt);r[key]=str(path)
                with patch.object(run,'load_model',side_effect=AssertionError('asset load')),patch.object(reg.Reader,'data',side_effect=AssertionError('asset read')):
                    with self.assertRaises(ValueError):run.preflight(r)
                self.assertFalse((self.data/'new').exists());self.assertFalse((self.storage/'run').exists())
    def test_SP1_storage_and_metadata_noncanonical(self):
        c=dict(self.config,storage_root=str(self.storage/'..'/'output'));p=Path(self.receipt['config']['path']);p.write_text(json.dumps(c))
        r=copy.deepcopy(self.receipt);r['config']['sha256']=reg.digest(p.read_bytes());r['execution_layer_review']['config_sha256']=r['config']['sha256']
        with self.assertRaises(ValueError):run.preflight(r)
        r=copy.deepcopy(self.receipt);r['manifest']['path']=str(self.storage/'..'/'manifest.json')
        with self.assertRaises(ValueError):run.preflight(r)
    def test_SP1_output_constructor_and_parent_validation(self):
        with self.assertRaises(ValueError):run.BudgetOutput(self.storage/'..'/'data'/'new',{},1000000)
        link=self.storage/'link';link.symlink_to(self.data,target_is_directory=True)
        r=copy.deepcopy(self.receipt);r['output_dir']=str(link/'new')
        with self.assertRaises(ValueError):run.preflight(r)
        r['output_dir']=str(self.storage/'missing'/'new')
        with self.assertRaises((ValueError,FileNotFoundError)):run.preflight(r)
        self.assertFalse((self.data/'new').exists())
    def test_SP2_first_error_survives_evidence_failure(self):
        out=self.output();attempts=[]
        def evidence(name,value):attempts.append((name,value));raise OSError('synthetic evidence failure')
        with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)) as cleanup,patch.object(out,'evidence',side_effect=evidence),patch('sys.stderr',new=io.StringIO()) as stderr:
            with self.assertRaisesRegex(RuntimeError,'source worker nonzero exit 3'):run.supervise_one(lambda:self.fixture_process(3),self.config,out)
        self.assertEqual(cleanup.call_count,1);self.assertIn('supervisor.json',[n for n,v in attempts]);self.assertFalse(attempts[-1][1]['retry']);self.assertIn('synthetic evidence failure',stderr.getvalue())
    def test_SP2_start_timeout_cleanup_and_evidence_failures(self):
        for kind in ('start','timeout','cleanup','evidence_only'):
            with self.subTest(kind=kind):
                out=run.BudgetOutput(self.storage/kind,{},1000000);attempts=[];p=self.fixture_process()
                start=Mock(side_effect=LookupError('first start')) if kind=='start' else Mock(return_value=p)
                cleanup=Mock(side_effect=ArithmeticError('first cleanup')) if kind=='cleanup' else Mock(side_effect=lambda r:r.update(cleaned=True))
                def evidence(name,value):attempts.append((name,value));raise OSError('first evidence')
                expected={'start':LookupError,'timeout':TimeoutError,'cleanup':ArithmeticError,'evidence_only':OSError}[kind]
                with patch.object(run,'cleanup_owned',cleanup),patch.object(out,'evidence',side_effect=evidence),patch('sys.stderr',new=io.StringIO()),patch.object(run.time,'monotonic',side_effect=[0]+[2 if kind=='timeout' else 0]*100):
                    with self.assertRaises(expected):run.supervise_one(start,dict(wall_seconds=1,output_bytes=1000000),out)
                self.assertEqual(start.call_count,1);self.assertEqual(cleanup.call_count,int(kind!='start'));self.assertIn('supervisor.json',[n for n,v in attempts]);summary=next(v for n,v in attempts if n=='supervisor.json');self.assertEqual(summary['complete'],kind=='evidence_only');self.assertFalse(summary['retry']);self.assertIn('supervisor.first_error.json',[n for n,v in attempts]);self.assertEqual(len(attempts),len({n for n,v in attempts}))
    def test_SP3_already_exited_child_over_tree_cap(self):
        out=self.output();(out.path/'worker.log').write_bytes(b'x'*220000)
        with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)):
            with self.assertRaisesRegex(ValueError,'output cap'):run.supervise_one(lambda:self.fixture_process(),dict(wall_seconds=30,output_bytes=200000),out)
        self.assertFalse(json.loads((out.path/'supervisor.json').read_text())['complete'])
    def test_SP3_terminal_wall_boundary(self):
        for elapsed in (1,2):
            out=run.BudgetOutput(self.storage/str(elapsed),{},1000000)
            with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)),patch.object(run.time,'monotonic',side_effect=[0]+[elapsed]*100):
                with self.assertRaises(TimeoutError):run.supervise_one(lambda:self.fixture_process(),dict(wall_seconds=1,output_bytes=1000000),out)
            self.assertFalse(json.loads((out.path/'supervisor.json').read_text())['complete'])
    def test_SP3_growth_at_exit_includes_nested_records(self):
        out=self.output();p=self.fixture_process();calls=[]
        def poll():
            calls.append(1)
            if len(calls)==1:return None
            (out.path/'worker').mkdir();(out.path/'worker'/'diagnostics.bin').write_bytes(b'x'*220000);return 0
        p.poll=poll
        with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)),patch.object(run.time,'sleep'):
            with self.assertRaisesRegex(ValueError,'output cap'):run.supervise_one(lambda:p,dict(wall_seconds=30,output_bytes=200000),out)
        self.assertEqual(len(calls),2)
    def test_SP3_global_terminal_reserve_and_evidence_budget(self):
        out=run.BudgetOutput(self.storage/'reserve',{},200000);(out.path/'worker.log').write_bytes(b'x'*110000)
        with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)):
            with self.assertRaisesRegex(ValueError,'output cap'):run.supervise_one(lambda:self.fixture_process(),dict(wall_seconds=30,output_bytes=200000),out)
        self.assertFalse(json.loads((out.path/'supervisor.json').read_text())['complete'])
        (out.path/'worker.log').write_bytes(b'x'*200000)
        with self.assertRaisesRegex(ValueError,'cap'):out.evidence('late.json',{})
        self.assertFalse((out.path/'late.json').exists())
    def test_SP3_worker_first_error_preserved_with_terminal_audit_failure(self):
        out=self.output();(out.path/'worker.log').write_bytes(b'x'*220000)
        with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)):
            with self.assertRaisesRegex(RuntimeError,'nonzero exit 3'):run.supervise_one(lambda:self.fixture_process(3),dict(wall_seconds=30,output_bytes=200000),out)
        self.assertIn('output cap',(out.path/'supervisor.resource_error.json').read_text())
    def test_SP3_main_completion_resource_audit(self):
        out=self.output();worker=out.path/'worker';worker.mkdir();(worker/'execution.json').write_text(json.dumps(dict(status='SOURCE_PREP_COMPLETE_PENDING_REVIEW',source_after_check='UNCHANGED')))
        (out.path/'worker.log').write_bytes(b'x'*220000)
        with self.assertRaisesRegex(ValueError,'output cap'):run.publish_completion(out,dict(wall_seconds=30,output_bytes=200000),run.time.monotonic())
        self.assertFalse((out.path/'completion.json').exists())

    def test_SP2_summary_failure_records_failure_without_retry(self):
        out=self.output();original=out.evidence;attempts=[]
        def evidence(name,value):
            attempts.append(name)
            if name=='supervisor.json':raise OSError('summary storage fault')
            return original(name,value)
        with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)),patch.object(out,'evidence',side_effect=evidence),patch('sys.stderr',new=io.StringIO()):
            with self.assertRaisesRegex(OSError,'summary storage fault'):run.supervise_one(lambda:self.fixture_process(),self.config,out)
        self.assertIn('summary storage fault',(out.path/'supervisor.first_error.json').read_text());self.assertTrue((out.path/'supervisor.evidence_errors.json').exists());self.assertEqual(len(attempts),len(set(attempts)))
    def test_SP3_terminal_audit_exception_preserves_worker_error(self):
        out=self.output();original=run.tree_bytes
        def size(path):
            if (out.path/'process.json').exists():raise OSError('audit IO failure')
            return original(path)
        with patch.object(run,'cleanup_owned',side_effect=lambda r:r.update(cleaned=True)),patch.object(run,'tree_bytes',side_effect=size),patch('sys.stderr',new=io.StringIO()) as stderr:
            with self.assertRaisesRegex(RuntimeError,'nonzero exit 3'):run.supervise_one(lambda:self.fixture_process(3),self.config,out)
        self.assertIn('audit IO failure',stderr.getvalue())
    def test_SP3_completion_counts_terminal_write_time_and_bytes(self):
        for kind in ('time','bytes','success'):
            out=run.BudgetOutput(self.storage/kind,{},1000000);worker=out.path/'worker';worker.mkdir();(worker/'execution.json').write_text(json.dumps(dict(status='SOURCE_PREP_COMPLETE_PENDING_REVIEW',source_after_check='UNCHANGED')))
            original=out.evidence;clock=[0]
            def evidence(name,value):
                original(name,value)
                if kind=='time':clock[0]=1
                elif kind=='bytes':(out.path/'worker.log').write_bytes(b'x'*1000000)
            with patch.object(out,'evidence',side_effect=evidence),patch.object(run.time,'monotonic',side_effect=lambda:clock[0]):
                if kind=='success':run.publish_completion(out,dict(wall_seconds=1,output_bytes=1000000),0)
                else:
                    with self.assertRaises((ValueError,TimeoutError)):run.publish_completion(out,dict(wall_seconds=1,output_bytes=1000000),0)
            self.assertEqual((out.path/'completion.json').exists(),kind=='success');self.assertEqual((out.path/'completion.pending.json').exists(),kind!='success')

    def test_cleanup_EPERM_requires_bounded_reap_and_absent_group(self):
        for state in ('reaped','alive','group_present'):
            with self.subTest(state=state):
                first=PermissionError('signal zero denied');p=self.fixture_process();p.wait=Mock(side_effect=subprocess.TimeoutExpired('owned child',.5)) if state=='alive' else Mock(return_value=0);record=dict(process=p)
                with patch.object(run,'stop_owned',side_effect=first),patch.object(run.sys,'platform','darwin'),patch.object(run.subprocess,'check_output',return_value=str(p.pid) if state=='group_present' else '1\n2\n'):
                    if state=='reaped':run.cleanup_owned(record);self.assertTrue(record['cleaned'])
                    else:
                        with self.assertRaises(PermissionError) as raised:run.cleanup_owned(record)
                        self.assertIs(raised.exception,first);self.assertNotIn('cleaned',record)
                p.wait.assert_called_once_with(timeout=.5)
