"""Execution-layer checks on temporary synthetic files; no real source assets."""
import copy,io,json,os,subprocess,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
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
            self.assertIsNotNone(processes[0].poll());self.assertTrue(json.loads((out.path/'supervisor.json').read_text())['cleaned'])
    def test_default_entry_refuses_without_receipt(self):
        with patch.dict(os.environ,{},clear=True),patch.object(run,'preflight',side_effect=AssertionError('no preflight without receipt')):
            with self.assertRaises(PermissionError):run.main()
