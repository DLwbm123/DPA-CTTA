"""Connected production worker/queue/journal regressions on CPU synthetic assets.
Only private data/model construction, hardware checks and process transport are substituted.
No R9 execution or profile authorization is supplied by these tests.
"""
import copy
import errno
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from test_core import host_fixture
from dpa_ctta.r9_current_first import queue,runtime,target,assets,factory,resolution,streams,profile
from dpa_ctta.r9_current_first import identity as identity_module
from dpa_ctta.r9_current_first.ledger import Ledger
from dpa_ctta.r9_current_first.protocol import CAPS,RECOVERY_POLICY,digest,disabled_config,SPEC_SHA,graph
from dpa_ctta.r9_current_first.storage import write_json
from dpa_ctta.r9_current_first.recovery import bound_failure,verify_failure
from dpa_ctta.r7_source_prep import registry


class Faults(unittest.TestCase):
    def test_worker_eio_restart_journal_score_and_cross_attempt_storage(self):
        torch.set_num_threads(1)
        image=torch.rand(1,3,512,512);mask=torch.zeros(1,2,512,512)
        rows=[dict(group_id=str(i),domain='d',subset='remaining_dev',image_path='synthetic',image_sha256='1'*64,image_size=[512,512]) for i in range(52)]
        payload=dict(spec_sha256=SPEC_SHA,sources={str(i):'bound' for i in range(43)})
        lock=dict(schema='R9_SOURCE_LOCK_V1',payload=payload,sha256=digest(payload))
        nodes=[dict(id='j__online',kind='online',resource='gpu',job={'id':'j','order':0},needs=[]),
               dict(id='j__score',kind='score',resource='cpu',job={'id':'j','order':0},needs=['j__online'])]
        resolved=dict(source_job=None,arm='N',diagnostic=None,alpha=1,gradient=None,seed=20260924,order=0)
        reads=[0];inject=[True];executed=[];crash_settle=[True];probability_seals=[]
        class Reader:
            def __init__(self,root,cap,kind):self.kind=kind
            def read(self,row):
                if self.kind=='image':
                    reads[0]+=1
                    if inject[0] and reads[0]==52:raise OSError(errno.EIO,'injected after checkpoint 50 and tail 51')
                return image if self.kind=='image' else mask
            def after_check(self):pass
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            root=Path(d)/'r9';root.mkdir();identity={'runtime':'synthetic'}
            for name,value in [('source_lock',lock),('recipe_selection',{}),('gradient_selection',{})]:write_json(root/(name+'.json'),value)
            budget=dict.fromkeys(CAPS,1000);budget['disk_bytes']=160*1024**2
            config=dict(disabled_config(),profile={'node_budgets':{n['id']:budget for n in nodes},'peak_vram_bytes':{'j__online':1024}},
                        gpu_assignments=[{'physical_id':9,'uuid':'GPU-synthetic'}],bindings={'target_root':'synthetic','checkpoint_sha256':'b','refs':{'target':{'path':'synthetic','sha256':'t'}}},source_inventory={})
            for module in (queue,runtime):
                stack.enter_context(patch.object(module,'require_authorized',return_value=root))
                stack.enter_context(patch.object(module,'verify_runtime',return_value=identity))
            for module,name,value in [(queue,'graph',{'nodes':nodes}),(runtime,'source_receipts',{}),(target,'lock_sources',lock),
                                      (resolution,'resolve',resolved),(registry,'verified','{}'),(streams,'sequence',rows),
                                      (assets,'available_memory',1024),(assets,'gpu_policy',None)]:
                stack.enter_context(patch.object(module,name,return_value=value))
            stack.enter_context(patch.object(factory,'construct',side_effect=lambda *a:(host_fixture(),lambda:None)))
            stack.enter_context(patch.object(target,'TargetReader',Reader))
            stack.enter_context(patch.object(torch.cuda,'reset_peak_memory_stats'))
            original_observe=Ledger.observe
            def observe(ledger,*a,**kw):
                if kw.get('settle') and crash_settle[0]:
                    crash_settle[0]=False
                    raise OSError(errno.EIO,'injected process death after receipt before ledger settlement')
                return original_observe(ledger,*a,**kw)
            stack.enter_context(patch.object(Ledger,'observe',observe))
            original_retire=target.retire_probabilities
            def retire(path):
                probability_seals.append(runtime.read(path/'online_complete.json')['prediction_sha256'])
                # Crash before unlink leaves the retirement intent and sealed file.
                unlink=Path.unlink
                def fail_unlink(p,*a,**kw):
                    if p==path/'predictions.bits':raise OSError(errno.EIO,'before unlink')
                    return unlink(p,*a,**kw)
                with patch.object(Path,'unlink',fail_unlink):
                    with self.assertRaises(OSError):original_retire(path)
                original_retire(path)
                # Re-open ledger after unlink but before the normal release call.
                Ledger(root/'ledger',identity,[9]).sync_disk('restart-after-unlink')
                original_retire(path)
            stack.enter_context(patch.object(target,'retire_probabilities',retire))
            class Process:
                pid=12345
                def __init__(self,args,env,**kwargs):
                    packet=runtime.read(env['R9_PACKET']);packet['config']=runtime.read(packet.pop('config_path'))
                    executed.append(packet['attempt']);self.result=runtime.worker(**packet)
                def wait(self,timeout=None):return int(self.result['status']!='COMPLETE')
            stack.enter_context(patch.object(queue.subprocess,'Popen',Process))
            with self.assertRaises(OSError):queue.run(config,'/tmp/w.py','/tmp/e/bin/python')
            ledger=Ledger(root/'ledger',identity,[9]);state=ledger._read()
            self.assertEqual(state['attempts']['j__online.0']['status'],'RUNNING')
            failed=runtime.read(root/'attempts/j__online.0.json');self.assertEqual(failed['failure']['class'],'INFRASTRUCTURE')
            # Receipt and snapshot tampering rejected before any resumed call.
            failure=bound_failure(failed);bad=copy.deepcopy(failure);bad['evidence']['receipt_sha256']='bad'
            with self.assertRaises(ValueError):verify_failure(root,root/'target/j','j__online',identity,bad)
            meta=root/'target/j/checkpoint.1.json';old=meta.read_bytes();meta.write_bytes(old+b' ')
            with self.assertRaises(ValueError):verify_failure(root,root/'target/j','j__online',identity,failure)
            meta.write_bytes(old)
            original_accept=queue.accept_receipt
            def interrupted_accept(root_,state_,node_,record_,config_,ledger_):
                if node_['kind']=='score':raise OSError(errno.EIO,'queue died after COMPLETE receipt and settlement')
                return original_accept(root_,state_,node_,record_,config_,ledger_)
            with patch.object(queue,'accept_receipt',interrupted_accept):
                with self.assertRaises(OSError):queue.run(config,'/tmp/w.py','/tmp/e/bin/python')
            self.assertEqual(runtime.read(root/'queue.json')['nodes']['j__score']['status'],'RUNNING')
            state=queue.run(config,'/tmp/w.py','/tmp/e/bin/python')
            self.assertEqual(len(state['reused']),1)
            self.assertEqual(state['status'],'COMPLETE');self.assertEqual(executed,['j__online.0','j__online.1','j__score.0'])
            self.assertEqual(runtime.read(root/'target/j/recovery.json')['visits'],50)
            self.assertEqual([json.loads(x)['visit'] for x in (root/'target/j/visits.jsonl').read_text().splitlines()],list(range(1,53)))
            self.assertEqual(len((root/'target/j/scalars.private.jsonl').read_text().splitlines()),52)
            self.assertEqual(len((root/'target/j/physical.jsonl').read_text().splitlines()),53)
            self.assertFalse((root/'target/j/predictions.bits').exists())
            state=ledger._read();self.assertEqual(len(state['recovery_claims']),1)
            self.assertEqual(state['attempts']['j__online.0']['status'],'FAILED')
            self.assertIsNone(state['attempts']['j__online.0']['actual'])
            self.assertLess(state['attempts']['j__online.1']['reserved']['disk_bytes'],budget['disk_bytes']-50*target.PREDICTION_BYTES)
            self.assertLess(Ledger.total(state)['disk_bytes'],budget['disk_bytes'])
            # A next LONG10 admission sees occupancy, not two old attempt reservations.
            long_budget=dict.fromkeys(CAPS,0);long_budget['disk_bytes']=19510*target.PREDICTION_BYTES
            token=ledger.reserve('next_LONG10',9,long_budget)
            ledger.observe('next_LONG10',token,dict.fromkeys(CAPS,0),settle=True)
            ledger.sync_disk('restart-after-retirement');ledger.sync_disk('restart-after-retirement')
            queue.run(config,'/tmp/w.py','/tmp/e/bin/python');self.assertEqual(len(executed),3)
            # Uninterrupted reference uses the same real host and full journal.
            inject[0]=False;reference=Path(d)/'reference'
            seal=target.online(host_fixture(),rows,None,reference,'reference',lock,lambda:None)
            self.assertEqual(probability_seals,[seal['prediction_sha256']])

    def test_restart_transition_retry_limit_numerical_and_complete(self):
        for error in (OSError(errno.EIO,'fault'),FloatingPointError('nonfinite')):
            with self.subTest(error=type(error).__name__),tempfile.TemporaryDirectory() as d:
                root=Path(d)/'r9';root.mkdir();ident={'test':1};ledger=Ledger(root/'ledger',ident,[9]);ledger.create()
                node=dict(id='x',kind='bind_assets',resource='cpu',needs=[]);budget=dict.fromkeys(CAPS,0)
                config=dict(lr_source_policy='first_two_mean',gpu_assignments=[{'physical_id':9}],profile={'node_budgets':{'x':budget}})
                state={'nodes':{},'reused':{}}
                with patch.object(runtime,'require_authorized',return_value=root),patch.object(runtime,'verify_runtime',return_value=ident),patch.object(runtime,'dispatch',side_effect=error):
                    for i in range(2 if isinstance(error,OSError) else 1):
                        attempt=f'x.{i}';token=ledger.reserve(attempt,None,budget)
                        record=runtime.worker(config,node,None,attempt,token)
                        queue.verify_attempt(record,'x',attempt,ident,root/'ledger')
                        queue.accept_receipt(root,state,node,record,config,ledger)
                        self.assertEqual(state['nodes']['x']['status'],'RETRY' if isinstance(error,OSError) and i==0 else 'FAILED')
                self.assertEqual(state['nodes']['x']['status'],'FAILED')
                if not isinstance(error,OSError):
                    with self.assertRaises(ValueError):bound_failure(record)

    def test_complete_receipt_before_settlement_is_reconciled_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ident={'test':1};ledger=Ledger(root/'ledger',ident,[9]);ledger.create()
            budget=dict.fromkeys(CAPS,0);node=dict(id='done',kind='bind_assets',resource='cpu')
            config=dict(lr_source_policy='first_two_mean',gpu_assignments=[{'physical_id':9}],profile={'node_budgets':{'done':budget}})
            token=ledger.reserve('done.0',None,budget)
            with patch.object(runtime,'require_authorized',return_value=root),patch.object(runtime,'verify_runtime',return_value=ident),patch.object(runtime,'dispatch',return_value={'sealed':True}),patch.object(Ledger,'observe',side_effect=OSError(errno.EIO,'death before settlement')):
                with self.assertRaises(OSError):runtime.worker(config,node,None,'done.0',token)
            record=runtime.read(root/'attempts/done.0.json')
            self.assertEqual(ledger._read()['attempts']['done.0']['status'],'RUNNING')
            for _ in range(2):queue.verify_attempt(record,'done','done.0',ident,root/'ledger')
            self.assertEqual(ledger._read()['attempts']['done.0']['status'],'COMPLETE')
            self.assertEqual(len(ledger._read()['attempts']),1)

    def test_identity_failure_is_global_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ident={'test':1};ledger=Ledger(root/'ledger',ident,[9]);ledger.create()
            budget=dict.fromkeys(CAPS,0);node=dict(id='identity',kind='bind_assets',resource='cpu')
            config=dict(lr_source_policy='first_two_mean',gpu_assignments=[{'physical_id':9}],profile={'node_budgets':{'identity':budget}})
            token=ledger.reserve('identity.0',None,budget)
            with patch.object(runtime,'require_authorized',return_value=root),patch.object(runtime,'verify_runtime',return_value=ident),patch.object(runtime,'dispatch',side_effect=ValueError('identity mismatch')):
                record=runtime.worker(config,node,None,'identity.0',token)
            self.assertEqual(record['failure']['class'],'IDENTITY_OR_IMPLEMENTATION')
            self.assertTrue(runtime.read(root/'ledger/state.json')['stop'])
            with self.assertRaises(ValueError):bound_failure(record)

    def test_three_claims_and_projection_are_connected_to_frozen_policy(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger',{'test':1},[9]);ledger.create()
            for i in range(3):self.assertTrue(ledger.claim_recovery(str(i),str(i)+'.0','seal'))
            self.assertTrue(ledger.claim_recovery('0','0.0','seal'))
            self.assertFalse(ledger.claim_recovery('0','0.1','another'))
            self.assertFalse(ledger.claim_recovery('3','3.0','seal'))
        nodes=graph()['nodes'];measured={u:dict.fromkeys(CAPS,0) for u in profile.units()}
        for row in measured.values():row.update(measured=True,evidence='synthetic')
        budgets={n['id']:dict.fromkeys(CAPS,0) for n in nodes}
        for n in nodes:
            budgets[n['id']]['model_forwards']=10
            if n['kind']=='online':budgets[n['id']]['disk_bytes']=(19510 if n['job']['order']=='LONG10' else 1951)*target.PREDICTION_BYTES
        proof=profile.projection(measured,2,100,node_budgets=budgets)
        self.assertEqual(proof['upper_bound']['model_forwards'],len(nodes)*10+30)
        self.assertEqual(proof['recovery_reserve']['model_forwards'],30)
        self.assertEqual(proof['recovery_reserve']['disk_bytes'],0)
        self.assertEqual(proof['recovery_policy'],RECOVERY_POLICY)
        self.assertFalse(disabled_config()['execution_authorized'])
        self.assertEqual(disabled_config()['lr_source_policy'],'first_two_mean')
        with self.assertRaises(PermissionError):profile.measure_cpu(lambda:self.fail('must not run'),1,disabled_config())
        with self.assertRaises(PermissionError):profile.measure(lambda:self.fail('must not run'),None,1,disabled_config())
        measured['fit/A/LEGACY']['model_forwards']=1e9
        proof=profile.projection # Incorrect budgets must fail, never manufacture PASS.
        with self.assertRaises(ValueError):proof(measured,2,100,node_budgets=budgets)

if __name__=='__main__':unittest.main()
