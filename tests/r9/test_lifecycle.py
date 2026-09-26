import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_core import fit_fixture,host_fixture
from dpa_ctta.r9_current_first import target,streams
from dpa_ctta.r9_current_first.validation import Calibrator
from dpa_ctta.r9_current_first.protocol import SPEC_SHA,CAPS,digest
from dpa_ctta.r9_current_first.ledger import Ledger
from dpa_ctta.r9_current_first.physical import Meter
from dpa_ctta.r9_current_first.runtime import Budget
from dpa_ctta.r9_current_first.gradient_calibration import select
from dpa_ctta.r9_current_first.gradient import STEPS,BN_LR,LATENT_LR
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.schedule import CURRICULA

class Lifecycle(unittest.TestCase):
    def test_calibration_fresh_256_and_replay(self):
        args=fit_fixture();s,m,data,_=args
        o=Oracles('cal',.3,torch.zeros(1024,128),(tuple(data.folds['cal'][:2]),)*128,(tuple(data.folds['cal'][2:4]),)*128,((),)*128)
        m.freeze();a=Calibrator(s,m,data,o,'id');b=Calibrator(s,m,data,o,'id')
        self.assertEqual(a.optimizer.state,{})
        a.step();snap=a.snapshot();a.step();expected=a.method.cal_raw.clone();a.restore(snap);a.step()
        self.assertTrue(torch.equal(expected,a.method.cal_raw));self.assertEqual(b.steps,0)
        a.run();self.assertEqual(a.steps,256)
        with self.assertRaises(ValueError):a.step()
        self.assertTrue(torch.equal(b.method.cal_raw,m.cal_raw))

    def test_online_seal_cpu_score_retire(self):
        h=host_fixture();image=torch.rand(1,3,512,512);mask=torch.zeros(1,2,512,512)
        mask[:,:,100:200,100:200]=1
        rows=[dict(group_id=str(i),domain='d',subset='remaining_dev',image_path='synthetic',image_sha256='1'*64,image_size=[512,512],mask_path='never-online') for i in range(2)]
        payload=dict(spec_sha256=SPEC_SHA,sources={str(i):'bound' for i in range(43)})
        lock=dict(schema='R9_SOURCE_LOCK_V1',payload=payload,sha256=digest(payload));reads=[]
        class Reader:
            def __init__(self,root,cap,kind):self.kind=kind
            def read(self,row):
                reads.append(self.kind)
                if self.kind=='image':
                    assert 'mask_path' not in row
                    return image
                return mask
            def after_check(self):pass
        with tempfile.TemporaryDirectory() as d,patch.object(target,'TargetReader',Reader):
            root=Path(d)/'job';target.online(h,rows,None,root,'test',lock,lambda:None)
            self.assertEqual(reads,['image','image'])
            receipt=target.score(rows,None,root,'test',h.context['sha256'],lock,lambda:None)
            self.assertEqual(receipt['principal'],2);self.assertEqual(reads.count('mask'),2)
            target.retire_probabilities(root);self.assertFalse((root/'predictions.bits').exists())
            self.assertTrue((root/'scalars.private.jsonl').exists());target.retire_probabilities(root)

    def test_scoring_recovery_retains_tail_and_after_read_audit(self):
        h=host_fixture();image=torch.rand(1,3,512,512);mask=torch.zeros(1,2,512,512)
        rows=[dict(group_id=str(i),domain='d',subset='remaining_dev',image_path='synthetic',image_sha256='1'*64,image_size=[512,512]) for i in range(2)]
        payload=dict(spec_sha256=SPEC_SHA,sources={str(i):'bound' for i in range(43)})
        lock=dict(schema='R9_SOURCE_LOCK_V1',payload=payload,sha256=digest(payload))
        class Reader:
            def __init__(self,root,cap,kind):self.kind=kind
            def read(self,row):return image if self.kind=='image' else mask
            def after_check(self):pass
        with tempfile.TemporaryDirectory() as d,patch.object(target,'TargetReader',Reader):
            root=Path(d)/'one';target.online(h,rows,None,root,'j',lock,lambda:None)
            count=[0]
            def fail():
                count[0]+=1
                if count[0]==2:raise OSError('synthetic interruption')
            with self.assertRaises(OSError):target.score(rows,None,root,'j',h.context['sha256'],lock,fail)
            self.assertFalse((root/'score_complete.json').exists())
            target.score(rows,None,root,'j',h.context['sha256'],lock,lambda:None,{'class':'INFRASTRUCTURE','reason':'synthetic I/O','evidence':'test'})
            self.assertEqual(len((root/'scalars.private.jsonl').read_text().splitlines()),2)
            self.assertEqual(len((root/'score_uncommitted_tail.private.jsonl').read_text().splitlines()),1)
            other=Path(d)/'two';h2=host_fixture();target.online(h2,rows,None,other,'k',lock,lambda:None)
            class Bad(Reader):
                def after_check(self):raise ValueError('identity changed after reading')
            with patch.object(target,'TargetReader',Bad):
                with self.assertRaises(ValueError):target.score(rows,None,other,'k',h2.context['sha256'],lock,lambda:None)
            self.assertFalse((other/'score_complete.json').exists())

    def test_long10_and_mixed_ties(self):
        base=[dict(group_id=str(i),subset='remaining_dev' if i<1695 else 'legacy_dev',manifest_index=1951-i) for i in range(1951)]
        with patch.object(streams,'stream',return_value=base):
            long=streams.sequence({},'LONG10');self.assertEqual(len(long),19510)
            visits=streams.visits(long);self.assertEqual(visits[1951]['visit'],1952);self.assertEqual(visits[1951]['content'],visits[0]['content'])
            class Hash:
                def digest(self):return b'fixed'
            with patch.object(streams,'registration_digest',return_value='registration'),patch.object(streams.hashlib,'sha256',return_value=Hash()):
                mixed=streams.sequence({},'MIXED');self.assertEqual(mixed[0]['manifest_index'],1)

    def test_ledger_soft_time_and_idempotent_retirement(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger',{'id':1},[9]);ledger.create()
            budget=dict.fromkeys(CAPS,0);budget.update(gpu_seconds=.001,disk_bytes=1000)
            token=ledger.reserve('job',9,budget)
            actual=dict(budget,gpu_seconds=.1)
            ledger.observe('job',token,actual,settle=True)
            ledger.release_disk('job',900,'verified');ledger.release_disk('job',900,'verified')
            self.assertEqual(ledger.total(ledger._read())['disk_bytes'],100)
            token=ledger.reserve('second',9,budget)
            b=Budget(ledger,'second',token,Path(d)/'job',budget,True)
            b.cost['disk_bytes']=20;b.observe(dict.fromkeys(CAPS,0));self.assertEqual(b.cost['disk_bytes'],20)

    def test_lr_policy_is_explicit_and_balanced(self):
        rows=[]
        for a in STEPS:
            for lr in (BN_LR if a=='BN_RESET_G1' else LATENT_LR):
                for seed in (20260924,20260925):
                    rows += [dict(arm=a,lr=lr,source_seed=seed,episode=i,visits=4,curriculum=CURRICULA[i//16],hard_Dice=.7) for i in range(64)]
        with self.assertRaises(ValueError):select(rows,None)
        result=select(rows,'first_two_mean')
        self.assertEqual(result['selected_lr']['B_G1']['global'],.001)
        self.assertEqual(result['selected_lr']['BN_RESET_G1']['global'],1e-5)

if __name__=='__main__':unittest.main()
