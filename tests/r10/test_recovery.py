import errno,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.r10_use_write_rl.protocol import CAPS,graph,digest
from dpa_ctta.r10_use_write_rl.ledger import Ledger
from dpa_ctta.r10_use_write_rl.runtime import worker,read
from dpa_ctta.r10_use_write_rl.queue import reconcile
from dpa_ctta.r10_use_write_rl.target import score,retire_probabilities
from dpa_ctta.r10_use_write_rl.recovery import verify_failure
from dpa_ctta.r9_current_first.storage import write_json,write_torch,load_torch,PhaseJournal
from dpa_ctta.r8_ba.journal import TargetJournal
from dpa_ctta.r8_ba.streams import rows_sha


class FakeHost:
    def __init__(self):self.visits=0;self.context={'sha256':'a'*64};self.state=0
    def check_frozen(self,boundary=False):pass
    def snapshot(self):return dict(visits=self.visits,state=self.state)
    def restore(self,s):self.visits=s['visits'];self.state=s['state']
    def step(self):self.state+=1;self.visits+=1;return dict(visit=self.visits,state_committed=True)


class MaskReader:
    def __init__(self,*a):assert a[-1]=='mask'
    def read(self,row):return torch.zeros(1,2,512,512)
    def after_check(self):pass


class Recovery(unittest.TestCase):
    def test_actual_worker_receipt_restart_recovery_score_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity={'test':'R10'};ledger=Ledger(root/'ledger',identity,[5]);ledger.create()
            job=dict(id='toy',arrivals=2);node=dict(id='toy__online',kind='online',job=job)
            config=dict(output_root=str(root),gpu_assignments=[{'physical_id':5}],bindings={})
            budget=dict.fromkeys(CAPS,0);budget['disk_bytes']=8*1024**2
            token=ledger.reserve(node['id']+'.0',None,budget)
            rows=[dict(domain='d',subset='remaining_dev',group_id=str(i)) for i in range(2)]
            lock={'sha256':'b'*64};path=root/'target'/'toy';raw=torch.zeros(1,2,512,512).numpy().tobytes()
            def first(*args):
                host=FakeHost();j=TargetJournal(path,host,'toy',rows_sha(rows),len(raw));j.create();j.append(raw,host.step())
                raise OSError(errno.EIO,'injected after uncommitted prediction')
            with patch('dpa_ctta.r10_use_write_rl.runtime.require',return_value=identity),patch('dpa_ctta.r10_use_write_rl.runtime.dispatch',side_effect=first):
                failed=worker(config,node,None,node['id']+'.0',token)
            state=dict(nodes={node['id']:dict(status='RUNNING',attempt=node['id']+'.0')})
            # Same transition after an actual queue-state reload (worker already settled).
            write_json(root/'queue.json',state);state=read(root/'queue.json');reconcile(root,state,node,failed,identity,ledger)
            self.assertEqual(state['nodes'][node['id']]['status'],'RETRY')
            failure=state['nodes'][node['id']]['failure'];verify_failure(root,path,node['id'],identity,failure)
            token=ledger.reserve(node['id']+'.1',None,budget)
            def second(*args):
                host=FakeHost();j=TargetJournal(path,host,'toy',rows_sha(rows),len(raw));j.recover_once(failure)
                self.assertEqual(host.visits,0)
                for _ in rows:j.append(raw,host.step())
                result=j.complete(2);write_json(path/'source_lock.json',lock);return result
            with patch('dpa_ctta.r10_use_write_rl.runtime.require',return_value=identity),patch('dpa_ctta.r10_use_write_rl.runtime.dispatch',side_effect=second):
                done=worker(config,node,None,node['id']+'.1',token,failure)
            self.assertEqual(done['status'],'COMPLETE');state['nodes'][node['id']]=dict(status='RUNNING',attempt=node['id']+'.1');reconcile(root,state,node,done,identity,ledger)
            occupied=read(ledger.root/'state.json')['occupied_disk_bytes']
            with patch('dpa_ctta.r10_use_write_rl.target.check_lock'),patch('dpa_ctta.r10_use_write_rl.target.TargetReader',MaskReader):
                score(rows,'unused',path,'toy','a'*64,lock,lambda:None)
            retire_probabilities(path);ledger.sync_disk('retired')
            self.assertFalse((path/'predictions.bits').exists());self.assertLess(read(ledger.root/'state.json')['occupied_disk_bytes'],occupied)
            ledger.reserve('next.0',None,budget)
            self.assertFalse(ledger.claim_recovery('toy','toy__score.0','new'))
            self.assertTrue(ledger.claim_recovery('other1','x.0','x'));self.assertTrue(ledger.claim_recovery('other2','y.0','y'));self.assertFalse(ledger.claim_recovery('other3','z.0','z'))
    def test_own_graph(self):
        ns=graph();self.assertEqual(sum(n['kind']=='train' for n in ns),25);self.assertEqual(sum(n['kind']=='online' for n in ns),410)
        self.assertEqual(sum(n['kind']=='score' for n in ns),410)
        self.assertTrue(all('SOURCE_LOCK' in n['needs'] for n in ns if n['kind']=='online'))


class TerminalBranches(unittest.TestCase):
    def test_failure_selection_and_independent_work(self):
        from dpa_ctta.r10_use_write_rl.queue import ready_nodes
        from dpa_ctta.r10_use_write_rl.evaluation import choose_families
        from dpa_ctta.r10_use_write_rl.target import lock_sources,check_lock
        from dpa_ctta.r10_use_write_rl.protocol import SPEC
        from tempfile import TemporaryDirectory
        from pathlib import Path
        rows={f'FIT_SUP_SEQ_{s}':{'selection':{'S':.6}} for s in (20260924,20260925)}
        selected=choose_families(rows);self.assertEqual(selected['selected'],{'SUP':'SUP_SEQ','GR':None})
        with TemporaryDirectory() as d:
            p=Path(d);write_json(p/'selection.json',selected)
            state={'nodes':{'failed':{'status':'FAILED'},'ok':{'status':'COMPLETE'}}}
            nodes=[dict(id='select',kind='select',needs=['failed','ok']),dict(id='dependent',kind='d0',needs=['failed']),dict(id='independent',kind='d0',needs=['ok'])]
            self.assertEqual([n['id'] for n in ready_nodes(p,state,nodes)],['select','independent'])
            self.assertEqual(state['nodes']['dependent']['status'],'BLOCKED')
        statuses={j['id']:'FAILED' for j in SPEC['source_jobs']}
        check_lock(lock_sources({},selected,statuses))
