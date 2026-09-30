import copy,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.r10_12h_core.run import subset,resolved,terminate_group,VAL,check_lock,sha
from dpa_ctta.r10_use_write_rl.learning import Trainer,lr
from dpa_ctta.r10_use_write_rl.controller import Actor,Controller,Host
from test_connected import carrier,ToySource,ToySegmenter

class Core(unittest.TestCase):
    def test_subset_routes_and_validation(self):
        rows=[dict(group_id=str(i),domain=str(i//25),subset='remaining_dev' if i%4 else 'screen') for i in range(100)]
        a,b=subset([rows,list(reversed(rows))],52,'registered')
        self.assertEqual(len(a),52);self.assertEqual(a,list(reversed(b)))
        self.assertEqual(a,subset([rows],52,'registered')[0]);self.assertEqual(len({r['group_id'] for r in a}),52)
        self.assertEqual([sum(r['domain']==str(d) for r in a) for d in range(4)],[13]*4)
        artifact={'file':'actor.pt','sha256':'a'*64}
        p=resolved(dict(arm='GR_RET_EMA',seed=20260924),artifact)
        q=resolved(dict(arm='GR_RET_EMA_CONST_HALF',seed=20260924),artifact)
        self.assertEqual({k:v for k,v in p.items() if k!='diagnostic'},{k:v for k,v in q.items() if k!='diagnostic'})
        self.assertEqual(q['diagnostic'],'CONST_HALF')
        self.assertEqual(sorted(i//16 for i in VAL),[0]*4+[1]*4+[2]*4+[3]*4)
        with self.assertRaises(ValueError):check_lock({'schema':'R10_SOURCE_LOCK_V1','payload':{}})
    def test_short_schedule_exact_restore_and_state_isolation(self):
        c=Controller(carrier(),torch.ones(64,dtype=torch.float64));source=ToySource(c);a=Actor(3)
        t=Trainer(a,c,source,3,'GR_RET_EMA','core',total=512)
        t.step();snap=t.snapshot();r=t.step();weights=copy.deepcopy(a.state_dict());t.restore(snap)
        self.assertEqual(r,t.step())
        for k,v in weights.items():self.assertTrue(torch.equal(v,a.state_dict()[k]))
        wrong=Trainer(Actor(3),c,source,3,'GR_RET_EMA','core',total=1024)
        with self.assertRaises(ValueError):wrong.restore(snap)
        self.assertAlmostEqual(lr(511,512,3e-5,3e-6),3e-6)
        h0=Host(ToySegmenter(),c,a,'o0');h1=Host(ToySegmenter(),c,a,'o1');before=h1.snapshot()
        h0.step(torch.randn(134))
        for k,v in before['memory'].items():self.assertTrue(torch.equal(v,h1.snapshot()['memory'][k]))
        self.assertEqual(h1.visits,0)
    def test_deadline_covers_child_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pth=Path(tmp)/'child'
            code='import subprocess,time,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); open(sys.argv[1],"w").write(str(p.pid)); time.sleep(60)'
            p=subprocess.Popen([sys.executable,'-c',code,str(pth)],start_new_session=True)
            try:
                until=time.time()+5
                while not pth.exists() and time.time()<until:time.sleep(.02)
                child=int(pth.read_text());terminate_group(p);self.assertIsNotNone(p.poll())
                # Linux orphan may be a zombie until PID 1 reaps it; it is not executing.
                stat=Path(f'/proc/{child}/stat')
                if stat.exists():self.assertEqual(stat.read_text().split()[2],'Z')
                else:
                    with self.assertRaises(ProcessLookupError):os.kill(child,0)
            finally:
                if p.poll() is None:terminate_group(p)
