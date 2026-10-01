import unittest,copy
from types import SimpleNamespace
import torch
from dpa_ctta.r8_ba.methods import film,R8B
from dpa_ctta.r9_current_first.deployment import Segmenter
from dpa_ctta.r10_carrier.diagnostic import ReadoutSegmenter,CONDITIONS,resolved,equal_state
from dpa_ctta.r10_carrier.run import lock,check_lock

class Carrier(unittest.TestCase):
    def test_repaired_config_keeps_exclusive_queue_and_previous_charge(self):
        import tempfile
        from dpa_ctta.r10_carrier.run import supervisor_lease,charged_wall
        with tempfile.TemporaryDirectory() as root:
            a=dict(output_root=root,code_sha='old');b=dict(output_root=root,code_sha='new')
            with supervisor_lease(a):
                with self.assertRaises(BlockingIOError):
                    with supervisor_lease(b):pass
            with supervisor_lease(b):pass
        c=dict(renewal=dict(previous_charged_seconds=2067.4,started_epoch=1000))
        self.assertAlmostEqual(charged_wall(c,1060),2127.4)
    def test_exact_readout_endpoints_and_nonlinear_location(self):
        torch.manual_seed(8);h=torch.randn(1,256,2,2);v=torch.randn(512)*2
        for a in (0.,.25,1.):
            obj=SimpleNamespace(alpha=a,amplitude=.3,diagnostics={})
            out=Segmenter._film(obj,h,v,'site');native=film(h,v,.3)
            self.assertTrue(torch.equal(out,h if a==0 else native if a==1 else h+a*(native-h)))
            if a==.25:self.assertFalse(torch.allclose(out,film(h,a*v,.3)))
    def test_native_full_and_reset_states_ignore_readout(self):
        torch.manual_seed(4);basis=torch.linalg.qr(torch.randn(1024,64,dtype=torch.float64)).Q
        m=R8B(basis,.3,'global');m.observer.fitted.fill_(True);m.freeze()
        states=[m.initial() for _ in range(3)]
        for _ in range(4):
            raw=torch.randn(134);tokens=torch.randn(64,64)
            states=[m.update(raw,tokens,s)[0] for s in states]
            self.assertTrue(all(equal_state(states[0],s) for s in states[1:]))
        prior=copy.deepcopy(states[0]);a=m.update(raw,tokens,prior,ablation='RESET_HISTORY')[0];b=m.update(raw,tokens,m.initial())[0]
        self.assertTrue(torch.equal(a['z'],b['z']));self.assertTrue(torch.equal(a['d'],b['d']));self.assertEqual(a['counter'],prior['counter']+1)
    def test_intervention_identity_and_forbidden_alpha(self):
        c=dict(bindings={'checkpoint_sha256':'x'},parent={'artifact':'bound'})
        for name in CONDITIONS:
            r=resolved(name);x=lock(c,name);check_lock(x)
            x['payload']['resolved']['output_alpha']=.5
            with self.assertRaises(ValueError):check_lock(x)
            if 'RESET' in name:self.assertEqual(r['arm'],'B_CARRIER_FULL_RESET')
        with self.assertRaises(ValueError):resolved('B_ALPHA_05')
    def test_factorial_pairing_and_interaction(self):
        from dpa_ctta.r10_carrier.report import interaction,report
        rows={}
        for name,v in [('B_FULL_READOUT_025',.6),('B_FULL_1',.3),('B_RESET_READOUT_025',.7),('B_RESET_1',.5)]:
            rs=[]
            for i,d in enumerate((0,0,1,2,3)):
                rs.append(dict(visit=i+1,content=str(i),domain=d,subset='remaining_dev',metrics=[dict(channel='OD',dice=v),dict(channel='OC',dice=v)]))
            rows[name,0]=rs
        self.assertAlmostEqual(interaction(rows,0)['delta'],.1)
        rows['B_FULL_1',0][0]['content']='wrong'
        with self.assertRaises(ValueError):interaction(rows,0)
        with self.assertRaises(ValueError):report({},dict(status='RUNNING',jobs={}))
    def test_source_trace_handles_native_one_based_visit(self):
        from dpa_ctta.r10_carrier.source import source_trace
        native=dict(visit=1,state_committed=True,counts={'forwards':1},B_state_sha256='private state',output_alpha=.25)
        row=source_trace('B_FULL_READOUT_025',16,0,'mode',native)
        self.assertEqual(row['visit'],0);self.assertEqual(row['output_alpha'],.25)
        self.assertNotIn('counts',row);self.assertEqual(native['visit'],1)
