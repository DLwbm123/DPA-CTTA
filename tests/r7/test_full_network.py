import copy,json,unittest
import torch
from common import method,segmenter,pixels,source_fixture
from dpa_ctta.r7_shared.host import OnlineHost
from dpa_ctta.r7_shared.source import SourceTrainer,pooled_jacobian
from dpa_ctta.r7_shared.numerics import COUNTS

class FullNetwork(unittest.TestCase):
    def test_A_B_C_actual_ResUNet_visits_and_source_microepisodes(self):
        data,o=source_fixture()
        for group in 'ABC':
            s=segmenter(full=True);m=method(group);before_weights={k:v.clone() for k,v in s.model.state_dict().items()}
            t=SourceTrainer(s,m,data,o['fit']);before=COUNTS.copy();t.fit_step()
            fit_counts=dict(COUNTS-before);self.assertEqual(fit_counts['backbone_forwards'],12);self.assertEqual(fit_counts['source_backward_calls'],1)
            expected={'A':['W','qraw','style','content','head.0.weight'],'B':['W','G','Hraw','head.0.weight','bias.0.weight'],'C':['Pc','Pa','O','query.0.weight','mapping.0.weight']}[group]
            for n in expected:self.assertGreater(t.gradients[n],0,n)
            t.start_calibration(o['cal'],procedural_micro=True);t.cal_step();m.freeze()
            host=OnlineHost(s,m);before=COUNTS.copy()
            for i in range(2):host.step(pixels(i))
            online=dict(COUNTS-before);self.assertEqual(online['backbone_forwards'],4);self.assertNotIn('source_backward_calls',online)
            for k,v in s.model.state_dict().items():self.assertTrue(torch.equal(v,before_weights[k]),k)
            self.assertFalse(any(p.grad is not None for p in s.model.parameters()));self.assertFalse(s.cache)
            print('FULL_NETWORK '+json.dumps(dict(group=group,parameters=sum(p.numel() for p in s.model.parameters()),method_parameters=sum(p.numel() for p in m.parameters()),fit=fit_counts,online=online,gradients=t.gradients)),flush=True)
            s.close();del host,t,s,m,before_weights
    def test_full_zero_identity_derivative_and_C0(self):
        s=segmenter(full=True);x=pixels();before=COUNTS.copy()
        with torch.no_grad():a=s(x)
        v=torch.zeros(1024,requires_grad=True);b=s(x,v);self.assertTrue(torch.equal(a,b))
        readout=torch.nn.functional.adaptive_avg_pool2d(b,(4,4)).flatten()
        # A-source Jacobian plumbing on two readouts; actual 1024-VJP source basis is future work.
        for i in range(2):
            g=torch.autograd.grad(readout[i],v,retain_graph=i==0)[0];COUNTS['source_VJP']+=1
            self.assertGreater(float(g.norm()),0);self.assertGreater(float(g[:512].norm()),0);self.assertGreater(float(g[512:].norm()),0)
        _,trace=OnlineHost(s).step(x);self.assertEqual(trace['counts']['backbone_forwards'],1)
        print('FULL_IDENTITY '+json.dumps(dict(COUNTS-before)),flush=True);s.close()
    def test_original_C_random_network_narrow_preservation(self):
        from dpa_ctta.r1.host import Host as Old
        from dpa_ctta.r6_regional_consistency.host import Host as Current
        from dpa_ctta.integrations.ctta_suite import build_reference_model
        from dpa_ctta.b4_run import capture
        from dpa_ctta.host_diagnostic import close
        from dpa_ctta.source_pilot import seed_all
        seed_all(20260907);model,_=build_reference_model('fundus');weights=copy.deepcopy(model.state_dict());del model
        seed_all(20260907);a=Old('C',weights,'cpu');za,ta=a.step(pixels());sa=capture(a);a.finish(weights);del a
        seed_all(20260907);b=Current('C',weights,'cpu');zb,tb=b.step(pixels());b.take_evaluation().clear()
        close(za,zb,exact=True);close(sa,capture(b.core),exact=True)
        self.assertEqual(tb['counts'],dict(network_forwards=8,loss_backward_calls=1,adam_calls=1,jacobian_vjp_calls=0));b.finish(weights)
        print('OLD_C_PRESERVATION '+json.dumps(dict(network_forwards=16,backward_calls=2,Adam=2,bitwise=True)),flush=True)
