import copy
import unittest
import torch
import torch.nn.functional as F
from dpa_ctta.b2_interval import radius,targets,interval_kl
from dpa_ctta.b2_host import Host
from dpa_ctta.b1_host import Host as CHost
from dpa_ctta.b2_analysis import validate_rows,summarize,ARMS,evaluate
from dpa_ctta.host_diagnostic import rng,close
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.integrations.ctta_suite import build_reference_model
from test_b1 import Tiny


class B2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_zero_radius_gradient_and_entropy(self):
        z=torch.tensor([-2.,.3,1.],dtype=torch.double,requires_grad=True);q=torch.tensor([.2,.7,.6],dtype=torch.double)
        loss,_,_,_=interval_kl(z,q,q);bce=F.binary_cross_entropy_with_logits(z,q)
        close(torch.autograd.grad(loss,z,retain_graph=True)[0],torch.autograd.grad(bce,z)[0],exact=True)
        self.assertLess(float(loss),float(bce))

    def test_projection_inside_outside_extremes_and_detach(self):
        z=torch.tensor([-2.,0.,2.],dtype=torch.double,requires_grad=True);lo=torch.full_like(z,.3,requires_grad=True);hi=torch.full_like(z,.7,requires_grad=True)
        loss,v,active,_=interval_kl(z,lo,hi);loss.backward()
        close(z.grad,torch.where(active,(z.detach().sigmoid()-v)/3,0));self.assertEqual(float(z.grad[1]),0);self.assertIsNone(lo.grad);self.assertIsNone(hi.grad);self.assertFalse(v.requires_grad)
        inside=torch.zeros(3,requires_grad=True);zero,*_=interval_kl(inside,lo,hi);zero.backward();self.assertEqual(float(zero),0);self.assertTrue((inside.grad==0).all())
        z=torch.tensor([-1000.,1000.,-1000.,1000.],requires_grad=True);q=torch.tensor([0.,1.,1.,0.]);loss,*_=interval_kl(z,q,q);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(z.grad).all())

    def test_gradcheck_and_population_targets(self):
        z=torch.tensor([-2.,.1,2.],dtype=torch.double,requires_grad=True);lo=torch.full_like(z,.3);hi=torch.full_like(z,.7)
        self.assertTrue(torch.autograd.gradcheck(lambda x:interval_kl(x,lo,hi)[0],(z,)))
        p=torch.linspace(.05,.95,6*2*3*3).reshape(6,1,2,3,3).requires_grad_();q,s,r,lo,hi=targets(p,'I',1)
        close(q,p.detach().mean(0),exact=True);close(s,p.detach().std(0,unbiased=False));self.assertTrue(all(not t.requires_grad for t in (q,s,r,lo,hi)))

    def test_partition_mean_multiset_empty_singleton_and_RNG(self):
        q=torch.tensor([[[[.1,.2,.4],[.7,.8,.9]],[[.1,.1,.1],[.1,.1,.8]]]])
        s=torch.arange(12,dtype=torch.float32).reshape_as(q)/24
        state=rng();u=radius(q,s,'U',1);shuffled=radius(q,s,'S',1);close(state,rng(),exact=True)
        close(shuffled,radius(q,s,'S',1),exact=True)
        for c in range(2):
            for b in (False,True):
                m=(q[0,c]>=.5)==b
                close(u[0,c][m],s[0,c][m].mean().expand_as(s[0,c][m]),exact=True)
                close(s[0,c][m].sort().values,shuffled[0,c][m].sort().values,exact=True)
        for arm in ('U','S'):
            close(radius(torch.zeros_like(q),torch.zeros_like(s),arm,1),torch.zeros_like(s),exact=True)
        self.assertFalse(torch.equal(shuffled,s))

    def test_adam_zero_new_gradient_retains_momentum(self):
        w=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.Adam([w],lr=1e-4);w.grad=torch.ones_like(w);opt.step();before=w.detach().clone();opt.zero_grad();(w*0).sum().backward();opt.step();self.assertFalse(torch.equal(w,before));self.assertEqual(int(opt.state[w]['step']),2)

    def test_actual_model_zero_C_regression(self):
        seed_all(20260907);model,_=build_reference_model('fundus');state=copy.deepcopy(model.state_dict());del model
        x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        seed_all(20260907);c=CHost('C',state);p,_=c.step(x)
        seed_all(20260907);z=Host('zero',state);q,m=z.step(x)
        close(p,q);close(c.base.state_dict(),z.base.state_dict());close(c.rng,z.rng,exact=True);close([v for v in c.params],[v for v in z.params]);self.assertEqual(len(z.names),82);self.assertEqual(sum(p.numel() for p in z.params),19136)
        c.finish(state);z.finish(state)

    def test_labels_future_state_and_frozen_parameters(self):
        for arm in ('U','S','I'):
            seed_all(20260907);model=Tiny();initial=copy.deepcopy(model.state_dict());h=Host(arm,model=copy.deepcopy(model));other=Host(arm,model=copy.deepcopy(model))
            for i in range(3):
                x=torch.linspace(.01,.99,3*12*12).reshape(1,3,12,12).roll(i,-1);p,m=h.normalized_step(x);q,_=other.normalized_step(x)
                close(p,q,exact=True);close(h.base.state_dict(),other.base.state_dict(),exact=True);close(h.rng,other.rng,exact=True)
                before=rng();a=evaluate(p.sigmoid(),torch.zeros_like(p),'fundus');b=evaluate(q.sigmoid(),torch.ones_like(q),'fundus');self.assertNotEqual(a,b);close(before,rng(),exact=True)
                self.assertEqual(m['counts'],dict(forwards=8,backwards=1,base_adam=1,perturb=0,restore=0))
            h.finish(initial);other.finish(initial)

    def test_scalar_validation_and_summary(self):
        seed_all(20260907);h=Host('I',model=Tiny());rows=[];stream=[]
        for i in range(2):
            p,m=h.normalized_step(torch.linspace(.01,.99,3*12*12).reshape(1,3,12,12));metrics=evaluate(p.sigmoid(),torch.zeros_like(p),'fundus')
            identity=dict(task='fundus',order=0,arm='I',visit=i+1,domain='REFUGE',subset='remaining_dev',sample_id=str(i),group_id=str(i));stream.append(identity)
            rows.append(dict(identity,**m,metrics=metrics,prediction_fixed_before_label=True,frozen_parameters_checked=True))
        self.assertTrue(validate_rows(rows,stream,0,'I',rows))
        for kind in ('count','state','metric','diagnostic','missing'):
            bad=copy.deepcopy(rows)
            if kind=='count':bad[0]['counts']['forwards']=9
            elif kind=='state':bad[0]['adam_step']=2
            elif kind=='metric':bad[0]['metrics'][0]['dice']+=.1
            elif kind=='diagnostic':bad[0]['diagnostics']['channels'][0]['partitions']['0']['active']=-1
            else:bad.pop()
            with self.assertRaises(ValueError):validate_rows(bad,stream,0,'I',rows)
        r=summarize({a:rows for a in ARMS});self.assertEqual(r['task_comparisons_pp']['I-C'],0);self.assertNotIn('assd_conditional_mean_px',r['domains']['REFUGE']['arms']['I']['macro'])

    def test_complete_CPU_reconstruction(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from dpa_ctta.b2_analysis import recompute,COMPLETE
        from dpa_ctta.p2_data import SUBSETS
        h=Host('I',model=Tiny());p,m=h.normalized_step(torch.linspace(.01,.99,3*12*12).reshape(1,3,12,12));metrics=evaluate(p.sigmoid(),torch.zeros_like(p),'fundus')
        stream=[]
        for domain in ('REFUGE','ORIGA','REFUGE_Valid','Drishti_GS'):
            for subset in SUBSETS:
                if subset=='all_dev':continue
                i=len(stream)+1;stream.append(dict(domain=domain,subset=subset,sample_id=str(i),group_id=str(i)))
        n=len(stream);totals=dict(records=n*6,forwards=n*48,backwards=n*6,base_adam=n*6,perturb=0,restore=0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();old=root/'old';old.mkdir()
            def write(name,v): (out/name).write_text(json.dumps(v))
            (old/'execution_audit.json').write_text(json.dumps(dict(cost={})))
            write('run.completion.json',dict(status='B2_RUN_COMPLETE',progress=totals,trainable={},gpu_seconds=1))
            write('smoke.completion.json',dict(status='B2_SMOKE_PASS',base_adam=20,evidence={},gpu_seconds=1,physical=dict(forwards=160,backwards=20,base_adam=20,perturb=0,restore=0),exit_code=0))
            write('CPU_validation.json',dict(exit_code=0))
            for order in (0,1):
                for arm in ('U','S','I'):
                    rows=[dict(e,task='fundus',order=order,arm=arm,visit=i+1,**dict(m,adam_step=i+1),metrics=metrics,prediction_fixed_before_label=True,frozen_parameters_checked=True,host_seconds=.1,pipeline_seconds=.2,peak_allocated_bytes=1) for i,e in enumerate(stream)]
                    (out/f'fundus_{order}_{arm}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            controls={a:[dict(e,metrics=metrics) for e in stream] for a in ('N','A','EA','O2','D4','C','G')}
            reg=dict(formal_budget=totals,tasks=dict(fundus=dict(counts={})),limitations=[],p2_directory=str(old),b1_directory=str(old))
            with patch('dpa_ctta.b2_analysis.expected',return_value=stream),patch('dpa_ctta.b2_analysis.old_records',side_effect=lambda *args:copy.deepcopy(controls)):
                recompute(out,reg,dict(commit='CPU_FIXTURE'))
            report=json.loads((out/'public_aggregate.json').read_text());self.assertEqual(report['status'],COMPLETE);self.assertEqual(report['prespecified_interpretation']['two_order_mean_I_minus_C_pp'],0)
            self.assertEqual(json.loads((out/'execution_audit.json').read_text())['formal'],totals)
