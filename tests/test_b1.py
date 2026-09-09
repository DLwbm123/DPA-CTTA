import copy
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from dpa_ctta.b1_host import Host,official,configure,reference_step,finite
from dpa_ctta.host_diagnostic import rng,restore,close
from dpa_ctta.source_pilot import seed_all,SourceOnlyHost
from dpa_ctta.integrations.ctta_suite import build_reference_model
from dpa_ctta.b1_analysis import validate_rows,evaluate,summarize
from dpa_ctta.m1_run import new_log
from dpa_ctta.host_diagnostic_run import append


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,4,1);self.bn=nn.BatchNorm2d(4);self.head=nn.Conv2d(4,2,1)
    def forward(self,x):
        feature=self.bn(self.conv(x)).relu();return self.head(feature),feature


class B1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):official();torch.set_num_threads(2)

    def test_source_parity_and_strict_mapping(self):
        model,_=build_reference_model('fundus');state=copy.deepcopy(model.state_dict())
        source=SourceOnlyHost('fundus',state);x=torch.rand(1,3,512,512)
        from dpa_ctta.hosts.vptta import model_input_from_pixels
        with torch.no_grad():close(source.step(x),model(model_input_from_pixels(x,'fundus'))[0],exact=True)
        broken=state.copy();broken.pop(next(iter(broken)))
        with self.assertRaises(ValueError):SourceOnlyHost('fundus',broken)

    def test_six_views_and_inverse(self):
        x=torch.arange(27).reshape(1,3,3,3);aug=official().Rotate_and_Flip();views=[x]
        for f in range(5):
            v=aug(x,f);self.assertTrue(torch.equal(aug.inverse(v,f),x));views.append(v)
        self.assertEqual(len({tuple(v.flatten().tolist()) for v in views}),6)

    def test_published_paths_gradients_state_and_label_isolation(self):
        for arm in ['C','G']:
            seed_all(20260907);model=Tiny();initial=copy.deepcopy(model.state_dict());h=Host(arm,model=copy.deepcopy(model));other=Host(arm,model=copy.deepcopy(model))
            names,params=configure(model);base=torch.optim.Adam(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);opt=official().GraTa(params,base,model,device='cpu');ref_rng=rng()
            self.assertEqual(h.names,['bn.weight','bn.bias'])
            for i in range(4):
                x=torch.linspace(.01,.99,3*12*12).reshape(1,3,12,12).roll(i,-1)
                restore(ref_rng);ref=reference_step(model,opt,arm,x);ref_rng=rng()
                pred,meta=h.normalized_step(x);q,_=other.normalized_step(x)
                close(pred,ref);close(pred,q);close(h.model.state_dict(),model.state_dict());close(h.base.state_dict(),base.state_dict());close(h.rng,ref_rng,exact=True)
                close([p.grad for p in h.params],[p.grad for p in params])
                state=copy.deepcopy(h.model.state_dict());oldrng=rng()
                a=evaluate(pred.sigmoid(),torch.zeros_like(pred),'fundus');b=evaluate(pred.sigmoid(),torch.ones_like(pred),'fundus')
                self.assertNotEqual(a,b);close(state,h.model.state_dict(),exact=True);close(oldrng,rng(),exact=True)
                self.assertEqual(meta['adam_step'],i+1)
            h.finish(initial);other.finish(initial)
            for key in ['cal_groundtruth_loss','cal_recon_loss']:
                with self.assertRaises(ValueError):getattr(h.opt,key)({'mask':'forbidden'})
            with self.assertRaises(ValueError):h.step({'data':x,'mask':x})

    def test_G_analytic_perturb_restore_cosine_and_zero_gradient(self):
        for zero in [False,True]:
            w=nn.Parameter(torch.tensor([0.,0.] if zero else [.2,.7]));base=torch.optim.Adam([w],lr=1e-4,betas=(.9,.999),eps=1e-8);g=official().GraTa([w],base,nn.Identity(),device='cpu')
            original=w.detach().clone();trace=[]
            @torch.enable_grad()
            def aux(data):
                base.zero_grad();loss=w.square().sum()/2;loss.backward();trace.append('aux');return loss
            @torch.enable_grad()
            def pse(data):
                close(w,torch.zeros_like(w),exact=True);base.zero_grad();loss=(w-torch.tensor([2.,-1.])).square().sum();loss.backward();trace.append('pse');return loss
            g.cal_ent_loss=aux;g.cal_consis_loss=pse
            def check(opt,args,kw):
                trace.append('adam');close(w,original,exact=True);close(w.grad,torch.tensor([-4.,2.]),exact=True)
            hook=base.register_step_pre_hook(check);g.step({'data':None});hook.remove()
            pg=torch.tensor([-4.,2.]);cos=(original*pg).sum()/(original.norm()*pg.norm()+1e-12);lr=1e-4*(cos+1).square()/4
            ref=nn.Parameter(original.clone());adam=torch.optim.Adam([ref],lr=lr,betas=(.9,.999),eps=1e-8);ref.grad=pg;adam.step();close(w,ref);close(base.param_groups[0]['lr'],lr)
            self.assertEqual(trace,['aux','pse','adam'])
            if zero:self.assertAlmostEqual(float(lr),.000025,places=10)

    def test_coverage_state_metric_failure_and_prefix(self):
        seed_all(20260907);h=Host('C',model=Tiny());stream=[];rows=[];old=[]
        for i in range(3):
            p,m=h.normalized_step(torch.rand(1,3,12,12));mask=(torch.rand_like(p)>.5).float();metrics=evaluate(p.sigmoid(),mask,'fundus')
            row=dict(task='fundus',order=0,arm='C',visit=i+1,domain='REFUGE',subset='remaining_dev',sample_id=str(i),group_id=str(i));stream.append(row)
            rows.append(dict(row,**m,metrics=metrics,prediction_fixed_before_label=True,frozen_parameters_checked=True));old.append(dict(metrics=metrics))
        self.assertTrue(validate_rows(rows,stream,0,'C',old))
        for change in ['missing','duplicate','wrong_order','state','metric','nonfinite']:
            bad=copy.deepcopy(rows)
            if change=='missing':bad.pop()
            elif change=='duplicate':bad[1]=bad[0]
            elif change=='wrong_order':bad.reverse()
            elif change=='state':bad[0]['adam_step']=5
            elif change=='metric':bad[0]['metrics'][0]['dice']+=.1
            else:bad[0]['lr']=float('nan')
            with self.assertRaises(ValueError):validate_rows(bad,stream,0,'C',old)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prefix.jsonl'
            with new_log(path) as f:
                append(f,rows[0])
                with self.assertRaises(ValueError):finite(torch.tensor(float('nan')))
            self.assertEqual(len(path.read_text().splitlines()),1)
        with self.assertRaises(ValueError):
            h.model.conv.weight.add_(1);h.normalized_step(torch.rand(1,3,12,12))

    def test_matched_summary_and_undefined_ASSD(self):
        p=torch.zeros(1,2,3,3);metrics=evaluate(p,p,'fundus');self.assertIsNone(metrics[0]['assd'])
        arms={a:[dict(domain='REFUGE',metrics=metrics)] for a in ['N','A','EA','O2','D4','C','G']}
        result=summarize(arms);self.assertEqual(result['task_comparisons_pp']['G-C'],0.)
        self.assertNotIn('assd_conditional_mean_px',result['domains']['REFUGE']['arms']['G']['macro'])
