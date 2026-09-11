import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.b4_host import C0,PolypC,logits,reference_polyp_step
from dpa_ctta.b4_run import capture
from dpa_ctta.b4_data import map_rows,BUDGET
from dpa_ctta.b4_analysis import validate_rows,recompute,COMPLETE
from dpa_ctta.integrations.ctta_suite import build_reference_model
from dpa_ctta.b1_host import official
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.host_diagnostic import rng,close
from dpa_ctta.hosts.vptta import model_input_from_pixels
from dpa_ctta.p1_analysis import evaluate
from dpa_ctta.p2_data import SUBSETS
from test_vptta_host import pixels


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);official();cls.states={}
        for task in ('fundus','polyp'):
            seed_all(20260907);model,_=build_reference_model(task);cls.states[task]=copy.deepcopy(model.state_dict())

    def test_C0_stateless_and_tensor_adapter(self):
        for task,state in self.states.items():
            h=C0(task,state);saved={};before=rng()
            self.assertFalse(hasattr(h,'optimizer'));self.assertFalse(hasattr(h,'base'))
            for i in (0,1,1,0):
                z,m=h.step(pixels(task,i));self.assertEqual(z.shape,(1,2,512,512) if task=='fundus' else (1,1,352,352))
                if i in saved:close(z,saved[i],exact=True)
                else:saved[i]=z
            close(before,rng(),exact=True);self.assertEqual(h.counts['forwards'],4);self.assertEqual(h.counts['base_adam'],0);h.finish(state)

    def test_independent_polyp_reference_and_future_label_isolation(self):
        state=self.states['polyp'];seed_all(20260907);ref=PolypC(state);saved=[]
        for i in (0,1):
            z=reference_polyp_step(ref,pixels('polyp',i));saved.append((z,capture(ref)))
        ref.finish(state);del ref
        seed_all(20260907);h=PolypC(state)
        for i in (0,1):
            z,m=h.step(pixels('polyp',i));close(z,saved[i][0]);close(capture(h),saved[i][1]);close(rng(),saved[i][1]['rng'],exact=True)
            before=capture(h);a=evaluate(z.sigmoid(),torch.zeros_like(z),'polyp');b=evaluate(z.sigmoid(),torch.ones_like(z),'polyp');self.assertNotEqual(a,b);close(before,capture(h),exact=True)
            self.assertEqual(m['counts']['forwards'],8)
        h.finish(state);self.assertNotEqual(len(h.names)//2,41)
        print('PRANET_BN',len(h.names)//2,'AFFINE',sum(p.numel() for p in h.params))

    def test_weak_mean_detach_RGB_normalization_and_no_mutation(self):
        seed_all(20260907);h=PolypC(self.states['polyp']);x=pixels('polyp',0);before=x.clone();inputs=[];outputs=[];strong=[];targets=[]
        hook=h.model.register_forward_hook(lambda m,a,z:(inputs.append(a[0].detach().clone()),outputs.append(z.detach().clone())) and None)
        api=official();aug=api.augmentation_strong_style;bce=torch.nn.functional.binary_cross_entropy_with_logits
        def style(data):
            close(torch.from_numpy(data['data']),x,exact=True);r=aug(data);strong.append(torch.from_numpy(r.copy()));return r
        def loss(z,q,*a,**k):targets.append(q);return bce(z,q,*a,**k)
        with patch.object(api,'augmentation_strong_style',style),patch('dpa_ctta.b4_host.F.binary_cross_entropy_with_logits',side_effect=loss):h.step(x)
        hook.remove();close(x,before,exact=True);self.assertFalse(targets[0].requires_grad)
        weak=api.Rotate_and_Flip();q=torch.stack([outputs[0]]+[weak.inverse(outputs[i+1],i) for i in range(5)]).sigmoid().mean(0)
        close(q,targets[0],exact=True);close(inputs[0],model_input_from_pixels(x,'polyp'),exact=True)
        for i in range(5):close(inputs[i+1],model_input_from_pixels(weak(x,i),'polyp'),exact=True)
        rgb=(strong[0]-strong[0].min())/(strong[0].max()-strong[0].min());close(inputs[6],model_input_from_pixels(rgb,'polyp'),exact=True);close(inputs[7],inputs[0],exact=True)
        c0=C0('polyp',self.states['polyp']);z,_=c0.step(x);close(z,outputs[0]);h.finish(self.states['polyp'])

    def test_canonical_mapping_and_scalar_failure(self):
        rows=[dict(group_id=str(i),sample_id=str(i),domain='D',subset='remaining_dev') for i in range(3)]
        self.assertEqual(map_rows(rows,list(reversed(rows))),list(reversed(rows)))
        with self.assertRaises(ValueError):map_rows(rows[:-1],rows)
        bad=copy.deepcopy(rows);bad[0]['sample_id']='other'
        with self.assertRaises(ValueError):map_rows(bad,rows)
        self.assertEqual(BUDGET['records'],1951+1802+2*1802);self.assertEqual(BUDGET['forwards']+76,32661)

    def test_CPU_closeout_fixture(self):
        streams={t:[dict(group_id=str(i),sample_id=str(i),domain='D',subset=s) for i,s in enumerate(SUBSETS) if s!='all_dev'] for t in ('fundus','polyp')}
        def rows(task,arm,order):
            z=torch.zeros(1,2 if task=='fundus' else 1,3,3);metrics=evaluate(z,z,task);zero=arm=='C0'
            return [dict(r,task=task,arm=arm,order=order,visit=i+1,adam_step=0 if zero else i+1,counts=dict(forwards=1 if zero else 8,backwards=int(not zero),base_adam=int(not zero),perturb=0,restore=0),lr=None if zero else 1e-4,prediction_origin='canonical_stateless' if zero else 'continuous_update',stateless_checked=zero,prediction_fixed_before_label=True,frozen_parameters_checked=True,metrics=copy.deepcopy(metrics),host_seconds=.1,pipeline_seconds=.2,peak_allocated_bytes=1) for i,r in enumerate(streams[task])]
        def old(reg,task,o):return {a:rows(task,a,o) for a in (('N','A','C') if task=='fundus' else ('N','A','EA','O2','D4'))}
        for task in streams:
            rs=rows(task,'C0',None);validate_rows(rs,streams[task],task,'C0',None,rs)
            bad=copy.deepcopy(rs);bad[0]['counts']['forwards']=4
            with self.assertRaises(ValueError):validate_rows(bad,streams[task],task,'C0',None,rs)
        totals=dict(records=12,forwards=54,backwards=6,base_adam=6,perturb=0,restore=0)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for task in streams:(out/f'{task}_canonical_C0.jsonl').write_text('\n'.join(json.dumps(r) for r in rows(task,'C0',None)))
            for o in (0,1):(out/f'polyp_{o}_C.jsonl').write_text('\n'.join(json.dumps(r) for r in rows('polyp','C',o)))
            for name,v in {'run.completion.json':dict(status='B4_RUN_COMPLETE',progress=totals,trainable={},gpu_seconds=1),'smoke.completion.json':dict(physical=dict(forwards=76,backwards=8,base_adam=8,perturb=0,restore=0)),'CPU_validation.json':dict(exit_code=0)}.items():(out/name).write_text(json.dumps(v))
            with patch('dpa_ctta.b4_analysis.stream',side_effect=lambda r,t,o:streams[t]),patch('dpa_ctta.b4_analysis.old_records',side_effect=old),patch('dpa_ctta.b4_analysis.BUDGET',totals):recompute(out,{},dict(commit='CPU_FIXTURE'))
            self.assertEqual(json.loads((out/'verification.json').read_text())['status'],COMPLETE)
