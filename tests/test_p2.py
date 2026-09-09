import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from test_p1 import tiny,row
from dpa_ctta.p2_run import Bundle,make_record
from dpa_ctta.p2_data import ARMS,PARENTS,VIEWS,select_full,expected
from dpa_ctta.p2_analysis import evaluate,pixel_cells,marginal,complement,validate_bundle,summarize,recompute
from dpa_ctta.p1_host import ensemble
from dpa_ctta.m1_run import Observed
from dpa_ctta.host_diagnostic import rng,snapshot,close
from dpa_ctta.m2_run import deterministic_smoke_pair


def bundle(task):
    b=Bundle.__new__(Bundle);b.hosts={a:tiny(task) for a in PARENTS};b.watches={a:Observed(h) for a,h in b.hosts.items()};b.rngs={a:rng() for a in b.hosts}
    model=nn.Conv2d(3,2 if task=='fundus' else 1,1).eval().requires_grad_(False)
    b.n=SimpleNamespace(model=model,step=lambda x:model(x).detach());b.tw=Observed(b.n);return b


def records(task='polyp'):
    rows=[dict(row(i),group_id=str(i),subset=['legacy_dev','p1_extension_dev','remaining_dev'][i]) for i in range(3)]
    b=bundle(task);arms={a:[] for a in ARMS}
    for i,r in enumerate(rows):
        x=torch.rand(1,3,8,8);ps,ms=b.step(x);mask=(torch.rand_like(ps['N'])>.5).float()
        for a in ARMS:
            cells=pixel_cells(mask,ps['N'],ps[VIEWS[a]],ps[a]) if a in VIEWS else None
            arms[a].append(make_record(task,0,a,i,r,ps[a],ms[a],evaluate(ps[a],mask,task),cells,.1,0))
    return rows,arms

class P2Tests(unittest.TestCase):
    def test_probability_cells_endpoints_channels_and_dice(self):
        gt=torch.tensor([[[[0.,0.,1.,1.]],[[1.,0.,1.,0.]]]])
        n=torch.tensor([[[[0.,1.,0.,1.]],[[.1,.9,.8,.2]]]]);b=1-n;e=ensemble(n,b)
        self.assertTrue(torch.equal(e,torch.full_like(e,.5)))
        cells=pixel_cells(gt,n,b,e)
        for axis,p in [(1,n),(2,b),(3,e)]:
            metrics=evaluate(p,gt,'fundus')
            for c in range(2):
                for k,v in marginal(cells[c],axis).items():self.assertEqual(metrics[c][k],v)
        self.assertEqual(complement(cells[0])['foreground']['repaired_B_error'],1)
        empty=evaluate(torch.zeros(1,1,3,3),torch.zeros(1,1,3,3),'polyp')[0];self.assertIsNone(empty['assd']);self.assertEqual(empty['dice'],1.)

    def test_parent_state_label_and_ensemble_do_not_change_updates(self):
        for task in ['fundus','polyp']:
            b,c=bundle(task),bundle(task);close(b.n.model.state_dict(),c.n.model.state_dict(),exact=True)
            for i in range(3):
                x=torch.rand(1,3,8,8);p,m=b.step(x);q,_=c.step(x)
                for a in ARMS:close(p[a],q[a])
                snap={a:snapshot(h) for a,h in b.hosts.items()}
                for e,parent in VIEWS.items():
                    evaluate(p[e],torch.zeros_like(p[e]),task);evaluate(p[e],torch.ones_like(p[e]),task)
                    close(p[e],ensemble(p['N'],p[parent]),exact=True);self.assertEqual(m[e]['counts']['online_adam'],0)
                for a in PARENTS:close(snap[a],snapshot(b.hosts[a]),exact=True);close(snap[a],snapshot(c.hosts[a]))
            self.assertTrue(all(p.grad is None for p in b.n.model.parameters()))

    def test_full_selection_three_labels_conflicts_roles_and_order(self):
        rows=[row(i) for i in range(90)]+[dict(row(91,domain='ORIGA'),image_sha256='3',mask_sha256='m3'),dict(row(92),image_sha256='4',mask_sha256='bad'),row(93,sealed_final=True),row(94,video_id='sealed')]
        p1=[dict(rows[0],subset='legacy_dev'),dict(rows[1],subset='extension_dev')]
        source=[row(100,video_id='sealed')]
        chosen,counts,exc,missing=select_full(rows,'fundus',source,p1)
        self.assertEqual(len(chosen),89);self.assertEqual(counts['REFUGE']['remaining_dev'],87);self.assertFalse(missing)
        self.assertEqual([r['subset'] for r in chosen[:2]],['legacy_dev','p1_extension_dev']);self.assertEqual([r['manifest_index'] for r in chosen],sorted(r['manifest_index'] for r in chosen))
        reg={'tasks':{'fundus':{'target':chosen+[dict(row(95,domain='ORIGA'),group_id='95',subset='remaining_dev')]}}}
        rev=expected(reg,'fundus',1);self.assertEqual(rev[0]['domain'],'ORIGA');self.assertEqual(rev[1:],chosen)
        zero,cs,_,missing=select_full([row(0,sealed_final=True)],'fundus',[],p1[:1]);self.assertEqual(zero,[]);self.assertEqual(cs['REFUGE']['all_dev'],0);self.assertEqual(missing[0]['reason'],'protected_role')

    def test_reject_missing_duplicate_wrong_order_parent_counts_and_cells(self):
        rows,arms=records();self.assertTrue(validate_bundle(arms,rows,'polyp',0))
        for change in ['missing','duplicate','order','parent','counts','cells','dice','subset']:
            bad=copy.deepcopy(arms)
            if change=='missing':bad['EA'].pop()
            elif change=='duplicate':bad['A'][1]=bad['A'][0]
            elif change=='order':bad['O2'].reverse()
            elif change=='parent':bad['EO2'][0]['parent']='A'
            elif change=='counts':bad['ED4'][0]['counts']['online_adam']=1
            elif change=='cells':bad['EA'][0]['pixel_cells'][0][0]+=1
            elif change=='dice':bad['N'][0]['metrics'][0]['dice']+=.001
            elif change=='subset':bad['D4'][0]['subset']='remaining_dev'
            with self.assertRaises(ValueError,msg=change):validate_bundle(bad,rows,'polyp',0)

    def test_complete_scalar_recompute_and_interactions(self):
        rows,arms=records()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);reg=dict(tasks={'polyp':dict(target=rows,counts={})},limitations=[],budget=dict(records=42,online=18,teacher_forwards=6,outer=0,inner=0))
            for order in [0,1]:
                for a,rs in arms.items():
                    copied=copy.deepcopy(rs)
                    if order:
                        for r in copied:
                            r['order']=1
                            for k in ['prediction_id','parent_state_tag']:r[k]=r[k].replace(':0:',':1:')
                            r['parent_prediction_ids']=[v.replace(':0:',':1:') for v in r['parent_prediction_ids']]
                    (out/f'polyp_{order}_{a}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in copied))
            (out/'run.completion.json').write_text(json.dumps(dict(status='P2_RUN_COMPLETE',progress=reg['budget'],gpu_seconds=1)))
            (out/'smoke.completion.json').write_text(json.dumps(dict(status='P2_SMOKE_PASS',online=12,evidence={},gpu_seconds=1,exit_code=0)))
            recompute(out,reg,{'commit':'test'});self.assertEqual(json.loads((out/'verification.json').read_text())['status'],'P2_FROZEN_FULL_STREAM_COMPLETE')
        r=summarize(arms,'polyp');self.assertEqual(r['task_interactions_pp']['(EO2-O2)-(EA-A)'],0.)
        self.assertIsNone(summarize({a:[] for a in ARMS},'polyp')['task_domain_macro_dice_percent']['EA'])

    def test_smoke_backend_restored_on_exception(self):
        before=torch.are_deterministic_algorithms_enabled()
        with self.assertRaises(RuntimeError):
            with deterministic_smoke_pair():
                self.assertTrue(torch.are_deterministic_algorithms_enabled());raise RuntimeError('fixture')
        self.assertEqual(torch.are_deterministic_algorithms_enabled(),before)
