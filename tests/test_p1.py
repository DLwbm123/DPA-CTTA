import copy
import importlib
import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
import torch
from torch import nn
from test_m1 import TinyPrompt,TinyBN
from dpa_ctta.p1_host import anchor_kl,ensemble,anchor_step,SourceAnchorHost
from dpa_ctta.p1_data import select_extension,load_proxy,ARMS,expected
from dpa_ctta.p1_analysis import evaluate,validate_rows,summarize,recompute
from dpa_ctta.p1_run import Pair,record,teacher
from dpa_ctta.host_diagnostic import snapshot,close,rng
from dpa_ctta.m1_run import Observed
from dpa_ctta.hosts.vptta import native_step_from_source
from dpa_ctta.integrations.ctta_suite import checkout_root,native_imports
from dpa_ctta.proxy_loss import FixedProxy,ProxyProvenance
from dpa_ctta.source_pilot import evaluate_after_step

class BN(TinyBN):
    def forward(self,x):
        if self.new_sample:self.sample_num+=1
        return super().forward(x)

class Model(nn.Module):
    def __init__(self,task):
        super().__init__();self.task=task;self.resnet=nn.Sequential(BN(),nn.Conv2d(3,2 if task=='fundus' else 1,1))
    def change_BN_status(self,new_sample):
        self.resnet[0].new_sample=new_sample
    def forward(self,x):
        z=self.resnet(x);return (z,None,None) if self.task=='fundus' else z


def tiny(task,weight=None):
    torch.manual_seed(101)
    with native_imports(checkout_root()[0],task) as directory:
        Memory=importlib.import_module('utils.memory').Memory
        step=(native_step_from_source if weight is None else anchor_step)(directory/'vptta.py',BN)
    h=SimpleNamespace(task=task,prompt=TinyPrompt(),model=Model(task),adabn=BN,neighbor=16,iters=1,memory_bank=Memory(40,3),anchor_weight=weight,last_anchor=None)
    h.prompt.prompt_size=1;h.optimizer=torch.optim.Adam(h.prompt.parameters(),lr=.05 if task=='fundus' else .01,betas=(.9,.99))
    h._anchor_loss=MethodType(SourceAnchorHost._anchor_loss,h)
    def apply(x,q=None):
        h.q0=q
        try:return step(h,x)
        finally:h.q0=None
    h.step=apply;return h


def pair(task):
    p=Pair.__new__(Pair);p.hosts={'A':tiny(task),'SA':tiny(task,.1)};p.watches={k:Observed(h) for k,h in p.hosts.items()};p.rngs={a:rng() for a in p.hosts}
    model=nn.Sequential(nn.Conv2d(3,2 if task=='fundus' else 1,1),nn.BatchNorm2d(2 if task=='fundus' else 1)).eval().requires_grad_(False)
    p.n=SimpleNamespace(model=model,step=lambda x:model(x).detach());p.tw=Observed(p.n);return p


def row(i,domain='REFUGE',**kw):
    return dict(sample_id=str(i),image_sha256=str(i),mask_sha256='m'+str(i),domain=domain,split='combined',manifest_index=i,**kw)

class P1Tests(unittest.TestCase):
    def test_kl_gradient_endpoints_and_teacher_detach(self):
        z=torch.tensor([[-30.,.7,30.]],requires_grad=True);q=torch.tensor([[0.,.3,1.]],requires_grad=True)
        kl,bce,h=anchor_kl(z,q);g,=torch.autograd.grad(kl,z)
        torch.testing.assert_close(g,(z.detach().sigmoid()-q.detach())/z.numel());self.assertIsNone(q.grad)
        self.assertTrue(torch.isfinite(torch.stack([kl,bce,h])).all());self.assertGreater(float(g.abs().max()),0)
        with self.assertRaises(ValueError):anchor_kl(z,torch.tensor([[-1.,.3,1.]]))

    def test_probability_average_threshold_and_native_metrics(self):
        q=torch.tensor([[[[.1,.9],[0.,1.]]]]);p=1-q
        result=evaluate(ensemble(q,p),torch.ones_like(q),'polyp')[0];self.assertEqual(result['dice'],1.)
        z=torch.randn(1,1,8,8);mask=(torch.rand_like(z)>.5).float()
        old=evaluate_after_step(z,mask,'polyp')[0];new=evaluate(z.sigmoid(),mask,'polyp')[0]
        for k in ['dice','assd','gt_empty','gt_full','pred_empty','pred_full']:self.assertEqual(old[k],new[k])

    def test_lambda_zero_native_state_cold_and_warm_both_tasks(self):
        for task in ['fundus','polyp']:
            a,b=tiny(task),tiny(task,0)
            for i in range(18):
                torch.manual_seed(200+i);x=torch.rand(1,3,8,8);q=torch.rand(1,2 if task=='fundus' else 1,8,8)
                pa=a.step(x);pb=b.step(x,q);close(pa,pb);close(snapshot(a),snapshot(b))

    def test_pair_labels_ensemble_and_teacher_invariants(self):
        p=pair('polyp');q=pair('polyp');close(q.n.model.state_dict(),p.n.model.state_dict(),exact=True);initial=copy.deepcopy(p.n.model.state_dict())
        for i in range(3):
            x=torch.rand(1,3,8,8);preds,_=p.step(x);others,_=q.step(x)
            for a in preds:close(preds[a],others[a])
            before={a:snapshot(h) for a,h in p.hosts.items()}
            evaluate(preds['ENS_SA'],torch.zeros(1,1,8,8),'polyp');evaluate(preds['ENS_SA'],torch.ones(1,1,8,8),'polyp')
            for a,h in p.hosts.items():close(before[a],snapshot(h),exact=True)
        close(initial,p.n.model.state_dict(),exact=True)
        self.assertTrue(all(v.grad is None and not v.requires_grad for v in p.n.model.parameters()))
        self.assertEqual(p.watches['A'].counts['online_adam'],3);self.assertEqual(p.watches['SA'].counts['online_adam'],3)

    def test_selector_hash_rank_dedup_roles_and_zero_extension(self):
        rows=[row(i) for i in range(80)];legacy=[dict(rows[0],group_id='0')]
        rows += [dict(row(80),image_sha256='1',domain='ORIGA'),dict(row(81),image_sha256='2',mask_sha256='conflict'),row(82,sealed_final=True),row(83,patient_id='protected')]
        source=[row(100,patient_id='protected')];chosen,counts,excluded=select_extension(rows,'fundus',source,legacy)
        self.assertEqual(counts['REFUGE']['extension_dev'],32);self.assertEqual(len(chosen),33)
        self.assertEqual(chosen,select_extension(rows,'fundus',source,legacy)[0]);self.assertEqual(chosen,sorted(chosen,key=lambda r:r['manifest_index']))
        self.assertFalse({'2','82','83'}&{r['group_id'] for r in chosen});self.assertTrue(all(r['split']=='combined' for r in chosen))
        only,cs,_=select_extension([rows[0]],'fundus',[],legacy);self.assertEqual(cs['REFUGE']['extension_dev'],0);self.assertEqual(len(only),1)

    def test_frozen_proxy_step_shape_mask_binding(self):
        real=FixedProxy(torch.rand(4,3,8,8),torch.zeros(4,1,8,8),None,ProxyProvenance.SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'p.pt'
            for arm,step in [('O2',600),('D4',150),('L4',150)]:
                torch.save(dict(pixels=real.pixel_rgb,outer_step=step),p);s=p.stat();reg={'artifacts':{arm:dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)}}
                got=load_proxy(reg,'polyp',arm,real);self.assertTrue(torch.equal(got.mask,real.mask))
                torch.save(dict(pixels=real.pixel_rgb,outer_step=1),p);s=p.stat();reg['artifacts'][arm].update(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
                with self.assertRaises(ValueError):load_proxy(reg,'polyp',arm,real)

    def test_records_coverage_parent_binding_and_cpu_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rows=[dict(row(1),group_id='1',subset='legacy_dev'),dict(row(2),group_id='2',subset='extension_dev')]
            reg=dict(tasks={'polyp':dict(target=rows,counts={'REFUGE':dict(legacy_dev=1,extension_dev=1,combined=2)})},limitations=[],budget=dict(groups=2,records=32,online=20))
            for order in [0,1]:
                bundle=pair('polyp');streams={a:[] for a in ARMS}
                for i,r in enumerate(rows):
                    x=torch.rand(1,3,8,8);ps,ms=bundle.step(x)
                    for a in ['O2','D4','L4']:ps[a]=ps['A'];ms[a]=dict(ms['A'])
                    for a in ARMS:
                        rec=record('polyp',order,a,i,r,ps[a],ms[a],.01,0);rec['metrics']=evaluate(ps[a],torch.ones(1,1,8,8),'polyp');streams[a].append(rec)
                for a,rs in streams.items():
                    self.assertTrue(validate_rows(rs,rows,'polyp',order,a));(out/f'polyp_{order}_{a}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs))
                bad=copy.deepcopy(streams['ENS_SA']);bad[0]['parent_prediction_ids']=[]
                with self.assertRaises(ValueError):validate_rows(bad,rows,'polyp',order,'ENS_SA')
            (out/'run.completion.json').write_text(json.dumps(dict(status='P1_RUN_COMPLETE',progress=dict(records=32,online=20,outer=0,inner=0,teacher_forwards=4),gpu_seconds=1)))
            (out/'smoke.completion.json').write_text(json.dumps(dict(status='P1_SMOKE_PASS',online=12)))
            recompute(out,reg,{'commit':'test'})
            self.assertEqual(json.loads((out/'verification.json').read_text())['status'],'P1_NO_DD_COMPARISON_COMPLETE')
            self.assertEqual(summarize({a:[] for a in ARMS},'polyp')['groups'],0)
