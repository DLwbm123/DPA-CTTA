import copy,tempfile,unittest
from pathlib import Path
import torch
from test_connected import carrier,ToySource
from dpa_ctta.r10_use_write_rl.controller import Actor,Controller
from dpa_ctta.r10_use_write_rl.learning import Trainer
from dpa_ctta.r10_attribution.audit import AuditTrainer,CONTEXTS,gates
from dpa_ctta.r10_attribution.run import resolve_job,source_lock,check_lock,A_ARMS,B_ARMS
from dpa_ctta.r10_12h_core.run import save

class Attribution(unittest.TestCase):
    def test_telemetry_preserves_supervised_updates(self):
        for method in ('SUP_STATIC','SUP_RET'):
            c=Controller(carrier(),torch.ones(64,dtype=torch.float64));s=ToySource(c);a=Actor(7);initial=copy.deepcopy(a)
            t=AuditTrainer(a,c,s,7,method,'fixture',initial,total=1024);r=t.step();weights=copy.deepcopy(a.state_dict())
            b=Trainer(copy.deepcopy(initial),c,s,7,method,'fixture',initial,total=1024);q=b.step()
            for k,v in q.items():self.assertEqual(v,r[k])
            for k,v in weights.items():self.assertTrue(torch.equal(v,b.actor.state_dict()[k]))
            self.assertEqual(len(r['branch_gradients']),2)
            self.assertGreater(r['branch_gradients'][0]['use']['grad_norm'],0)
            if method=='SUP_STATIC':self.assertEqual(r['branch_gradients'][0]['write']['trainable'],0)
            else:self.assertGreater(r['branch_gradients'][0]['write']['grad_norm'],0)
    def test_supplement_does_not_update_parameters(self):
        c=Controller(carrier(),torch.ones(64,dtype=torch.float64));s=ToySource(c);a=Actor(7);before=copy.deepcopy(a.state_dict())
        t=AuditTrainer(a,c,s,7,'GR_RET_EMA','probe',total=1024,updates=False);t.step()
        for k,v in before.items():self.assertTrue(torch.equal(v,a.state_dict()[k]))
    def test_routes_share_required_endpoint_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(output_root=tmp,endpoints={'WARM':{},'POST':{}})
            for name in ('WARM','POST','SUP_STATIC','SUP_RET'):save(Path(tmp)/'source'/name/'complete.json',dict(artifact={'file':'actor.pt','sha256':name}))
            rows={a:resolve_job(c,dict(arm=a)) for a in A_ARMS+B_ARMS}
            self.assertEqual(rows['WARM_CONST_HALF']['diagnostic'],'CONST_HALF')
            self.assertEqual(rows['WARM_CONST_HALF']['source_job'],'WARM')
            self.assertEqual(rows['R10_RESET_ALL']['diagnostic'],'RESET_ALL')
            self.assertEqual(rows['R10_FORCE_WRITE']['diagnostic'],'FORCE_WRITE')
            self.assertEqual(rows['SUP_RET']['artifact'],rows['SUP_RET_CONST_HALF']['artifact'])
            self.assertEqual(rows['B_PARENT_FULL']['source_job'],None)
            self.assertEqual(rows['C0_CURRENT_STATS']['arm'],'C0')
            lock=source_lock(c,dict(arm='R10_RESET_ALL'));check_lock(lock);lock['payload']['resolved']['diagnostic']='FORCE_WRITE'
            with self.assertRaises(ValueError):check_lock(lock)
        self.assertEqual([sum(i//16==m for i in CONTEXTS) for m in range(4)],[8]*4)
        self.assertEqual(gates([.5,.6])['max_abs_from_half'],.6-.5)

    def test_paired_reporting_and_embargo(self):
        from dpa_ctta.r10_attribution.report import paired,means,report
        a=[]
        for i,d in enumerate((0,0,1,2,3)):
            a.append(dict(visit=i+1,content=str(i),domain=d,subset='remaining_dev',metrics=[dict(channel='OD',dice=.2 if d==0 else .8),dict(channel='OC',dice=.1 if d==0 else .6)]))
        b=copy.deepcopy(a)
        for r in b:
            for m in r['metrics']:m['dice']-=.02
        self.assertAlmostEqual(means(a)['Dice_macro'],.5625)
        self.assertAlmostEqual(paired(a,b)['domain_equal_delta'],.02)
        b[0]['content']='different'
        with self.assertRaises(ValueError):paired(a,b)
        with self.assertRaises(ValueError):report({},dict(status='RUNNING',jobs={}))

    def test_counterfactual_full_context_schedule(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from dpa_ctta.r10_attribution.audit import counterfactual
        from dpa_ctta.r10_12h_core.run import OPS
        c=Controller(carrier(),torch.ones(64,dtype=torch.float64));s=ToySource(c);s.controller=c
        cost=dict.fromkeys(OPS,0);visited=set();original_item=s.item;original_segmenter=s.segmenter
        def item(fold,seed,episode,visit):
            if (episode,visit) not in visited:cost['model_forwards']+=1;visited.add((episode,visit))
            return original_item(fold,seed,episode,visit)
        def segment(*args,**kw):cost['model_forwards']+=1;return original_segmenter(*args,**kw)
        s.item=item;s.segmenter=segment;s.schedule=lambda fold,i,seed:(None,None,str(i//16))
        class Guard:
            meter=SimpleNamespace(cost=cost)
            def __call__(self):pass
        with tempfile.TemporaryDirectory() as tmp,patch('dpa_ctta.r10_attribution.audit.actor_at',return_value=(Actor(8),{})):
            r=counterfactual(dict(output_root=tmp,previous_root='unused',counterfactual_contexts=CONTEXTS),s,Guard())
            self.assertEqual(r,dict(contexts=32,future_segmentations=1536));self.assertEqual(cost['model_forwards'],2496)
            saved=__import__('json').loads((Path(tmp)/'SOURCE_COUNTERFACTUAL.private.json').read_text())
            self.assertEqual(len(saved['rows']),32)
            self.assertEqual(set(saved['rows'][0]['horizons']),{'4','16'})
