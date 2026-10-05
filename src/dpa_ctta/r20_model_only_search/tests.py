"""Generated-input qualification: native parity, nontrivial gradients and recovery."""
import copy,io,random,unittest
import numpy as np
import torch
from torch import nn
from .method import Host,Adapter,weights_for,boundary
from ..b1_host import Host as Native
from ..r19_model_only.method import equal,snapshot
from ..r8_ba.rng import capture,restore

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,4,1);self.bn=nn.BatchNorm2d(4)
    def forward(self,x):return self.bn(self.conv(x)).relu()
class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.up3=Block();self.head=nn.Conv2d(4,2,1)
    def forward(self,x):
        h=self.up3(torch.nn.functional.avg_pool2d(x,16));z=torch.nn.functional.interpolate(self.head(h),size=x.shape[-2:],mode='nearest');return z,[h],h

CONFIGS={'G':dict(family='control',params={}), 'T':dict(family='teacher',params=dict(teacher_mix=.25,ema_alpha=.99)), 'W':dict(family='pixel_weight',params=dict(variance_temperature=.03,boundary_boost=2)), 'A':dict(family='online_adapter',params=dict(rank=4,adapter_lr_multiplier=1)), 'P':dict(family='target_prototype',params=dict(prototype_mix=.25,cosine_temperature=.1))}
class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42);cls.model=Tiny();cls.x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    def host(self,c):return Host(None,copy.deepcopy(c),20260907,'test',device='cpu',model=copy.deepcopy(self.model))
    def native(self,arm='G'):
        random.seed(20260907);np.random.seed(20260907);torch.manual_seed(20260907);return Native(arm,device='cpu',model=copy.deepcopy(self.model))
    def test_native_and_zero_limits(self):
        configs=[CONFIGS['G'],dict(CONFIGS['G'],host='C')]
        for k,param in [('T',dict(teacher_mix=0)),('W',dict(variance_disabled=True,boundary_boost=0)),('A',dict(disabled=True)),('P',dict(prototype_mix=0))]:
            c=copy.deepcopy(CONFIGS[k]);c['params'].update(param);configs.append(c)
        configs.append(CONFIGS['P']) # Empty past bank must leave first update exactly native.
        for c in configs:
            h=self.host(c);r=self.native('C' if c.get('host')=='C' else 'G')
            for i in range(2 if c!=CONFIGS['P'] else 1):
                a,_=h.step(self.x);b,_=r.step(self.x)
                self.assertTrue(torch.equal(a,b),c)
                sa,sb=h.snapshot(),snapshot(r)
                for k in ['parameters','gradients','adam','grata','native_rng']:self.assertTrue(equal(sa[k],sb[k]),(c,k))
            h.close()
            for x in r.handles:x.remove()
    def test_snapshot_next_arrival_exact(self):
        combos=[dict(family='combo',params={},components={a:CONFIGS[a]['params'],b:CONFIGS[b]['params']}) for a,b in [('T','W'),('P','W'),('A','T'),('A','W')]]
        for c in list(CONFIGS.values())+combos:
            h=self.host(c);h.step(self.x);s=h.snapshot();buf=io.BytesIO();torch.save(s,buf);buf.seek(0);s=torch.load(buf,weights_only=True)
            a,_=h.step(self.x.roll(3,-1));end=h.snapshot();h.restore(s);b,_=h.step(self.x.roll(3,-1))
            self.assertTrue(torch.equal(a,b),c);self.assertTrue(equal(end,h.snapshot()),c);h.check_frozen(True);h.close()
    def test_adapter_gradients_and_alignment(self):
        h=self.host(CONFIGS['A']);self.assertEqual(len(h.native.opt.param_groups),1);self.assertEqual(len(h.native.base.param_groups),2)
        h.step(self.x);self.assertGreater(h.diag['adapter_U_gradient'],0);self.assertEqual(h.diag['adapter_V_gradient'],0)
        h.step(self.x.roll(7,-1));self.assertGreater(h.diag['adapter_V_gradient'],0)
        self.assertGreater(h.diag['adapter']['relative_rms'],0);h.check_frozen(True);h.close()
    def test_readonly_and_output_fusion_native_state(self):
        c=dict(CONFIGS['G'],params=dict(horizontal_output_weight=.25));h=self.host(c);r=self.host(CONFIGS['G'])
        for i in range(2):
            h.step(self.x);r.step(self.x)
            a,b=h.snapshot(),r.snapshot()
            for k in ['parameters','gradients','adam','grata','native_rng','counts']:self.assertTrue(equal(a[k],b[k]),k)
        before=h.snapshot();h._readonly(h.native.model,self.x)
        # Host wrapper restores native bookkeeping around read-only fusion.
        h.restore(before);self.assertTrue(equal(before,h.snapshot()));h.close();r.close()
    def test_two_updates_and_lr(self):
        h=self.host(dict(CONFIGS['G'],params=dict(online_steps_per_arrival=2)));r=self.native()
        a,_=h.step(self.x);r.step(self.x);b,_=r.step(self.x);self.assertTrue(torch.equal(a,b));self.assertTrue(equal(h.native.base.state_dict(),r.base.state_dict()));h.close()
        for x in r.handles:x.remove()
    def test_positive_channel_normalized_weights(self):
        g=torch.Generator().manual_seed(5);p=torch.rand(6,1,2,32,32,generator=g);w=weights_for(p,.03,2)
        self.assertTrue((w>0).all());self.assertTrue(torch.allclose(w.mean((-2,-1)),torch.ones(1,2),atol=1e-6));self.assertFalse(w.requires_grad)
    def test_labels_rejected(self):
        h=self.host(CONFIGS['G'])
        with self.assertRaises((AttributeError,ValueError)):h.step(dict(image=self.x,label=self.x))
        with self.assertRaises(ValueError):h.native.opt.cal_groundtruth_loss({'mask':self.x})
        h.close()

    def test_adapter_gradients_do_not_accumulate_across_arrivals(self):
        h=self.host(CONFIGS['A']);seen=[];hook=h.adapter.U.weight.register_hook(lambda g:seen.append(g.clone()))
        for i in range(3):
            h.step(self.x.roll(i,-1));self.assertEqual(len(seen),i+1);self.assertTrue(torch.equal(h.adapter.U.weight.grad,seen[-1]))
        hook.remove();h.close()
    def test_prototypes_read_past_then_write_and_expire(self):
        h=self.host(CONFIGS['P']);feature=torch.zeros(1,4,32,32);feature[:,0,:,:11]=1;feature[:,1,:,11:22]=1;feature[:,2,:,22:]=1
        logits=torch.full((1,2,512,512),-8.);logits[:,0,:,176:]=8.;logits[:,1,:,352:]=8.;calls=[0]
        def current_only(model,x):
            h.feature=feature;calls[0]+=1
            return logits if calls[0]%2 else torch.flip(logits,(-1,))
        h._readonly=current_only
        h._prepare_prototypes(self.x);self.assertIsNone(h.proto_target);self.assertEqual(len(h.current_proto),3)
        h.step(self.x);self.assertEqual([len(b) for b in h.proto.values()],[1,1,1]);self.assertEqual(h.diag['prototype_edit_coverage'],0)
        h._prepare_prototypes(self.x);self.assertIsNotNone(h.proto_target);self.assertTrue((h.proto_target[:,1]<=h.proto_target[:,0]).all());self.assertTrue(torch.isfinite(h.proto_target).all())
        h.visits=128;h._prepare_prototypes(self.x);self.assertIsNone(h.proto_target);self.assertEqual(h.diag['prototype_expired'],3);h.close()
    def test_split_is_content_bound_and_rank_bucket_is_transitive(self):
        from .selection import split,ranking
        rows=[dict(image_sha256=f'{d}-{i}',domain=d,subset='remaining_dev',patient_linkage='UNKNOWN') for d,n in [('a',37),('b',586),('c',336),('d',736)] for i in range(n)]
        assignment,pool,audit=split(rows);a,p,b=split(list(reversed(rows)))
        self.assertEqual(assignment,a);self.assertEqual(pool,p);self.assertEqual(audit,b);self.assertEqual(len(pool),384);self.assertEqual(audit['SEARCH'],1017);self.assertTrue(all(assignment[x]=='SEARCH' for x in pool))
        base=dict(nonrisk=True,order_delta_pp=[.1,.2],seconds_per_image=1)
        rs=[dict(base,id=str(i),delta_pp=d) for i,d in enumerate([.049,.051,.099,.101])]
        self.assertEqual([r['id'] for r in ranking(rs)],['3','1','2','0'])

    def test_search_scorer_never_requests_sealed_label(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from .score import score_job
        from ..r10_12h_core.run import save
        from ..r8_ba.journal import TargetJournal
        from ..r8_ba.streams import rows_sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for d in ('stages','processes','scorer','private','target'): (root/d).mkdir()
            save(root/'stages/S1.barrier.json',dict(status='ALL_WORKERS_RETIRED'))
            rows=[dict(image_path=str(i),image_sha256=str(i),image_size=[512,512],mask_path=str(i),domain='d',subset='remaining_dev') for i in (0,1)]
            online=[{k:r[k] for k in ('image_path','image_sha256','image_size')} for r in rows]
            save(root/'private/SCREEN_o0.json',online);save(root/'scorer/SCREEN_o0.json',rows);save(root/'scorer/SPLIT.private.json',{'0':'SEARCH','1':'SEALED_REVIEW'})
            h=self.host(CONFIGS['G']);j=TargetJournal(root/'target/toy',h,'toy',rows_sha(online));j.create()
            for i in (0,1):
                z,t=h.step(self.x);t['seconds']=.01;j.append(np.packbits(z.sigmoid().numpy()[0]>=.5).tobytes(),t)
            j.complete(2);h.close();opened=[]
            class Reader:
                def __init__(self,*a):pass
                def read(self,row):
                    opened.append(row['image_sha256'])
                    if row['image_sha256']=='1':raise AssertionError('sealed label requested')
                    return torch.zeros(1,2,512,512)
                def after_check(self):pass
            job=dict(id='toy',stream='screen',order=0,seed=20260907,candidate=dict(id='G'))
            with patch('dpa_ctta.r20_model_only_search.score.TargetReader',Reader):
                result=score_job(dict(output_root=tmp,target_root=tmp),lambda:None,dict(job=job,stage='S1'))
            self.assertEqual(result['rows'],1);self.assertEqual(opened,['0'])
    def test_clustered_bootstrap_requires_all_frozen_seeds(self):
        from .report import bootstrap_pair
        rows=[]
        for method in ('M','G'):
            for seed in (1,2):
                for order in (0,1):
                    for i in range(4):
                        rows.append(dict(condition=method,seed=seed,order=order,content=str(i),domain=str(i//2),role='SEALED_REVIEW',metrics=[dict(channel=k,dice=.6+(.01 if method=='M' else 0)) for k in ('OD','OC')]))
        result,ci,negative=bootstrap_pair(rows,'M','G',[1,2],repeats=100)
        self.assertEqual(result['content_identities'],4);self.assertEqual(result['paired_observations'],16);self.assertAlmostEqual(result['delta_pp'],1);self.assertAlmostEqual(result['CI_low_pp'],1);self.assertEqual(negative,[])
        missing,_,_=bootstrap_pair(rows,'M','G',[1,2,3],repeats=100);self.assertEqual(missing['status'],'INCOMPLETE')

if __name__=='__main__':unittest.main()
