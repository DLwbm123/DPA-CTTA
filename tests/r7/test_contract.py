import copy,inspect,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from common import method,obs,segmenter,pixels,source_fixture
from dpa_ctta.r7_shared.network import film
from dpa_ctta.r7_shared.host import OnlineHost
from dpa_ctta.r7_shared.source import (SourceTrainer,SourceData,Record,split,roles,sequence,anchors,simulate,shared_basis)
from dpa_ctta.r7_shared.preparation import inference_from_tensors,prepared_artifact
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_shared.plan import real_entry,matrix,source_binding
from dpa_ctta.r7_shared.io import Output,legacy_reader

def eq(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    return a==b

class Contracts(unittest.TestCase):
    def test_film_bitwise_and_zero_derivative(self):
        h=torch.randn(1,256,3,3);v=torch.zeros(512,requires_grad=True);z=film(h,v)
        self.assertTrue(torch.equal(z,h));z.sum().backward();self.assertTrue(torch.all(v.grad[256:]!=0))
    def test_observer_shapes_current_BN_and_cache_release(self):
        s=segmenter();x=pixels();z,raw,e=s(x,observe=True)
        self.assertEqual(raw.shape,(134,));self.assertEqual(e.shape,(64,64));self.assertFalse(s.cache);self.assertIsNone(s.v)
        for m in s.modules():
            if isinstance(m,torch.nn.BatchNorm2d):self.assertFalse(m.track_running_stats);self.assertIsNone(m.running_mean)
        self.assertFalse(any(p.requires_grad for p in s.model.parameters()));s.close()
    def test_scaler_is_fit_only_and_no_online_running_change(self):
        m=method('A');old=copy.deepcopy(m.observer.state_dict());m.observe(*obs())
        self.assertTrue(eq(old,m.observer.state_dict()))
        for fold in ('cal','val','target','fit'):
            with self.assertRaises(ValueError):m.observer.fit_scaler(torch.randn(4,134),fold)
    def test_independent_RNG_and_identical_full_static_initialization(self):
        rng=torch.get_rng_state().clone()
        for group in 'ABC':
            a,b=method(group),method(group,True)
            self.assertTrue(eq(a.state_dict(),b.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_online_two_forward_no_autograd_and_current_only(self):
        for group in 'ABC':
            s=segmenter();m=method(group);m.freeze();h=OnlineHost(s,m)
            with patch('torch.autograd.grad',side_effect=AssertionError('no VJP')),patch('torch.Tensor.backward',side_effect=AssertionError('no backward')):
                z,a=h.step(pixels());self.assertEqual(a['counts']['backbone_forwards'],2)
            for k in ('label','domain','future','sample_id'):
                with self.assertRaises(TypeError):h.step(pixels(),**{k:None})
            self.assertFalse(s.cache);self.assertEqual(set(h.state),set(m.initial()));s.close()
    def test_state_restore_next_output_and_GT_future_independence(self):
        for group in 'ABC':
            s=segmenter();m=method(group);m.freeze();h=OnlineHost(s,m);h.step(pixels());saved=h.save_state()
            a,_=h.step(pixels(1));state=h.save_state();h.load_state(saved)
            # Evaluator GT/future tensors are constructed and never passed to host.
            gt=torch.ones(1,2,512,512);future=pixels(2);del gt,future
            b,_=h.step(pixels(1));self.assertTrue(torch.equal(a,b));self.assertTrue(eq(state,h.save_state()));s.close()
    def test_no_commit_on_second_forward_failure_and_no_retry(self):
        s=segmenter();m=method('B');m.freeze();h=OnlineHost(s,m);before=h.save_state();orig=s.forward;call=[0]
        def fail(*a,**kw):
            call[0]+=1
            if call[0]==2:raise ValueError('fixture final failure')
            return orig(*a,**kw)
        with patch.object(s,'forward',side_effect=fail):
            with self.assertRaises(ValueError):h.step(pixels())
        self.assertTrue(eq(before['state'],h.state));self.assertEqual(h.visits,0);self.assertTrue(h.failed)
        with self.assertRaises(RuntimeError):h.step(pixels())
        with self.assertRaises(ValueError):h.load_state(before)
        s.close()
    def test_state_corruption_binding_rejected(self):
        s=segmenter();m=method('A');m.freeze();h=OnlineHost(s,m)
        for which in ('binding','dtype','shape','finite','counter'):
            p=h.save_state()
            if which=='binding':p['binding']='bad'
            elif which=='counter':p['state']['counter']=2
            elif which=='dtype':p['state']['m']=p['state']['m'].float()
            elif which=='shape':p['state']['m']=torch.zeros(2,dtype=torch.float64)
            else:p['state']['m'][0]=float('nan')
            with self.assertRaises(ValueError):h.load_state(p)
        s.close()
    def test_batch_and_nonfinite_inputs_hard_fail(self):
        for bad in (pixels().repeat(2,1,1,1),torch.full((1,3,512,512),float('nan'))):
            s=segmenter();h=OnlineHost(s)
            with self.assertRaises(ValueError):h.step(bad)
            self.assertTrue(h.failed);s.close()
    def test_C0_one_forward(self):
        s=segmenter();z,a=OnlineHost(s).step(pixels());self.assertEqual(a['counts']['backbone_forwards'],1);s.close()
    def test_ablation_lifecycle_and_constant_source(self):
        for group,variant in [('A','A_ISO_OBS'),('B','B_PRED_ONLY'),('C','C_CONST_R')]:
            s=segmenter();m=method(group);m.freeze()
            if group=='C':
                with self.assertRaises(ValueError):OnlineHost(s,m,ablation=variant)
                with self.assertRaises(ValueError):m.set_constant_variance(torch.ones(8),'target')
                m.set_constant_variance(torch.ones(8),'cal')
            _,a=OnlineHost(s,m,ablation=variant).step(pixels());self.assertEqual(a['counts']['backbone_forwards'],2)
            if group=='B':self.assertNotIn('ISTA_iterations',a['counts'])
            stat=method(group,True);stat.freeze()
            with self.assertRaises(ValueError):OnlineHost(s,stat,ablation=variant)
            s.close()
    def test_saved_assets_inference_binding_and_eta(self):
        s=segmenter();m=method('B');m.freeze();artifact=prepared_artifact(s,m);w=copy.deepcopy(m.state_dict())
        h=inference_from_tensors(s,'B',False,m.basis,w,artifact['binding']);self.assertEqual(h.visits,0)
        w['frozen_eta']*=2
        with self.assertRaises(ValueError):inference_from_tensors(s,'B',False,m.basis,w,artifact['binding'])
        s.close()
    def test_split_overlap_oracle_fit_only_and_roles(self):
        data,oracle=source_fixture()
        self.assertEqual(len(set(sum(data.folds.values(),[]))),48)
        for step in range(3):
            ids=sequence('fit',step);rs,reuse=roles(data,'fit',step,ids,oracle['fit'])
            self.assertFalse(reuse);self.assertEqual(len(set(rs)),8)
            for i,aid in enumerate(ids):self.assertNotIn(rs[2*i+1],oracle['fit'].support_pairs[aid])
        bad=copy.deepcopy(data.folds);bad['cal'][0]=bad['fit'][0]
        with self.assertRaises(ValueError):split(list(data.records),bad)
        with self.assertRaises(ValueError):shared_basis(oracle['cal'],data)
        with self.assertRaises(ValueError):data.get(data.folds['val'][0],'fit')
        records=list(data.records.values());r=records[0];records[0]=Record(r.group,r.fold,r.image,r.label,'target')
        with self.assertRaises(ValueError):SourceData(records,data.folds)
    def test_styles_identity_fold_factors_and_local_noise(self):
        rng=torch.get_rng_state().clone()
        for fold in ('fit','cal','val'):
            a=anchors(fold);self.assertTrue(torch.equal(a,anchors(fold)))
            for i in range(1,len(a)):self.assertEqual(int((a[i]!=a[0]).sum()),3 if fold=='val' else 1+(i-1)%2)
        x=pixels();self.assertTrue(torch.equal(simulate(x,anchors('fit')[0],'x'),x))
        style=anchors('fit')[0].clone();style[7]=.03
        self.assertTrue(torch.equal(simulate(x,style,'a'),simulate(x,style,'a')));self.assertFalse(torch.equal(simulate(x,style,'a'),simulate(x,style,'b')))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_source_fit_static_and_calibration_freeze(self):
        data,o=source_fixture()
        for group in 'ABC':
            s=segmenter();m=method(group,True);t=SourceTrainer(s,m,data,o['fit']);before=s.forwards
            loss,a=t.fit_step();self.assertEqual(s.forwards-before,12);self.assertTrue(t.gradients)
            with self.assertRaises(ValueError):t.start_calibration(o['cal'])
            t.start_calibration(o['cal'],procedural_micro=True)
            before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad};count=s.forwards
            t.cal_step();self.assertEqual(s.forwards-count,4)
            for n,p in m.named_parameters():
                if n in before:self.assertTrue(torch.equal(p,before[n]));self.assertIsNone(p.grad)
            s.close()
    def test_source_query_labels_change_loss_not_support_state(self):
        data,o=source_fixture();s=segmenter();m=method('B');t=SourceTrainer(s,m,data,o['fit'])
        a,aa=t.episode('fit',0,o['fit']);ids=sequence('fit',0);rs,_=roles(data,'fit',0,ids,o['fit'])
        for g in rs[1::2]:
            r=data.records[g];data.records[g]=Record(r.group,r.fold,r.image,1-r.label)
        b,bb=t.episode('fit',0,o['fit']);self.assertNotEqual(float(a),float(b));self.assertTrue(eq(aa['state'],bb['state']))
        with self.assertRaises(ValueError):t.episode('fit',0,o['val'])
        s.close()
    def test_disabled_real_entries_before_asset_read(self):
        with patch('builtins.open',side_effect=AssertionError('no reads')):
            for scope in ('SOURCE_PREP','TARGET_SCREEN','TARGET_MECHANISM','TARGET_EXTENSION','GPU_SMOKE','AB','AUTO'):
                with self.assertRaises(PermissionError):real_entry(scope,enabled=True,receipt={'status':'PASS'})
    def test_matrix_bounds_source_pending_no_receipts(self):
        for scope,n,f in [('TARGET_SCREEN',24,122913),('TARGET_MECHANISM',9,35118),('TARGET_EXTENSION',12,66334)]:
            rows=matrix(scope,('A','B') if scope=='TARGET_EXTENSION' else ())
            self.assertEqual(len(rows),n);self.assertEqual(sum(x['network_forwards'] for x in rows),f)
            self.assertTrue(all(x['status']=='NOT_RUN' and x['device'] is None for x in rows))
        with self.assertRaises(ValueError):matrix('TARGET_EXTENSION',('A','B','C'))
        self.assertIsNone(source_binding()['source_manifest_sha256'])
    def test_fresh_output_owner_and_first_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()/'new';out=Output(p,{'sha':'fixed'});out.fail(ValueError('first'),{'F':2})
            with self.assertRaises(FileExistsError):Output(p,{})
            with self.assertRaises(FileExistsError):out.fail(ValueError('second'),{})
            self.assertEqual(json.loads((p/'first_error.json').read_text())['message'],'first')
            with self.assertRaises(ValueError):out.write('../outside',{})
            (p/'owner.json').write_text('{}')
            with self.assertRaises(ValueError):out.write('x.json',{})
    def test_reused_reader_rejects_symlink_hardlink_escape(self):
        import os
        reader=legacy_reader()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a.json').write_text('{}');(p/'link').symlink_to('a.json');os.link(p/'a.json',p/'hard')
            r=reader.Reader(p)
            for name in ('link','../outside','hard'):
                with self.assertRaises((ValueError,OSError)):
                    with r.opened(name):pass
    def test_scientific_review_paired_tails_and_ASSD_denominators(self):
        from dpa_ctta.r7_shared.report import review
        rows=[]
        for arm in ['C_BASE','C0']+[g+'_'+v for g in 'ABC' for v in ('FULL','STATIC')]:
            for o in (0,1,4):
                for i in range(4):
                    rows.append(dict(arm=arm,order=o,content=str(i),domain=str(i%2),subset='remaining_dev',dice=[.5+(arm.endswith('FULL'))*.01,.6],assd=[None if i%2 else 1.,2.]))
        r=review(rows);self.assertFalse(r['automatic_nomination']);self.assertAlmostEqual(r['summary'][0]['primary'],.005)
        self.assertTrue(any(row['ASSD'][0]['common_valid']==0 for row in r['paired']))
        with self.assertRaises(ValueError):review(rows[:-1])
    def test_oracle_16_combined_steps_support_only(self):
        from dpa_ctta.r7_shared.source import oracle_one
        data,o=source_fixture();s=segmenter();before=COUNTS.copy();v,pair=oracle_one(s,data,'fit',0)
        counts=COUNTS-before;self.assertEqual(counts['backbone_forwards'],32);self.assertEqual(counts['source_backward_calls'],16);self.assertEqual(counts['source_Adam'],16)
        self.assertEqual(len(set(pair)),2);self.assertGreater(float(v.norm()),0);self.assertFalse(v.requires_grad);s.close()
    def test_fixed_published_alias_mapping_metadata_only(self):
        import os
        reader=legacy_reader();root=Path(__file__).resolve().parents[2]
        add=json.loads((root/'analysis/r6d_posthoc_v1/io_addendum/R6D_IO_ADDENDUM.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.mkdir();version=add['pointer']['required_result_directory'];v=p/version;v.mkdir(parents=True)
            (v/'public_aggregate.json').write_text('{}')
            pointer=dict(valid=True,status='R6A_COMPLETE_NO_ADVANCE',binding=add['historical_binding'],result_directory=version)
            (p/'current_result.json').write_text(json.dumps(pointer));(p/'current').symlink_to(version);(p/'public_aggregate.json').symlink_to('current/public_aggregate.json')
            r=reader.Reader(p);layout,_=reader.layout(r,add);reader.check_layout(r,add,layout)
            from dpa_ctta.r7_shared.io import snapshot_published
            out=snapshot_published(p,Path(d).resolve()/'snapshot',add)
            self.assertEqual((out.path/'public_aggregate.json').read_text(),'{}')
            with self.assertRaises((ValueError,OSError)):r.record('public_aggregate.json')
            target=Path(d)/'copy.json';r.record(add['aggregate_input']['source_relative_path'],target);self.assertEqual(target.read_text(),'{}')
            pointer['binding']={};(p/'current_result.json').write_text(json.dumps(pointer))
            with self.assertRaises(ValueError):reader.layout(r,add)
    def test_source_constant_R_all_cal_observations_only(self):
        data,o=source_fixture();s=segmenter();m=method('C');t=SourceTrainer(s,m,data,o['fit'])
        m.freeze();t.mode='FROZEN';before=s.forwards;t.constant_variance()
        self.assertTrue(m.constant_ready);self.assertEqual(s.forwards-before,32*len(data.folds['cal']));s.close()
    def test_source_first_failure_is_terminal(self):
        data,o=source_fixture();s=segmenter();m=method('B');t=SourceTrainer(s,m,data,o['fit'])
        with patch.object(t,'episode',side_effect=ValueError('first')):
            with self.assertRaises(ValueError):t.fit_step()
        self.assertEqual(t.first_error['message'],'first')
        with self.assertRaises(RuntimeError):t.fit_step()
        self.assertEqual(t.fit_steps,0);s.close()
    def test_fixed_fit_and_calibration_limits(self):
        data,o=source_fixture();s=segmenter();m=method('B');t=SourceTrainer(s,m,data,o['fit']);t.fit_steps=1000
        t.start_calibration(o['cal']);t.cal_steps=256
        with self.assertRaises(ValueError):t.cal_step()
        self.assertEqual(t.mode,'FAILED');s.close()
    def test_same_appearance_only_B_difference_not_raw_image(self):
        m=method('B');raw,e=obs();s,_=m.update(raw,e,m.initial());_,a=m.update(raw,e+4,s)
        self.assertEqual(float(a['difference'].norm()),0.)
    def test_source_query_pixel_cannot_update_episode_state(self):
        data,o=source_fixture();s=segmenter();m=method('A');t=SourceTrainer(s,m,data,o['fit'])
        _,aa=t.episode('fit',0,o['fit']);ids=sequence('fit',0);rs,_=roles(data,'fit',0,ids,o['fit'])
        for g in rs[1::2]:
            r=data.records[g];data.records[g]=Record(r.group,r.fold,1-r.image,r.label)
        _,bb=t.episode('fit',0,o['fit']);self.assertTrue(eq(aa['state'],bb['state']));s.close()
    def test_source_destination_overlap_denied_before_any_write(self):
        from dpa_ctta.r7_shared.io import snapshot_published
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);before=list(p.iterdir())
            with self.assertRaises(ValueError):snapshot_published(p,p/'new',{})
            self.assertEqual(before,list(p.iterdir()))
