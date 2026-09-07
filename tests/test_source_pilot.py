import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from dpa_ctta.proxy_loss import FixedProxy, ProxyProvenance
from dpa_ctta.hosts.vptta import VPTTAHost, load_source_state, model_input_from_pixels
from dpa_ctta.integrations.ctta_suite import INPUT_SIZES, build_reference_model
from dpa_ctta.source_io import SOURCES, SPLITS, read_pixels, read_mask, registered_source, signed_distance, source_proxy
from dpa_ctta.source_pilot import (APPROVAL_REQUIRED, SourceOnlyHost, _run_arm, assemble,
                                  evaluate_after_step, main, run_registered, summarize, transform_pixels)
from test_vptta_host import pixels, proxy, same, native_snapshot

EVIDENCE = {}
EXTERNAL = os.environ.get('DPA_CTTA_BASE_ROOT')


def encoded(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    stream.seek(0)
    return stream


def mock_registration(root, task):
    """Metadata for nonexistent procedural files: never real patient registration."""
    source = SOURCES[task]
    manifest, sections, csv_rows = [], {}, []
    count = 20 if task == 'fundus' else 32
    for name, n in zip(SPLITS, (5, 2, count, 2)):
        groups, records = [], []
        for i in range(n):
            sid = f'{name}_{i}'
            relative = f'{source}/{sid}.png'
            mask = f'{source}/{sid}_mask.png'
            ih = hashlib.sha256(('PROCEDURAL_IMAGE_'+sid).encode()).hexdigest()
            mh = hashlib.sha256(('PROCEDURAL_MASK_'+sid).encode()).hexdigest()
            gid = hashlib.sha256((ih + mh).encode()).hexdigest()
            row = dict(sample_id=task+':'+relative, source_split='train', image_sha256=ih, mask_sha256=mh)
            groups.append(gid); records.append(row)
            manifest.append(dict(sample_id=row['sample_id'], domain=source, split='train',
                image_path=str(root / relative), mask_path=str(root / mask), image_size=[9, 7],
                image_sha256=ih, mask_sha256=mh))
            csv_rows.append(dict(image=relative, mask=mask))
        sections[name] = dict(group_ids=groups, records=records)
    artifact = dict(schema_version='crisp-source-content-split-v1', task=task, source=source,
                    counts={n:dict(groups=len(s['group_ids']),samples=len(s['records'])) for n,s in sections.items()}, splits=sections)
    def save_split():
        artifact.pop('split_payload_sha256', None)
        artifact['split_payload_sha256'] = hashlib.sha256(json.dumps(artifact,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        (root / 'split.json').write_text(json.dumps(artifact))
    save_split()
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with (root / f'{source}_train.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image','mask']); writer.writeheader(); writer.writerows(csv_rows)
    return [root/'manifest.json', root/'split.json', [dict(name=f'{source}_train.csv',split='train')], root, task], artifact, manifest, save_split


class SourceIOTests(unittest.TestCase):
    def test_sdf_brute_pixel_centers_non_square_small_nested_empty_full(self):
        for h,w in ((3,5),(7,4),(9,11)):
            mask = torch.zeros(2,2,h,w)
            mask[0,0,1:-1,1:-1]=1; mask[0,1,h//2,w//2]=1
            mask[1,1]=1
            distance, defined = signed_distance(mask)
            self.assertEqual(defined.tolist(), [[True,True],[False,False]])
            yy,xx=np.indices((h,w)); coords=np.stack((yy,xx),-1)
            for c in range(2):
                a=mask[0,c].numpy().astype(bool)
                expected=np.empty((h,w))
                for y in range(h):
                    for x in range(w):
                        points=coords[a != a[y,x]]
                        expected[y,x]=np.sqrt(((points-[y,x])**2).sum(1)).min() * (-1 if a[y,x] else 1)/math.hypot(h,w)
                np.testing.assert_allclose(distance[0,c],expected,rtol=1e-6)
            self.assertEqual(float(distance[1].abs().sum()),0.)
        EVIDENCE['sdf']={'reference':'independent brute opposite-class pixel-center distance','shapes':[[3,5],[7,4],[9,11]],'nested_channels':True,'empty_full_zero_undefined':True}

    def test_rgb_mask_decode_nearest_and_geometry(self):
        gray=np.array([[255,128,0],[255,255,0]],dtype=np.uint8)
        rgb=np.stack((gray,255-gray,gray//2),-1)
        for task,size in INPUT_SIZES.items():
            actual=read_pixels(encoded(rgb),task,[3,2])
            resample=Image.Resampling.BICUBIC if task=='fundus' else Image.Resampling.BILINEAR
            expected=np.asarray(Image.fromarray(rgb).resize((size,size),resample),np.float32).transpose(2,0,1)/255
            np.testing.assert_array_equal(actual[0],expected)
            result=read_mask(encoded(gray),task,[3,2])
            scaled=np.asarray(Image.fromarray(gray).resize((size,size),Image.Resampling.NEAREST))
            expected=np.stack((scaled<255,scaled==0)) if task=='fundus' else (scaled>127)[None]
            np.testing.assert_array_equal(result[0],expected)
            if task=='fundus': self.assertTrue((result[:,1]<=result[:,0]).all())
            with self.assertRaises(ValueError): read_mask(encoded(gray),task,[4,2])
        EVIDENCE['decode']={'RGB':'BICUBIC fundus / BILINEAR polyp','GT':'NEAREST','channels':['OD<255','OC==0','polyp>127'],'real_files_read':0}

    def test_frozen_metadata_csv_join_selection_and_rejections(self):
        for task in SOURCES:
            with tempfile.TemporaryDirectory(prefix='dpa_procedural_registry_') as tmp:
                args, artifact, manifest, save = mock_registration(Path(tmp),task)
                first, second=registered_source(*args),registered_source(*args)
                self.assertEqual(first,second)
                self.assertEqual(len(first['proxy']),4)
                self.assertEqual(len(first['query']),20 if task=='fundus' else 32)
                self.assertFalse({r['group_id'] for r in first['proxy']} & {r['group_id'] for r in first['query']})
                with patch('dpa_ctta.source_io.read_pixels',side_effect=AssertionError('query entered proxy')):
                    with self.assertRaises(ValueError): source_proxy(first['query'][:4],task)
                # Exercise the positive registered proxy reader chain with encoded geometry,
                # replacing only path resolution at this test boundary; no real images exist.
                gray=np.full((7,9),255,np.uint8); gray[1:6,2:7]=128; gray[3,4]=0
                images={r['image_path']:np.stack((gray,255-gray,gray//2),-1) for r in first['proxy']}
                masks={r['mask_path']:gray for r in first['proxy']}
                with patch('dpa_ctta.source_io.read_pixels',side_effect=lambda path,task,size:read_pixels(encoded(images[path]),task,size)) as rp, \
                     patch('dpa_ctta.source_io.read_mask',side_effect=lambda path,task,size:read_mask(encoded(masks[path]),task,size)) as rm:
                    prepared=source_proxy(first['proxy'],task)
                self.assertEqual(rp.call_count,4); self.assertEqual(rm.call_count,4)
                self.assertEqual(len(prepared.pixel_rgb),4)
                expected,flags=signed_distance(prepared.mask)
                self.assertTrue(torch.equal(prepared.signed_distance,expected))
                self.assertTrue(flags.all())
                EVIDENCE[task+'_registered_proxy_io']={'fixture_only':True,'basis_groups':4,'image_reads':4,'mask_reads':4,'final_grid_sdf_matches':True,'query_reads':0}
                original=copy.deepcopy(artifact)
                artifact['splits']['critic_validation']['records'][0]['mask_sha256']='0'*64; save()
                with self.assertRaises(ValueError): registered_source(*args)
                artifact.clear(); artifact.update(original); save()
                manifest[0]['domain']='FORBIDDEN_TARGET'
                Path(args[0]).write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): registered_source(*args)
                args, artifact, manifest, save=mock_registration(Path(tmp),task)
                artifact['splits']['critic_validation']['group_ids'].append(artifact['splits']['basis_train']['group_ids'][0]); save()
                with self.assertRaises(ValueError): registered_source(*args)
                args, artifact, manifest, save=mock_registration(Path(tmp),task)
                csvpath=Path(tmp)/args[2][0]['name']
                csvpath.write_text('image,mask\n../FORBIDDEN_TARGET/a.png,../FORBIDDEN_TARGET/b.png\n')
                with self.assertRaises(ValueError): registered_source(*args)
        EVIDENCE['registration']='two temporary procedural manifests, deterministic selection; role/content/path/overlap mismatch rejected; no real data registration'

    def test_pixel_transforms_and_metrics_pair_cohorts(self):
        x=pixels('polyp')
        self.assertTrue(torch.equal(transform_pixels(x,'gamma_0.7'),x**.7))
        blur=transform_pixels(x,'blur_5_sigma_1')
        self.assertEqual(blur.shape,x.shape)
        self.assertTrue((blur>=0).all() and (blur<=1).all())
        mask=torch.zeros(1,1,5,7); mask[:,:,1:4,2:5]=1
        logits=(mask*2-1)*8
        met=evaluate_after_step(logits,mask,'polyp')[0]
        self.assertEqual(met['dice'],1.); self.assertEqual(met['assd'],0.)
        empty=evaluate_after_step(logits,torch.zeros_like(mask),'polyp')[0]
        self.assertIsNone(empty['assd']); self.assertFalse(empty['boundary_defined'])
        row=dict(group_id='PROCEDURAL_G',sample_id='PROCEDURAL_S',segment='clean',metrics=[met])
        results={a:[copy.deepcopy(row)] for a in ('A','B','C')}
        # All four segments must be present for complete comparisons.
        for arm in results:
            results[arm]=[dict(copy.deepcopy(row),segment=s) for s in ('clean','gamma_0.7','gamma_1.5','blur_5_sigma_1')]
        summary=summarize(results)
        self.assertEqual(summary['all']['C-B']['polyp']['visits'],4)
        self.assertEqual(summary['all']['C-B']['polyp']['independent_groups'],1)

    def test_every_real_entry_rejects_before_config_or_io(self):
        with patch('builtins.print'), patch('torch.load',side_effect=AssertionError('checkpoint read')):
            self.assertEqual(main([]),0)
            self.assertEqual(main(['dry-run']),0)
            self.assertEqual(main(['run','--config','DOES_NOT_EXIST']),2)
            with self.assertRaisesRegex(RuntimeError,APPROVAL_REQUIRED): run_registered({'enabled':True})
        EVIDENCE['real_entry']={'prepare_exit':0,'dry_run_exit':0,'run_exit':2,'direct_run_registered':APPROVAL_REQUIRED}

    def test_nonfinite_and_label_mismatch_stop_before_next_query(self):
        from types import SimpleNamespace
        for nonfinite in (True, False):
            step = unittest.mock.Mock(return_value=torch.full((1,1,5,7),float('nan') if nonfinite else 0.))
            host = SimpleNamespace(device=torch.device('cpu'), step=step)
            evaluator = unittest.mock.Mock(side_effect=lambda pred,row: evaluate_after_step(pred,torch.zeros(1,2,5,7),'polyp'))
            with self.assertRaises(ValueError):
                _run_arm(host,[dict(group_id='PROCEDURAL_1',sample_id='PROCEDURAL_1'),dict(group_id='PROCEDURAL_2',sample_id='PROCEDURAL_2')],
                         lambda row:pixels('polyp'),evaluator)
            self.assertEqual(step.call_count,1)
            self.assertEqual(evaluator.call_count,0 if nonfinite else 1)
        EVIDENCE['failure_stop']='nonfinite skips labels; mismatched labels stop after first completed step; no second query'


@unittest.skipUnless(EXTERNAL,'pinned CTTA checkout required')
class SourceHostTests(unittest.TestCase):
    def test_checkpoint_clone_strict_source_only_and_initial_guard(self):
        for task in INPUT_SIZES:
            torch.manual_seed(212)
            original=VPTTAHost(task)
            state={k:t.clone() for k,t in original.model.state_dict().items()}
            # Synthetic source, deliberately different from ALL fresh RNG construction.
            for k,t in state.items():
                if k.endswith('running_mean'): t.fill_(.25)
                elif k.endswith('running_var'): t.fill_(1.5)
            key=next(k for k,t in state.items() if t.ndim==4)
            state[key].add_(.001)
            host=VPTTAHost(task,source_state=state,device='cpu:0',mode='proxy_rehearsal',extra_weight=.1,proxy_factory=lambda:proxy(task))
            same(self,state,host.model.state_dict()); same(self,state,host._proxy_model.state_dict())
            self.assertFalse(torch.equal(original.model.state_dict()[key],host.model.state_dict()[key]))
            self.assertTrue(all(t.data_ptr()!=host._proxy_model.state_dict()[k].data_ptr() for k,t in host.model.state_dict().items()))
            self.assertIs(host.optimizer.param_groups[0]['params'][0],host.prompt.data_prompt)
            host.audit_initial_state()
            with torch.no_grad(): host.model.state_dict()[key].add_(.01)
            with self.assertRaisesRegex(RuntimeError,'SOURCE_CLONE_STATE_MISMATCH'): host.step(pixels(task))
            bad=dict(state); bad.pop(key)
            with self.assertRaises(ValueError): load_source_state(original.model,bad)
            bad=dict(state); bad[key]=state[key].flatten()
            with self.assertRaises(ValueError): load_source_state(original.model,bad)
            bad=dict(state); bad[key]=torch.full_like(state[key],float('nan'))
            with self.assertRaises(ValueError): load_source_state(original.model,bad)
            del host, bad, original
            n=SourceOnlyHost(task,state)
            reference,logits=build_reference_model(task); load_source_state(reference,state)
            with torch.no_grad(): expected=logits(reference,model_input_from_pixels(pixels(task),task))
            self.assertTrue(torch.equal(n.step(pixels(task)),expected))
            self.assertTrue(all(type(m) is torch.nn.BatchNorm2d for m in n.model.modules() if isinstance(m,torch.nn.BatchNorm2d)))
            bn=next(m for m in n.model.modules() if isinstance(m,torch.nn.BatchNorm2d))
            x=torch.linspace(-1,1,bn.num_features*6).reshape(1,bn.num_features,2,3)
            self.assertTrue(torch.allclose(bn(x),(x-.25)/math.sqrt(1.5+bn.eps)*bn.weight[None,:,None,None]+bn.bias[None,:,None,None],atol=1e-6))
            EVIDENCE[task+'_assembly']={'in_memory_synthetic_state':'noninitial conv and running stats','strict_keys_shapes_dtype_finite':True,'clone_equal_disjoint_frozen':True,'load_only_live_rejected':True,'source_only_standard_bn_reference':True,'optimizer_current_cpu_prompt':True,'cuda':'NOT_RUN'}
            del n,reference,state

    def test_default_45_unique_keys_native_equivalence_retrieval_eviction(self):
        for task in INPUT_SIZES:
            torch.manual_seed(113); direct=VPTTAHost(task)
            torch.manual_seed(113); wrapped=VPTTAHost(task)
            source={k:t.clone() for k,t in wrapped.model.state_dict().items()}
            retrieval, evictions, pushed, milestones=[],[],[],{}
            original_get=wrapped.memory_bank.get_neighbours
            original_push=wrapped.memory_bank.push
            index=0
            def get(*args,**kwargs):
                retrieval.append(index+1)
                return original_get(*args,**kwargs)
            def push(*args,**kwargs):
                before=set(wrapped.memory_bank.memory)
                result=original_push(*args,**kwargs)
                after=set(wrapped.memory_bank.memory)
                pushed.extend(after-before)
                if before-after: evictions.append(index+1)
                return result
            with patch.object(wrapped.memory_bank,'get_neighbours',get),patch.object(wrapped.memory_bank,'push',push):
                for index in range(45):
                    x=pixels(task,index)
                    rng=torch.get_rng_state().clone()
                    expected=direct.native_step(direct,model_input_from_pixels(x,task))
                    post=torch.get_rng_state().clone(); torch.set_rng_state(rng)
                    actual=wrapped.step(x)
                    self.assertTrue(torch.equal(expected,actual))
                    self.assertTrue(torch.equal(post,torch.get_rng_state()))
                    # Exclude instrumentation attributes from the otherwise native memory state.
                    a,b=native_snapshot(direct),native_snapshot(wrapped)
                    b['memory'].pop('get_neighbours',None); b['memory'].pop('push',None)
                    same(self,a,b)
                    if index in (0,4,5,15,16,39,40,41,44):
                        milestones[index+1]=dict(counts=sorted({m.sample_num for m in wrapped.model.modules() if isinstance(m,wrapped.adabn)}),memory=wrapped.memory_bank.get_size())
            self.assertEqual(retrieval,list(range(17,46)))
            self.assertEqual(evictions,[42,43,44,45])
            self.assertEqual(len(set(pushed)),45)
            same(self,source,wrapped.model.state_dict())
            self.assertEqual(int(wrapped.optimizer.state[wrapped.prompt.data_prompt]['step']),45)
            EVIDENCE[task+'_45_native']={'images':45,'distinct_low_frequency_keys':len(set(pushed)), 'neighbor':16,'retrieval_steps':retrieval,'eviction_steps':evictions,'milestones':milestones,'direct_wrapper_logits_state_rng_equal_each_step':True,'source_unchanged':True,'adam_steps':45,'capacity_parameter':40,'observed_final_size':41}
            del direct,wrapped,source

    def test_four_proxy_full_batch_b_c_gradients(self):
        for task in INPUT_SIZES:
            one=proxy(task)
            four=FixedProxy(torch.cat([pixels(task,i+50) for i in range(4)]),one.mask.repeat(4,1,1,1),
                            one.signed_distance.repeat(4,1,1,1),ProxyProvenance.FIXTURE)
            for arm in ('B','C'):
                torch.manual_seed(913)
                host=VPTTAHost(task,mode='proxy_rehearsal',extra_weight=.1,beta_boundary=0. if arm=='B' else .1,
                               proxy_factory=lambda:four)
                sizes=[]
                handle=host._proxy_model.register_forward_pre_hook(lambda module,args:sizes.append(len(args[0])))
                host.step(pixels(task,75))
                handle.remove()
                self.assertEqual(sizes,[4])
                self.assertGreater(float(host.prompt.data_prompt.grad.norm()),0.)
                self.assertEqual(int(host.optimizer.state[host.prompt.data_prompt]['step']),1)
                self.assertEqual({m.sample_num for m in host.model.modules() if isinstance(m,host.adabn)},{1})
                EVIDENCE[task+'_'+arm+'_K4']={'proxy_forward_batches':sizes,'prompt_gradient_norm':float(host.prompt.data_prompt.grad.norm()),'native_count':1,'adam_step':1}
                del host

    def test_fixture_query_replay_labels_and_enabled_b_c_lifecycle(self):
        for task in INPUT_SIZES:
            torch.manual_seed(381); base=VPTTAHost(task)
            source={k:t.clone() for k,t in base.model.state_dict().items()}; del base
            fixed=proxy(task)
            rows=[dict(sample_id='PROCEDURAL_QUERY',group_id='PROCEDURAL_GROUP')]
            observed=[]
            for inverted in (False,True):
                host=assemble(task,'C',source,fixed)
                predictions=[]; masks_read=[]
                def capture(pred, current):
                    predictions.append(pred.clone())
                    self.assertEqual(len(masks_read),len(predictions)-1)
                def evaluator(pred,row):
                    masks_read.append(row['sample_id'])
                    return evaluate_after_step(pred,1-fixed.mask if inverted else fixed.mask,task)
                records=_run_arm(host,rows,lambda row:pixels(task,3),evaluator,capture=capture)
                observed.append((records,predictions,copy.deepcopy(host.prompt.state_dict()),copy.deepcopy(host.optimizer.state_dict())))
                self.assertEqual(int(host.optimizer.state[host.prompt.data_prompt]['step']),4)
                self.assertEqual({m.sample_num for m in host.model.modules() if isinstance(m,host.adabn)},{4})
                same(self,source,host.model.state_dict())
                self.assertGreater(float(host.prompt.data_prompt.grad.norm()),0)
                del host
            same(self,observed[0][1:],observed[1][1:])
            self.assertEqual([(r['sample_id'],r['segment']) for r in observed[0][0]],[(r['sample_id'],r['segment']) for r in observed[1][0]])
            self.assertNotEqual([r['metrics'] for r in observed[0][0]],[r['metrics'] for r in observed[1][0]])
            b=assemble(task,'B',source,fixed)
            before=native_snapshot(b)
            region=b._proxy_term(); grad,=torch.autograd.grad(region,b.prompt.data_prompt)
            self.assertGreater(float(grad.norm()),0); same(self,before,native_snapshot(b))
            b.step(pixels(task,4))
            self.assertEqual(int(b.optimizer.state[b.prompt.data_prompt]['step']),1)
            self.assertEqual({m.sample_num for m in b.model.modules() if isinstance(m,b.adabn)},{1})
            same(self,source,b.model.state_dict())
            EVIDENCE[task+'_fixture_replay']={'visits_per_replay':4,'replays':2,'same_order_predictions_prompt_adam':True,'changed_fake_labels_changed_metrics_only':True,'B_C_nonzero_prompt_gradients':True,'source_weights_unchanged':True,'B_auxiliary_no_live_count_step_push':True,'C_counts_and_adam':4,'NOT_REAL_DATA':True}
            del b,source,observed
