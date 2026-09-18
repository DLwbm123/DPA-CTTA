"""Procedural CPU fixtures; never loads torch or any real image/model asset."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from core import (analysis, bins, decompose, digest, disjoint, divide, functions, local_values,
                  pair, partition, public_safe, quantile, summarize, timeline, validators)
from run import check_snapshot, source_inventory, source_record

ROOT=Path(__file__).resolve().parents[2]


def fixture(v,visit=1):
    channels=[]
    for c,ch in enumerate(('OD','OC')):
        seed=int.from_bytes(hashlib.sha256(f'R6_WEIGHT_PERM_V1|20260907|{visit}|{c}'.encode()).digest()[:8],'big')&((1<<63)-1)
        z=dict(channel=ch,n_fg=0,n_bg=262144,rho=1.,rho_clipped=False,fallback='ONE_PARTITION_EMPTY',
            w_fg=1.,w_bg=1.,applied_w_fg=1.,applied_w_bg=1.,mean_weight=1.,a=1.,b=1.,seed=seed,
            permutation_histogram_preserved=True,permutation_changed_positions=0,base_weight_sha256='0'*64,shuffled_weight_sha256='0'*64,
            weight_squared_sum=262144)
        for k in ['S0','Sw','Sperm','residual_fg_sse','residual_bg_sse','weighted_fg_sse','weighted_bg_sse','bce_sum','bce_fg_sum','bce_bg_sum',
                  'weighted_bce_sum','shuffled_bce_sum','residual_weighted_dot','weight_residual_dot','residual_shuffled_dot','expected_logit_gradient_l2','actual_logit_gradient_l2']:z[k]=0.
        channels.append(z)
    metric=lambda ch:dict(channel=ch,pred_pixels=0,gt_pixels=0,total_pixels=262144,intersection=0,dice=1.,gt_empty=True,gt_full=False,pred_empty=True,pred_full=False,assd=None)
    row=dict(binding={'test':1},visit=visit,group_id='g'+str(visit),sample_id='s'+str(visit),domain='A',subset='remaining_dev',
        trace=dict(visit=visit,arm='C',counts=v['PHYSICAL'],cumulative={k:n*visit for k,n in v['PHYSICAL'].items()},adam_step=visit,source_unchanged=True,
            transaction_complete=True,channels=channels,actual_loss=0.,bn_gradient_l2=0.,adam_affine_displacement_l2=0.,host_seconds=0.))
    ev={k:x for k,x in row.items() if k!='trace'}
    ev['evaluation']=dict(transaction_before_GT=True,metrics={p:[metric(ch) for ch in ('OD','OC')] for p in ['pre','q','post']},pipeline_seconds=0.,evaluator_seconds=0.)
    return row,ev


class ScalarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.v=validators(ROOT,0)

    def test_exact_decomposition(self):
        z=decompose(70.,69.,71.,72.)
        self.assertEqual(z,dict(delta_pre=-1.,delta_step=-2.,delta_post=-3.))

    def test_weighting_domains_orders_channels(self):
        # Two unequal domains represented at equal mass; order/channel factors stay explicit.
        s=summarize([2,4,4],[.5,.25,.25],['a','b','c'],'OD',0)
        self.assertEqual(s['signed_net'],3);self.assertEqual(s['macro_contribution'],1.5);self.assertEqual(s['primary_contribution'],.75)
        self.assertEqual(s['signed_net'],s['positive_mass']+s['negative_mass'])

    def test_segment_contribution_is_not_local_mean(self):
        a=summarize([-2],[.1],['a']);b=summarize([1,1],[.45,.45],['b','c'])
        self.assertAlmostEqual(a['signed_net']+b['signed_net'],.7)

    def test_paired_quantile_not_difference_of_quantiles(self):
        self.assertNotEqual(quantile([0-0,10-9,11-100],.5),quantile([0,10,11],.5)-quantile([0,9,100],.5))

    def test_cross_stream_content_pair(self):
        a=[dict(group_id=x,sample_id=x,domain='D',subset='S',visit=i) for i,x in enumerate(['x','y'])]
        b=list(reversed(a));self.assertEqual([(x['group_id'],y['group_id']) for x,y in pair(a,b)],[('x','x'),('y','y')])

    def test_duplicate_missing_metadata(self):
        r=dict(group_id='x',sample_id='x',domain='D',subset='S')
        for a,b in [([r,r],[r]),([r],[]),([r],[dict(r,domain='E')])]:
            with self.assertRaises(ValueError):pair(a,b)

    def test_switch_chunks_all_arrivals(self):
        rows=[dict(group_id=str(i),domain=d) for i,d in enumerate('AAAABBAA')]
        chunks=[dict(domain='A',first_visit=1,count=2),dict(domain='A',first_visit=3,count=2),dict(domain='B',first_visit=5,count=2),dict(domain='A',first_visit=7,count=2)]
        tm,segs,_=timeline(rows,chunks)
        self.assertEqual(len(segs),3);self.assertFalse(tm['2']['is_switch_segment']);self.assertEqual(tm['6']['prior_domain_exposures'],4)
        self.assertEqual(tm['6']['domain_occurrence'],2);self.assertEqual(tm['6']['gap_since_domain_visit'],3)
        self.assertEqual(tm['3']['time_half'],'second')

    def test_fixed_early_late(self):
        rows=[dict(group_id=str(i),domain='A' if i==0 else 'B') for i in range(11)]
        tm,_,_=timeline(rows)
        self.assertEqual(tm['8']['timing'],'early_1_8');self.assertEqual(tm['9']['timing'],'later_9_plus')

    def test_null_denominator(self):
        self.assertEqual(divide(0,0),(None,'ZERO_DENOMINATOR'))
        t,e=fixture(self.v);r=dict(t,metrics=e['evaluation']['metrics']);v,reason=local_values(r,0)
        self.assertIsNone(v['f0']);self.assertEqual(reason['q_precision'],'ZERO_DENOMINATOR')

    def test_integer_partition_boundaries(self):
        self.assertEqual([partition(i,9) for i in [0,1,8,9]],['EMPTY_FG','UNCLIPPED','UNCLIPPED','EMPTY_BG'])
        self.assertEqual(partition(1,10),'FG_MINOR_UPPER_CAP');self.assertEqual(partition(9,10),'BG_MINOR_LOWER_CAP')

    def test_quantile_ties_sparse_missing(self):
        edges=[quantile([2]*4,q) for q in [.25,.5,.75]]
        self.assertEqual(bins(2,edges),'Q1');self.assertEqual(bins(None,edges),'MISSING')
        s=summarize([],[],[]);self.assertEqual(s['support'],'LOW_SUPPORT');self.assertIsNone(s['mean']);self.assertEqual(s['signed_net'],0)

    def test_original_scalar_join(self):
        t,e=fixture(self.v);rows=self.v['join']([t],[e],[t],dict(records=1,arm='C'),t['binding'])
        self.assertEqual(len(rows),1)

    def test_original_wrong_binding_and_missing(self):
        t,e=fixture(self.v)
        with self.assertRaises(ValueError):self.v['join']([t],[e],[t],dict(records=1,arm='C'),{})
        with self.assertRaises(ValueError):self.v['join']([t],[],[t],dict(records=1,arm='C'),t['binding'])

    def test_original_duplicate_rejection(self):
        t,e=fixture(self.v)
        with self.assertRaises(ValueError):self.v['join']([t,t],[e,e],[t,t],dict(records=2,arm='C'),t['binding'])

    def test_q_count_lower_bound(self):
        t,e=fixture(self.v);m=e['evaluation']['metrics']['q'][0]
        m.update(pred_pixels=262144,gt_pixels=262144,intersection=0,dice=0.,gt_empty=False,gt_full=True,pred_empty=False,pred_full=True,assd=0.)
        with self.assertRaises(ValueError):self.v['validate_metric'](m)

    def test_q_partition_mismatch(self):
        t,e=fixture(self.v);m=e['evaluation']['metrics']['q'][0]
        m.update(pred_pixels=1,dice=0.,pred_empty=False)
        with self.assertRaises(ValueError):self.v['join']([t],[e],[t],dict(records=1,arm='C'),t['binding'])

    def test_cross_prediction_gt(self):
        t,e=fixture(self.v);m=e['evaluation']['metrics']['post'][0];m.update(gt_pixels=1,gt_empty=False,dice=0.)
        with self.assertRaises(ValueError):self.v['join']([t],[e],[t],dict(records=1,arm='C'),t['binding'])

    def test_cross_stream_gt(self):
        t,e=fixture(self.v);r=dict(t,metrics=e['evaluation']['metrics']);other=copy.deepcopy(r);other['metrics']['pre'][0]['gt_pixels']=1
        with self.assertRaises(ValueError):self.v['validate_cross_GT']({(0,'C'):[r],(1,'C'):[other]})

    def test_original_physical_rejection(self):
        t,e=fixture(self.v);t['trace']['adam_step']=0
        with self.assertRaises(ValueError):self.v['join']([t],[e],[t],dict(records=1,arm='C'),t['binding'])

    def test_snapshot_pointer_unchanged_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'source';snapshot=p/'snapshot';source.mkdir();snapshot.mkdir()
            original=source/'current_result.json';original.write_text('{"valid":true}')
            before=source_record(original,snapshot/original.name);check_snapshot({original.name:before},snapshot)
            self.assertEqual(before,source_record(original));(snapshot/original.name).write_text('{}')
            with self.assertRaises(ValueError):check_snapshot({original.name:before},snapshot)
            self.assertEqual(before,source_record(original))

    def test_hardlink_symlink_overlap_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=p/'a.json';a.write_text('{}');b=p/'b.json';b.symlink_to(a)
            with self.assertRaises(ValueError):source_record(b)
            b.unlink();os.link(a,b)
            with self.assertRaises(ValueError):source_record(a)
            with self.assertRaises(ValueError):disjoint(p,p/'out')
            with self.assertRaises(ValueError):disjoint(p,p)

    def test_export_deidentification(self):
        for x in [{'group_id':'x'},{'p':'/home/private'},{'device_uuid':'x'}]:
            with self.assertRaises(ValueError):public_safe(x)
        public_safe(dict(domain='A',n_unique=5))

    def test_original_mutating_entrypoints_not_loadable(self):
        with self.assertRaises(PermissionError):functions(ROOT/'src/dpa_ctta/r6_regional_consistency/analyze.py',['recompute'],{})

    def test_complete_scalar_export_fixture(self):
        rows={};domains=['A','B','C','D']
        for s in (0,1,4):
            for arm in ('C','R_BAL','R_SCALE','R_SHUFFLE'):
                rs=[]
                for i,d in enumerate(domains,1):
                    t,e=fixture(self.v,i);t['domain']=d;t['trace']['arm']=arm
                    rs.append(dict(t,metrics=e['evaluation']['metrics']))
                rows[s,arm]=rs
        chunks=[dict(domain=d,first_visit=i,count=1) for i,d in enumerate(domains,1)]
        meta=dict(domain_counts={d:1 for d in domains},chunk_schedule=chunks,contiguous_domain_segments=chunks)
        spec=dict(inputs=dict(remaining_dev_domain_counts={d:1 for d in domains}),analysis=dict(D3=dict(unavailable=[])))
        with tempfile.TemporaryDirectory() as d:
            result=analysis(rows,meta,spec,Path(d)/'public')
            self.assertEqual(result['joined_rows'],108)
            self.assertTrue(all(n>0 for n in result['table_rows'].values()))

    def test_guard_denies_model_gpu_and_source_write(self):
        code='''import tempfile, pathlib, sys
from core import install_guard
p=pathlib.Path(tempfile.mkdtemp()); out=p/'new';out.mkdir();src=p/'old.json';src.write_text('{}')
install_guard([src],[],out)
for action in [lambda: __import__('torch'),lambda: __import__('ctypes'),lambda: src.write_text('bad'),lambda: open('/dev/nvidia0','rb'),lambda: src.chmod(0o600)]:
 try: action()
 except PermissionError: pass
 else: raise AssertionError('guard did not deny')
assert src.read_text()=='{}'
print('five forbidden entries rejected; actual model/GPU calls zero')
'''
        r=subprocess.run([sys.executable,'-B','-'],input=code,text=True,capture_output=True,env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent)))
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
