import copy,json,os,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.b3_orders import check_balance,build_stream,ORDERS,COUNTS,ARMS,BUDGET
from dpa_ctta.b3_entry import Entry,direct,capture
from dpa_ctta.b3_analysis import reorder_N,validate_rows,recompute,COMPLETE
from dpa_ctta.b3_run import normalized_counts
from dpa_ctta.m1_run import Observed
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.source_pilot_release import source_unchanged
from dpa_ctta.integrations.ctta_suite import build_reference_model
from dpa_ctta.host_diagnostic import rng,close
from dpa_ctta.p1_analysis import evaluate
from dpa_ctta.p2_data import SUBSETS
from dpa_ctta.b2_interval import diagnostics


def fixture():
    p=torch.zeros(1,2,3,3);metrics=evaluate(p,p,'fundus');ds=dict(channels=diagnostics(p,p,p,p,p,p.bool()),bn_gradient_l2=0.,adam_update_l2=0.,zero_gradient=True,interval_loss=0.,point_bce=0.,minimum_raw_kl=0.)
    stream=[dict(domain=d,subset=s,sample_id=str(i),group_id=str(i)) for i,(d,s) in enumerate((d,s) for d in ORDERS[0] for s in SUBSETS if s!='all_dev')]
    def records(arm,o):
        rows=[]
        for i,e in enumerate(stream):
            native=dict(model_forwards=2,prompt_forwards=2+int(i>=16),online_adam=1,backward_calls=1,memory_pushes=1,retrievals=int(i>=16),proxy_forwards=0,proxy_images=0) if arm=='A' else None
            rows.append(dict(e,task='fundus',order=o,arm=arm,visit=i+1,adam_step=i+1,counts=dict(forwards=2 if arm=='A' else 8,backwards=1,base_adam=1,perturb=0,restore=0),lr=.05 if arm=='A' else 1e-4,native_counts=native,parent_counters=[i+1] if arm=='A' else None,parent_memory_size=min(41,i+1) if arm=='A' else None,diagnostics=copy.deepcopy(ds) if arm in ('U','S','I') else None,interval_diagnostics=copy.deepcopy(ds) if arm in ('U','S','I') else None,metrics=copy.deepcopy(metrics),prediction_fixed_before_label=True,source_unchanged=arm=='A',frozen_parameters_checked=arm!='A',host_seconds=.1,pipeline_seconds=.2,peak_allocated_bytes=1))
        return rows
    return stream,records


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_order_balance_identity_and_budget(self):
        self.assertEqual(check_balance()['directed_adjacent_pairs'],12)
        rows=[dict(domain=d,group_id=f'{d}:{i}') for d in ORDERS[0] for i in range(COUNTS[d])];reg=dict(tasks=dict(fundus=dict(target=rows)))
        for o in ORDERS:
            stream=build_stream(reg,o);self.assertEqual([d for d in dict.fromkeys(r['domain'] for r in stream)],list(ORDERS[o]))
            self.assertEqual({r['group_id'] for r in rows},{r['group_id'] for r in stream})
            for d in ORDERS[0]:self.assertEqual([r for r in rows if r['domain']==d],[r for r in stream if r['domain']==d])
        with self.assertRaises(ValueError):build_stream(reg,4)
        broken=copy.deepcopy(reg);broken['tasks']['fundus']['target'][0]=rows[1]
        with self.assertRaises(ValueError):build_stream(broken,2)
        self.assertEqual(BUDGET['forwards'],1951*2*(2+4*8));self.assertEqual(BUDGET['base_adam']+20,19530)

    def test_N_reuse_and_scalar_failure(self):
        stream,records=fixture();n=records('N',0)
        self.assertEqual(reorder_N(n,list(reversed(stream))),list(reversed(n)))
        with self.assertRaises(ValueError):reorder_N(n[:-1],stream)
        for arm in ARMS:
            rows=records(arm,2);self.assertTrue(validate_rows(rows,stream,2,arm,n))
            for error in ('step','counts','identity','metric'):
                bad=copy.deepcopy(rows)
                if error=='step':bad[0]['adam_step']=5
                elif error=='counts':bad[0]['counts']['forwards']=9
                elif error=='identity':bad[0]['group_id']='wrong'
                else:bad[0]['metrics'][0]['dice']=.7
                with self.assertRaises(ValueError):validate_rows(bad,stream,2,arm,n)

    def test_actual_model_entry_and_native_counts(self):
        seed_all(20260907);model,_=build_reference_model('fundus');state=copy.deepcopy(model.state_dict());del model
        x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        for arm in ARMS:
            seed_all(20260907);h=direct(arm,state,'cpu');w=Observed(h) if arm=='A' else None
            out=h.step(x);p=out if arm=='A' else out[0];saved=capture(h,arm);random_state=rng();counts=normalized_counts(h,arm,w)
            if arm=='A':source_unchanged(h,state);w.release()
            else:h.finish(state)
            del h,w
            seed_all(20260907);entry=Entry(arm,state);z,m=entry.step(x)
            close(p,z);close(saved,entry.capture());close(random_state,rng(),exact=True);close(counts,entry.counts,exact=True);entry.trainable();entry.finish(state)

    def test_neutral_git_transport(self):
        import io,tarfile
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            trace=Path(tmp)/'events';old=os.environ.get('GIT_TRACE2_EVENT');os.environ['GIT_TRACE2_EVENT']=str(trace)
            try:
                result=subprocess.run(['git','-C',str(root),'archive','HEAD','--','configs/b2_interval_consistency_v1.json'],capture_output=True,check=True)
            finally:
                if old is None:os.environ.pop('GIT_TRACE2_EVENT')
                else:os.environ['GIT_TRACE2_EVENT']=old
            with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
                self.assertEqual(archive.extractfile('configs/b2_interval_consistency_v1.json').read(),(root/'configs/b2_interval_consistency_v1.json').read_bytes())
            events=[json.loads(s) for s in trace.read_text().splitlines()]
            starts=[e['argv'] for e in events if e['event']=='start'];self.assertEqual(len(starts),2)
            self.assertFalse(any('ctta' in ' '.join(e).lower() or 'b2_' in ' '.join(e).lower() for e in starts))
            self.assertFalse(any(e.get('event')=='child_start' for e in events))

    def test_full_CPU_closeout_fixture(self):
        stream,records=fixture();n=len(stream);budget=dict(records=n*10,forwards=n*2*34,backwards=n*10,base_adam=n*10,perturb=0,restore=0)
        old={o:{a:records(a,o) for a in ('N',*ARMS,'G','O2','D4')} for o in (0,1)}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def write(name,x):(out/name).write_text(json.dumps(x))
            write('run.completion.json',dict(status='B3_RUN_COMPLETE',progress=budget,trainable={},gpu_seconds=1))
            write('smoke.completion.json',dict(status='B3_SMOKE_PASS',physical=dict(forwards=136,backwards=20,base_adam=20,perturb=0,restore=0)))
            write('CPU_validation.json',dict(exit_code=0))
            for o in (2,3):
                for a in ARMS:(out/f'fundus_{o}_{a}.jsonl').write_text('\n'.join(json.dumps(r) for r in records(a,o))+'\n')
            reg=dict(tasks=dict(fundus=dict(counts={})),execution_provenance={})
            with patch('dpa_ctta.b3_analysis.build_stream',return_value=stream),patch('dpa_ctta.b3_analysis.old_records',side_effect=lambda r,o:copy.deepcopy(old[o])),patch('dpa_ctta.b3_analysis.BUDGET',budget):
                recompute(out,reg,dict(commit='CPU_FIXTURE'))
            result=json.loads((out/'public_aggregate.json').read_text());self.assertEqual(result['status'],COMPLETE)
            self.assertEqual(result['combined']['remaining_dev']['comparisons']['I-C']['four_mean'],0)
            self.assertEqual(json.loads((out/'verification.json').read_text())['formal'],budget)
