import hashlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import torch
from dpa_ctta.r8_ba.scope import SCREEN,GRAPH_PATH,SPEC_PATH
from dpa_ctta.r8_ba.screen import tasks
from dpa_ctta.r8_ba.trainer import MAX_STEPS,SAVE_STEPS,lr_at
from dpa_ctta.r8_ba.oracle_shards import merge,binding,FULL_ORDER
from dpa_ctta.r8_ba.oracle_journal import load_oracles
from dpa_ctta.r8_ba.schedule import oracle_roles
from dpa_ctta.r8_ba.journal import SourceJournal
from dpa_ctta.r8_ba.target_jobs import resolve


class ScreenTests(unittest.TestCase):
    def test_graph_reachability_and_no_missing_source_dependencies(self):
        self.assertTrue(SCREEN);self.assertEqual((MAX_STEPS,SAVE_STEPS),(4000,(1000,4000)))
        self.assertGreater(lr_at(4000),0.0002) # Original schedule prefix, not compressed cosine.
        g=json.loads(GRAPH_PATH.read_text());configs=json.loads(SPEC_PATH.read_text())['configs']
        selected={c['route']:c['id'] for c in configs}
        for job in g['jobs']:
            if job['arrivals']:resolve(job,g,configs,selected)
        work,cpu=tasks(g);pending={r['id']:r['dependencies'] for r in work};pending.update(cpu);done=set()
        while pending:
            ready={k for k,deps in pending.items() if set(deps)<=done};self.assertTrue(ready)
            done.update(ready);pending={k:v for k,v in pending.items() if k not in ready}
        self.assertEqual((len(work),sum(r['kind']=='SOURCE_JOB' for r in work)),(96,10))

    def test_source_seals_at_4000_only_with_both_selected_points(self):
        class Fake:
            binding='synthetic';source_seed=20260924;steps=0
            def digest(self):return '0'*64
            def snapshot(self):return dict(steps=self.steps,binding=self.binding,source_seed=self.source_seed,method_digest=self.digest(),method_config=('synthetic',))
        with tempfile.TemporaryDirectory() as directory:
            trainer=Fake();trainer.method=trainer;work=SourceJournal(Path(directory)/'fit',trainer);work.create()
            for step in SAVE_STEPS:trainer.steps=step;work.checkpoint()
            work.physical.write_text(''.join(json.dumps(dict(step=i,counts={}))+'\n' for i in range(1,4001)))
            receipt=work.complete();self.assertEqual(receipt['steps'],4000);self.assertEqual(set(receipt['selected_sha256']),{'1000','4000'})

    def test_full_partition_merge_and_identity_rejection(self):
        folds={f:[f+str(i) for i in range(n)] for f,n in [('fit',111),('cal',23),('val',25)]}
        common=dict(code_sha='0'*40,protocol_sha256='1'*64,refs={},checkpoint_sha256='2'*64,amplitude=0.3)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);dirs=[]
            for shard in range(3):
                root=base/str(shard);root.mkdir();dirs.append(root);identity=dict(common,shard=shard)
                results=[];logs=[];counts=Counter()
                for ordinal,(fold,index) in enumerate(FULL_ORDER[shard*256:(shard+1)*256]):
                    roles=oracle_roles(folds[fold],fold,index)
                    results.append((torch.full((1024,),float(shard*256+ordinal)),roles[:2],roles[2:],[dict(step=s) for s in (16,64,256)]))
                    for step in range(1,257):
                        c=dict(backbone_forwards=2+2*int(step in (16,64,256)),source_backward_calls=1,source_Adam=1);counts.update(c)
                        logs.append(json.dumps(dict(ordinal=ordinal,fold=fold,index=index,step=step,counts=c),sort_keys=True)+'\n')
                raw=''.join(logs).encode();(root/'physical.jsonl').write_bytes(raw)
                payload=dict(schema='R8_ORACLE_SNAPSHOT_V1',identity=dict(binding=binding(identity),amplitude=0.3),next_ordinal=256,results=results,
                             physical_bytes=len(raw),physical_sha256=hashlib.sha256(raw).hexdigest(),counts=dict(counts),cpu_rng=torch.get_rng_state(),cuda_rng=[])
                buf=io.BytesIO();torch.save(payload,buf);snapshot=buf.getvalue();(root/'checkpoint.0.pt').write_bytes(snapshot)
                snapsha=hashlib.sha256(snapshot).hexdigest();(root/'checkpoint.0.json').write_text(json.dumps(dict(sha256=snapsha,next_ordinal=256)))
                receipt=dict(schema='R8_ORACLE_SHARD_COMPLETE_V1',identity=identity,anchors=256,snapshot_sha256=snapsha,physical_sha256=hashlib.sha256(raw).hexdigest(),physical_counts=dict(counts))
                encoded=json.dumps(receipt).encode();(root/'shard_complete.json').write_bytes(encoded)
                (root/'worker_complete.json').write_text(json.dumps(dict(schema='R8_ORACLE_SHARD_WORK_COMPLETE_V1',identity=identity,shard_receipt_sha256=hashlib.sha256(encoded).hexdigest())))
            merge(base/'merged',dirs,common,folds)
            loaded=load_oracles(base/'merged',SimpleNamespace(folds=folds),binding(common),0.3)
            self.assertEqual(float(loaded['fit'].values[0,511]),511);self.assertEqual(float(loaded['val'].values[0,127]),767)
            (dirs[1]/'worker_complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'producer identity'):merge(base/'bad',dirs,common,folds)
            self.assertFalse((base/'bad').exists())
