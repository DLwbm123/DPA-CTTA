"""Real-source equality check for independent oracle partitions; no target labels."""
import json,os,time
from pathlib import Path
import torch
from dpa_ctta.r8_ba.inputs import bind_metadata,open_source
from dpa_ctta.r8_ba.oracles import oracle_one
from dpa_ctta.r8_ba.oracle_shards import partition
from dpa_ctta.r7_shared.numerics import COUNTS


def main():
    cfg=json.loads(Path(os.environ['R8_PROFILE_CONFIG']).read_text());start=time.monotonic();COUNTS.clear()
    result=dict(schema='R8_SCREEN_ORACLE_EQUIVALENCE_V1',code_sha=cfg['code_sha'],physical_gpu=cfg['physical_gpu'],status='RUNNING')
    def guard():
        if time.monotonic()-start>=1200 or COUNTS['backbone_forwards']>2400:
            raise RuntimeError('R8 bounded oracle equality profile limit')
    try:
        bound=bind_metadata(cfg['refs'])
        with open_source(bound,cfg['source_root'],cfg['checkpoint_path'],0.3,cfg['physical_gpu'],256*1024**2,2*1024**3,guard) as (data,model,io):
            hook=model.register_forward_pre_hook(lambda *_:guard())
            try:
                reference={(fold,index):oracle_one(model,data,fold,index) for fold,index in [('fit',0),('val',0)]}
                for shard,fold,index in [(2,'val',0),(0,'fit',0)]:
                    with partition(shard):
                        actual=oracle_one(model,data,fold,index)
                    expected=reference[fold,index]
                    if not torch.equal(actual[0],expected[0]) or actual[1:]!=expected[1:]:
                        raise ValueError('R8 serial/sharded oracle numerical difference')
                result['status']='EXACT_SERIAL_PARTITION_MATCH'
                result['anchors']=[['fit',0],['val',0]]
                result['steps_per_call']=256
            finally:hook.remove()
    except BaseException as exc:
        result.update(status='FAILED',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,physical_counts=dict(COUNTS))
        with Path(cfg['output']).open('x') as f:json.dump(result,f,sort_keys=True,indent=2)


if __name__=='__main__':main()
