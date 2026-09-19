"""Read-only deployment check. Paths/independent inventory supplied privately."""
import io,json,os,subprocess,time,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from dpa_ctta.r7_source_prep.runner import configure_backend,load_model,load_artifact
from dpa_ctta.r7_source_prep.registry import verified
from dpa_ctta.r7_shared.context import environment
from dpa_ctta.r7_shared.numerics import COUNTS

def main():
    torch.set_num_threads(2);configure_backend(dict(device='cuda:0'))
    cfg=json.loads(Path(os.environ['ROUNDTRIP_CONFIG']).read_text())
    declared=json.loads(Path(cfg['declared_inventory']).read_text())
    inventory=declared['artifacts'];cp=cfg['checkpoint'];started=time.monotonic()
    result=dict(status='FAILED',code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),artifacts=[],deserializations=0,real_pixel_reads=0,external_review='NOT_RUN')
    before=COUNTS.copy();original_load=torch.load
    def load(source,*a,**kw):
        if not isinstance(source,io.BytesIO):raise AssertionError('verified byte buffers only')
        result['deserializations']+=1
        return original_load(source,*a,**kw)
    def forbidden(*a,**kw):raise AssertionError('no images or model execution during roundtrip')
    try:
        with patch('torch.load',side_effect=load),patch.object(Image,'open',side_effect=forbidden):
            for name in sorted(inventory):
                raw=verified(cp['path'],cp['sha256'],256*1024**2)
                assert len(raw)==cp['bytes']
                model=load_model(raw,device='cuda:0')
                guard=model.register_forward_pre_hook(forbidden)
                try:
                    actual=environment(model)
                    assert actual['effective_state_sha256']==declared['effective_state_sha256']
                    host=load_artifact(model,name,cfg['artifact_root'],inventory)
                    host._check_frozen(boundary=True)
                    assert host.visits==0
                    result['artifacts'].append(dict(name=name,**inventory[name],effective_state_sha256=actual['effective_state_sha256'],trusted_loader='PASSED'))
                    result['execution_backend']=actual['execution_backend']
                    del host
                finally:guard.remove();model.close()
                del model
        assert len(result['artifacts'])==6 and result['deserializations']==12
        assert not dict(COUNTS-before)
        result['status']='PASSED'
    except BaseException as exc:
        result['first_error']=dict(type=type(exc).__name__,message=str(exc));traceback.print_exc()
    finally:
        result.update(wall_seconds=time.monotonic()-started,observed_counts=dict(COUNTS-before))
        Path(os.environ['ROUNDTRIP_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    raise SystemExit(result['status']!='PASSED')

if __name__=='__main__':main()
