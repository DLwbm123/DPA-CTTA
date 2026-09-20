"""GPU-first finite qualification; synthetic model/input only."""
import io, json, os, subprocess, sys, time, traceback
from pathlib import Path
import torch
from unittest.mock import patch
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
root=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root/'tests/r7'))
from common import segmenter, pixels
from dpa_ctta.b1_host import Host
from dpa_ctta.r7_target_screen import runner as target
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.source_pilot import seed_all

def main():
    torch.set_num_threads(2); target.configure_backend({'device':'cuda:0'})
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    result=dict(status='FAILED', code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                arms=target.ARMS, physical_GPU_ids=[gpu], arm_devices={a:'cuda:0' for a in target.ARMS}, checks=[],
                real_pixel_reads=0, real_checkpoint_loads=0, external_review='NOT_RUN')
    before=COUNTS.copy(); started=time.monotonic(); direct=factory=repeat=None
    try:
        checkpoint = Path(os.environ['GPU_QUALIFICATION_CHECKPOINT'])
        payload = checkpoint.read_bytes()
        approved=dict(binding=dict(checkpoint=dict(path=str(checkpoint),sha256=target.digest(payload),bytes=len(payload))), config=dict(device='cuda:0',max_asset_bytes=512*1024**2))
        with patch.object(target,'verified',side_effect=lambda path,*args,**kwargs: payload):
            seed_all(20260907); direct=Host('C',state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True),device='cuda:0')
            seed_all(20260907); factory,_=target.make_host(approved,'C_BASE')
            seed_all(20260907); repeat=Host('C',state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True),device='cuda:0')
            for i in range(16):
                x=pixels(i); a,ta=direct.step(x); b,tb=factory.step(x); c,tc=repeat.step(x)
                torch.testing.assert_close(a,b,rtol=0,atol=0); torch.testing.assert_close(a,c,rtol=0,atol=0)
                assert ta['counts']==tb['counts']==tc['counts']==dict(forwards=8,backwards=1,base_adam=1,perturb=0,restore=0)
                assert all(torch.equal(direct.model.state_dict()[n],factory.model.state_dict()[n]) for n in direct.model.state_dict())
                result['checks'].append(dict(name='C_BASE_GPU_reference_factory_repeat',visit=i+1,exact=True))
            g=target.load_model(payload, device='cuda:0')
            for arm in target.ARMS[1:]:
                if arm=='C0':
                    h=target.OnlineHost(g)
                    for i in range(3): target.physical(h.step(pixels(i))[1],arm)
                    h.segmenter.close()
                result['checks'].append(dict(name=arm+'_GPU_state_and_counter',visits=3 if arm=='C0' else 0))
        result['status']='PASSED'; result['execution_backend']=dict(schema='R7_CUDA_FP32_BACKBONE_CPU_METHOD_V1',torch=torch.__version__,cuda=torch.version.cuda,deterministic=torch.are_deterministic_algorithms_enabled()); result['counts']=dict(forwards=COUNTS['backbone_forwards']-before['backbone_forwards'],backwards=COUNTS['backwards']-before['backwards'],Adam=COUNTS['Adam']-before['Adam'])
    except BaseException as exc:
        result['first_error']=dict(type=type(exc).__name__,message=str(exc)); traceback.print_exc()
    finally:
        for h in (direct,factory,repeat):
            if h is not None:
                for handle in getattr(h,'handles',[]): handle.remove()
        result.update(wall_seconds=time.monotonic()-started,observed_counts=dict(COUNTS-before))
        Path(os.environ['GPU_QUALIFICATION_RESULT']).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
    raise SystemExit(result['status']!='PASSED')
if __name__=='__main__': main()
