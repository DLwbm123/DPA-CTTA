"""Bounded random-network qualification; no real data/checkpoint or source run."""
import copy,io,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/r7'))
from common import segmenter,pixels,method,source_fixture
from dpa_ctta.r7_source_prep.runner import configure_backend,device_policy
from dpa_ctta.r7_shared.network import Segmenter
from dpa_ctta.r7_shared.source import SourceTrainer,oracle_one
from dpa_ctta.r7_shared.context import environment,tensor_digest,tensors
from dpa_ctta.r7_shared.preparation import prepared_artifact,inference_from_tensors
from dpa_ctta.r7_shared.numerics import COUNTS

def main():
    torch.set_num_threads(2)
    r=dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),status='FAILED',failures=0,checks=[],physical_GPU_ids=[int(os.environ['CUDA_VISIBLE_DEVICES'])],real_source_training=False,real_asset_reads=0,external_review='NOT_RUN',python=platform.python_version(),torch=torch.__version__)
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=root,text=True).strip()
    start=time.monotonic();before=COUNTS.copy();s=None;g=None
    def check(name,**evidence):r['checks'].append(dict(name=name,**evidence));print(json.dumps(r['checks'][-1]),flush=True)
    try:
        configure_backend(dict(device='cuda:0'))
        r['GPU_name']=torch.cuda.get_device_name(0);torch.cuda.reset_peak_memory_stats()
        with patch('torch.load',side_effect=AssertionError('real checkpoints forbidden')):
            s=segmenter(full=True);g=segmenter(full=True).to('cuda:0')
        original=tensor_digest(tensors(g.model));x=pixels();v=torch.zeros(1024,requires_grad=True)
        t=time.monotonic();zc,rc,ec=s(x,v,observe=True);gc=torch.autograd.grad(zc.square().mean(),v)[0];cpu_seconds=time.monotonic()-t
        t=time.monotonic();zg,rg,eg=g(x,v,observe=True);gg=torch.autograd.grad(zg.square().mean(),v)[0];gpu_seconds=time.monotonic()-t
        # Fixed acceptance tolerances declared before observing this comparison.
        for name,a,b in [('logits',zc,zg),('raw',rc,rg),('tokens',ec,eg),('FiLM_gradient',gc,gg)]:
            torch.testing.assert_close(a,b,rtol=3e-3,atol=3e-4)
            check('CPU_GPU_'+name,max_abs=float((a-b).abs().max()),rtol=3e-3,atol=3e-4)
        assert gg.norm()>0 and gg[:512].norm()>0 and gg[512:].norm()>0
        assert all(t.device.type=='cpu' for t in (zg,rg,eg,gg))
        check('gradient_bridge_and_CPU_method_boundary',CPU_forward_backward_seconds=cpu_seconds,GPU_forward_backward_seconds=gpu_seconds)
        with torch.no_grad():a=g(x);b=g(x)
        assert torch.equal(a,b);check('same_GPU_repeat_exact')
        assert environment(g)!=environment(s);check('backend_context_distinct')
        del zc,zg,rc,rg,ec,eg,gc,gg,a,b;s.close();s=None
        data,oracles=source_fixture()
        count=COUNTS.copy();oracle,_=oracle_one(g,data,'fit',0)
        assert torch.isfinite(oracle).all() and oracle.norm()>0
        delta=COUNTS-count;assert delta['backbone_forwards']==32 and delta['source_backward_calls']==16 and delta['source_Adam']==16
        check('full_16_step_procedural_oracle',counts=dict(delta))
        for group in 'ABC':
            for static in (False,True):
                m=method(group,static);t=SourceTrainer(g,m,data,oracles['fit']);count=COUNTS.copy()
                loss,_=t.fit_step();assert any(v>0 for v in t.gradients.values())
                t.start_calibration(oracles['cal'],procedural_micro=True);cal=t.cal_step();m.freeze()
                assert t.fit_steps==1 and t.cal_steps==1
                package=prepared_artifact(g,m);weights=package['weights'];buf=io.BytesIO();torch.save(weights,buf);buf.seek(0)
                weights=torch.load(buf,map_location='cpu',weights_only=True)
                inference_from_tensors(g,group,static,weights['basis'],weights,package['binding'])
                delta=COUNTS-count;assert delta['backbone_forwards']==16 and delta['source_backward_calls']==1 and delta['source_AdamW']==1 and delta['calibration_backward_calls']==1 and delta['calibration_Adam']==1
                check(group+('_STATIC' if static else '_FULL')+'_fit_cal_artifact',fit_loss=loss,cal_loss=cal,counts=dict(delta))
                del t,m,weights,package
        v=torch.zeros(1024,requires_grad=True);z=g(x,v);pool=torch.nn.functional.adaptive_avg_pool2d(z,(4,4)).flatten()
        for i in range(2):
            grad=torch.autograd.grad(pool[i],v,retain_graph=i==0)[0];COUNTS['source_VJP']+=1;assert torch.isfinite(grad).all() and grad.norm()>0
        assert tensor_digest(tensors(g.model))==original and not any(p.grad is not None for p in g.model.parameters())
        check('VJP_and_frozen_backbone',VJP=2)
        r['status']='PASSED'
    except BaseException as exc:
        r['failures']=1;r['first_error']=dict(type=type(exc).__name__,message=str(exc));traceback.print_exc()
    finally:
        if s is not None:s.close()
        if g is not None:g.close()
        r.update(wall_seconds=time.monotonic()-start,counts=dict(COUNTS-before),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
        Path(os.environ['GPU_QUALIFICATION_RESULT']).write_text(json.dumps(r,indent=2)+'\n');print('GPU_QUALIFICATION '+json.dumps(r),flush=True)
    raise SystemExit(r['failures'])
if __name__=='__main__':main()
