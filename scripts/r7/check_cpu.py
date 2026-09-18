"""Measured CPU suite; random tensors only, no asset loaders or GPU discovery."""
import os,sys,time,json,platform,resource,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tests/r7'),str(root/'docs/review/r7/input/references')]
torch.set_num_threads(2)
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.b1_host import Host as Old
actual=dict(Tensor_backward=0,autograd_grad=0,Adam_step=0,AdamW_step=0,old_backbone_forwards=0)
for cls,method,key in [(torch.Tensor,'backward','Tensor_backward'),(torch.autograd,'grad','autograd_grad'),(torch.optim.Adam,'step','Adam_step'),(torch.optim.AdamW,'step','AdamW_step'),(Old,'_forward','old_backbone_forwards')]:
    original=getattr(cls,method)
    def wrapped(*a,_fn=original,_key=key,**kw):
        actual[_key]+=1;return _fn(*a,**kw)
    setattr(cls,method,wrapped)
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
with ExitStack() as stack:
    for name in ('_lazy_init','init','device_count','get_device_properties'):
        stack.enter_context(patch('torch.cuda.'+name,side_effect=AssertionError('GPU forbidden')))
    # Existing CPU legacy seed helpers ask availability; return a fixed CPU result
    # without calling any CUDA discovery API or driver.
    stack.enter_context(patch('torch.cuda.is_available',return_value=False))
    stack.enter_context(patch('torch.load',side_effect=AssertionError('checkpoint reads forbidden')))
    for name in ('dpa_ctta.source_io.read_pixels','dpa_ctta.source_io.read_mask','dpa_ctta.r1.assets.checkpoint','dpa_ctta.r1.assets.target','dpa_ctta.source_io.source_proxy'):
        stack.enter_context(patch(name,side_effect=AssertionError('real assets forbidden')))
    modules=os.environ.get('R7_TESTS','test_math,test_contract,test_full_network').split(',')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
after=resource.getrusage(resource.RUSAGE_SELF)
summary=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),exit_code=int(not result.wasSuccessful()),wall_seconds=time.monotonic()-start,cpu_user_seconds=after.ru_utime-before.ru_utime,cpu_system_seconds=after.ru_stime-before.ru_stime,peak_RSS=after.ru_maxrss,peak_RSS_unit='bytes' if sys.platform=='darwin' else 'KiB',python=platform.python_version(),torch=torch.__version__,platform=platform.system()+' '+platform.machine(),physical_calls=actual,procedural_counts=dict(COUNTS),real_asset_reads=0,GPU_queries_or_initializations=0,external_review='NOT_RUN')
print('R7_CPU_RESULT '+json.dumps(summary,sort_keys=True),flush=True)
if os.environ.get('R7_RESULT'):Path(os.environ['R7_RESULT']).write_text(json.dumps(summary,indent=2)+'\n')
sys.exit(summary['exit_code'])
