"""Synthetic execution-layer acceptance; file decodes/loads use BytesIO only."""
import io,json,os,platform,resource,sys,time,unittest,subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
from PIL import Image
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_shared.context import HASH_COST
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses();torch.set_num_threads(2)
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/r7'))
calls=dict(backward=0,VJP=0,Adam=0,AdamW=0,synthetic_checkpoint_loads=0,synthetic_image_mask_decodes=0)
for obj,name,key in [(torch.Tensor,'backward','backward'),(torch.autograd,'grad','VJP'),(torch.optim.Adam,'step','Adam'),(torch.optim.AdamW,'step','AdamW')]:
 original=getattr(obj,name)
 def wrapped(*a,_fn=original,_key=key,**kw):calls[_key]+=1;return _fn(*a,**kw)
 setattr(obj,name,wrapped)
original_load=torch.load;original_open=Image.open

def load(source,*args,**kw):
 if not isinstance(source,io.BytesIO):raise AssertionError('only synthetic in-memory checkpoint allowed')
 calls['synthetic_checkpoint_loads']+=1;return original_load(source,*args,**kw)
def image(source,*args,**kw):
 if not isinstance(source,io.BytesIO):raise AssertionError('only encoded synthetic bytes allowed')
 calls['synthetic_image_mask_decodes']+=1;return original_open(source,*args,**kw)
from dpa_ctta.r7_source_prep import registry, runner as source_runner
from dpa_ctta.r7_target_screen import runner as screen_runner
original_verified=registry.verified

def procedural_verified(path,*args,**kwargs):
 p=Path(path).absolute()
 if not any(a.name.startswith('screen-procedural-') for a in p.parents) and not p.is_relative_to(root/'docs/review/r7/input/specs'):
  raise AssertionError('acceptance forbids real asset reads')
 return original_verified(path,*args,**kwargs)

print('IMPLEMENTATION_SHA '+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),flush=True)
print('ARGV_AUDIT '+subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip(),flush=True)
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
with ExitStack() as stack:
 for module in (registry,source_runner,screen_runner):stack.enter_context(patch.object(module,'verified',side_effect=procedural_verified))
 for name in ('_lazy_init','init','device_count','get_device_properties','is_available'):
  stack.enter_context(patch('torch.cuda.'+name,return_value=False) if name=='is_available' else patch('torch.cuda.'+name,side_effect=AssertionError('GPU forbidden')))
 stack.enter_context(patch('torch.load',side_effect=load));stack.enter_context(patch('PIL.Image.open',side_effect=image))
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_target_screen'))
after=resource.getrusage(resource.RUSAGE_SELF)
r=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),exit_code=int(not result.wasSuccessful()),wall_seconds=time.monotonic()-start,cpu_user_seconds=after.ru_utime-before.ru_utime,cpu_system_seconds=after.ru_stime-before.ru_stime,peak_RSS=after.ru_maxrss,peak_RSS_unit='bytes' if sys.platform=='darwin' else 'KiB',python=platform.python_version(),torch=torch.__version__,physical_calls=calls,procedural_counts=dict(COUNTS),hash_cost=dict(HASH_COST),real_RGB_mask_decodes=0,real_checkpoint_deserializations=0,GPU_queries_or_initializations=0,real_source_training=False,target_execution_layer_review='NOT_RUN')
print('TARGET_SCREEN_CPU_RESULT '+json.dumps(r),flush=True)
if os.environ.get('TARGET_SCREEN_CPU_RESULT'):Path(os.environ['TARGET_SCREEN_CPU_RESULT']).write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(r['exit_code'])
