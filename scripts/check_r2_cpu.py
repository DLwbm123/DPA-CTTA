"""CPU procedural suite with registered checkpoint use disabled unconditionally."""
import os,sys,time,unittest,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.pop('CHECKPOINT',None)
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
import torch
from unittest.mock import patch
torch.set_num_threads(2)
start=time.monotonic()
modules=['test_r2'] if os.environ.get('R2_ONLY')=='1' else ['test_r1','test_r1_fixes','test_r2']
with patch('torch.cuda._lazy_init',side_effect=AssertionError('Stage I forbids GPU initialization')):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames(modules))
status=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),seconds=time.monotonic()-start,python=sys.version.split()[0],torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),checkpoint_mode='programmatic_random_weights',real_target_reads=0,exit_code=int(not result.wasSuccessful()))
print(json.dumps(status,indent=2))
if os.environ.get('CHECK_OUTPUT'):Path(os.environ['CHECK_OUTPUT']).write_text(json.dumps(status,indent=2)+'\n')
sys.exit(status['exit_code'])
