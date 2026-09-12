"""Actual Stage-I CPU suite; no GPU initialization or dataset access."""
import os,sys,time,unittest,json,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
import torch
torch.set_num_threads(2)
from unittest.mock import patch
start=time.monotonic()
with patch('torch.cuda._lazy_init',side_effect=AssertionError('Stage I forbids GPU initialization')):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromName('test_r1'))
status=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),seconds=time.monotonic()-start,python=sys.version.split()[0],torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),checkpoint_mode='registered_weights' if os.environ.get('CHECKPOINT') else 'programmatic_random_weights',exit_code=int(not result.wasSuccessful()))
print(json.dumps(status,indent=2))
if os.environ.get('CHECK_OUTPUT'):Path(os.environ['CHECK_OUTPUT']).write_text(json.dumps(status,indent=2)+'\n')
sys.exit(status['exit_code'])
