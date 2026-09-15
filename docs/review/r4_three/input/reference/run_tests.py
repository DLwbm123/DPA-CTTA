"""Run only supplied mathematical reference tests and write actual summary."""
import sys, json, platform, time, unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
start=time.monotonic()
with (HERE/'final_reference_test.log').open('w') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
summary=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
             seconds=time.monotonic()-start,python=platform.python_version(),torch=torch.__version__,
             cuda_initialized=torch.cuda.is_initialized(),source_data_reads=0,target_RGB_reads=0,target_mask_reads=0,
             checkpoint_reads=0,model_forwards=0,
             scope='procedural kernel/graph tensor math and config preservation/budgets; not CTTA host integration',
             exit_code=0 if result.wasSuccessful() else 1)
(HERE/'REFERENCE_CHECK_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(summary['exit_code'])
