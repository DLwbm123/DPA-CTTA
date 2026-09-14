"""CPU negative control: run the new ordering check against the reviewed old worker."""
import ast,json,os,subprocess,unittest
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(2)
from dpa_ctta.r3 import execution
from test_r3_execution import ExecutionTests
revision='d601496a0827af3e1fe728612a4b9f17a613955d'
reply=subprocess.run(['git','cat-file','--batch'],input=(revision+':src/dpa_ctta/r3/execution.py\n').encode(),capture_output=True,check=True).stdout
header,body=reply.split(b'\n',1);size=int(header.split()[2]);source=body[:size].decode()
worker=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
namespace=execution.__dict__.copy()
exec(compile(ast.Module(body=[worker],type_ignores=[]),'<reviewed-old-worker>','exec'),namespace)
with patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU witness: GPU forbidden')),patch.object(execution,'worker',namespace['worker']):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([ExecutionTests('test_worker_smoke_audits_after_host_before_first_forward')]))
assert not result.wasSuccessful(),'negative control unexpectedly accepted old ordering'
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='EXPECTED_OLD_ORDER_REJECTION',reviewed_sha=revision,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),cuda_initialized=False,real_assets_read=False,scope='old worker only; new test and audit diagnostic helper; procedural collaborators'),indent=2))
