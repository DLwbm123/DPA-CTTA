"""Phase I procedural suite: deny real-asset decoding and GPU initialization."""
import os,sys,time,json,unittest,tempfile,io
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.pop('CHECKPOINT',None)
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
import torch
torch.set_num_threads(2)
modules=['test_r5_rule','test_r5_host','test_r5_analysis','test_r5_audit_fix']
if os.environ.get('R5_MODULES'):modules=os.environ['R5_MODULES'].split(',')
if os.environ.get('R3_REGRESSION')=='1':modules+=['test_r4t_execution','test_r3_reference','test_r3','test_r3_execution','test_r1','test_r1_fixes','test_r2','test_r2_continuation']
from dpa_ctta.b1_host import Host as B1
counts=dict(network_forwards=0,loss_backward_calls=0,adam_calls=0,jacobian_vjp_calls=0)
for method,key in [('_forward','network_forwards'),('_backward','loss_backward_calls'),('_after_adam','adam_calls')]:
    original=getattr(B1,method)
    def counted(*args,_fn=original,_key=key,**kwargs):
        counts[_key]+=1
        return _fn(*args,**kwargs)
    setattr(B1,method,counted)
started=time.monotonic()
with ExitStack() as stack:
    stack.enter_context(patch('torch.cuda._lazy_init',side_effect=AssertionError('Stage I GPU forbidden')))
    stack.enter_context(patch('dpa_ctta.source_io.source_proxy',side_effect=AssertionError('source proxy forbidden')))
    if os.environ.get('R3_REGRESSION')=='1':
        # Existing IO regression creates tiny images/weights itself. Allow only its
        # dedicated fresh temp tree; registered real assets still fail closed.
        from dpa_ctta import source_io
        from dpa_ctta.r1 import assets
        temp=stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.object(tempfile,'tempdir',temp))
        verified=assets.verified_bytes
        def safe_bytes(path,expected):
            if not Path(path).resolve().is_relative_to(Path(temp).resolve()):raise AssertionError('non-procedural asset path')
            return verified(path,expected)
        stack.enter_context(patch.object(assets,'verified_bytes',side_effect=safe_bytes))
        for name in ('read_pixels','read_mask'):
            original=getattr(source_io,name)
            def safe_read(path,*args,_reader=original,**kwargs):
                if not isinstance(path,io.BytesIO) and not Path(path).resolve().is_relative_to(Path(temp).resolve()):raise AssertionError('non-procedural pixels')
                return _reader(path,*args,**kwargs)
            stack.enter_context(patch.object(source_io,name,side_effect=safe_read))
    else:
        for name in ('dpa_ctta.source_io.read_pixels','dpa_ctta.source_io.read_mask','dpa_ctta.r1.assets.checkpoint','dpa_ctta.r1.assets.target'):
            stack.enter_context(patch(name,side_effect=AssertionError('Stage I real assets forbidden')))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames(modules))
status=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
    procedural_B1_physical=counts,physical_counter_scope='all R5 model tests; B1-derived paths only if inherited regressions requested',python=sys.version.split()[0],torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),real_target_RGB_reads=0,real_target_mask_reads=0,registered_checkpoint_reads=0,procedural_IO_fixtures=os.environ.get('R3_REGRESSION')=='1',
    source_data_reads=0,inputs='procedural CPU tensors; random model weights',exit_code=int(not result.wasSuccessful()))
print(json.dumps(status,indent=2))
if os.environ.get('CHECK_OUTPUT'):Path(os.environ['CHECK_OUTPUT']).write_text(json.dumps(status,indent=2)+'\n')
sys.exit(status['exit_code'])
