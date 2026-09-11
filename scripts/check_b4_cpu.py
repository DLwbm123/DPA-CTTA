import os,sys,time,unittest,subprocess
from pathlib import Path
import torch
from dpa_ctta.b3_runtime import neutral_subprocesses
from dpa_ctta.host_diagnostic_run import private_json
neutral_subprocesses();torch.set_num_threads(2);start=time.monotonic()
suite=unittest.TestLoader().loadTestsFromName('test_b4');result=unittest.TextTestRunner(verbosity=2).run(suite)
if os.environ.get('CHECK_OUTPUT'):
    private_json(Path(os.environ['CHECK_OUTPUT']),dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),seconds=time.monotonic()-start,python=sys.version.split()[0],torch=torch.__version__,exit_code=int(not result.wasSuccessful())))
sys.exit(int(not result.wasSuccessful()))
