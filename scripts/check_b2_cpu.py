#!/usr/bin/env python3
import argparse,sys,time,unittest,subprocess
import torch
from dpa_ctta.host_diagnostic_run import private_json
p=argparse.ArgumentParser();p.add_argument('--output');args=p.parse_args();torch.set_num_threads(2);start=time.monotonic()
suite=unittest.TestLoader().loadTestsFromName('test_b2');result=unittest.TextTestRunner(verbosity=2).run(suite)
if args.output:
 from pathlib import Path
 private_json(Path(args.output),dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),seconds=time.monotonic()-start,python=sys.version.split()[0],torch=torch.__version__,exit_code=int(not result.wasSuccessful())))
sys.exit(int(not result.wasSuccessful()))
