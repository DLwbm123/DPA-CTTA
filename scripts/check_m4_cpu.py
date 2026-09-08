#!/usr/bin/env python3
"""Bounded affected CPU suite; optional private JSON receipt for deployment evidence."""
import argparse,json,os,sys,time,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
torch.set_num_threads(2)
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_m4_trajectory','test_m1','test_m2_episodes'])
start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
if args.output:
 from dpa_ctta.host_diagnostic_run import private_json
 private_json(args.output,dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),seconds=time.monotonic()-start,python=sys.version.split()[0],torch=torch.__version__,exit_code=int(not result.wasSuccessful()),full_historical_regression='REUSED_NOT_RERUN'))
sys.exit(int(not result.wasSuccessful()))
