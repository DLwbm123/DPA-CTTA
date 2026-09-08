#!/usr/bin/env python3
"""Detached run then CPU reconstruction; a nonzero stage stops, never restarts."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
receipt=json.loads(args.receipt.read_text());out=Path(receipt['output_directory']);code=Path(__file__).resolve().parents[1];os.umask(0o077)
env=os.environ.copy();env.pop('CUBLAS_WORKSPACE_CONFIG',None)
env['CUDA_VISIBLE_DEVICES']=receipt['gpu_uuid']
result=dict(started_utc=time.time(),stages={},execution_commit=receipt['commit'])
try:
 for stage in ['run','recompute']:
  with (out/(stage+'.log')).open('x') as log:
   rc=subprocess.call([sys.executable,str(code/'scripts/run_m4_trajectory_distillation.py'),stage,'--receipt',str(args.receipt.resolve())],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
  result['stages'][stage]=rc
  if rc:raise RuntimeError(stage+' exited '+str(rc))
 result.update(status='M4_RUN_RECOMPUTE_REPORT_COMPLETE',exit_code=0)
except Exception as error:result.update(status='M4_PARTIAL',reason=str(error),exit_code=1)
finally:
 result['finished_utc']=time.time()
 with (out/'launcher.completion.json').open('x') as f:json.dump(result,f,indent=2)
sys.exit(result['exit_code'])
