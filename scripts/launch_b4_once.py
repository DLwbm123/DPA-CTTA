"""Single detached execution and CPU closeout, neutral argv for every stage."""
import json,os,subprocess,sys,time
from pathlib import Path
from dpa_ctta.b3_runtime import ENTRY,neutral_subprocesses
neutral_subprocesses();os.umask(0o077)
receipt=Path(os.environ['RUN_RECEIPT']);r=json.loads(receipt.read_text());out=Path(r['output_directory']);code=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.pop('CUBLAS_WORKSPACE_CONFIG',None);env.update(CUDA_VISIBLE_DEVICES=r['gpu_uuid'],RUN_FILE=str(code/'scripts/run_b4.py'))
result=dict(execution_commit=r['commit'],started_utc=time.time(),stages={})
try:
    for stage in ('run','recompute'):
        env['RUN_STAGE']=stage
        with (out/(stage+'.log')).open('x') as log:
            rc=subprocess.call([sys.executable,'-c',ENTRY],cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        result['stages'][stage]=rc
        if rc:raise RuntimeError(stage+' failed with '+str(rc))
    result.update(status='B4_RUN_RECOMPUTE_REPORT_COMPLETE',exit_code=0)
except Exception as e:result.update(status='INCOMPLETE',reason=str(e),exit_code=1)
finally:
    result['finished_utc']=time.time()
    with (out/'launcher.completion.json').open('x') as f:json.dump(result,f,indent=2)
sys.exit(result['exit_code'])
