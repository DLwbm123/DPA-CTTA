"""One finite profile -> admission -> queue -> report chain, no monitor."""
import os,json,subprocess
from pathlib import Path
from dpa_ctta.r10_use_write_rl.identity import require
from dpa_ctta.r10_use_write_rl.protocol import digest
from dpa_ctta.r10_use_write_rl.profile import projection
from dpa_ctta.r9_current_first.storage import write_json
config_path=Path(os.environ['R10_CONFIG']);config=json.loads(config_path.read_text())
root=Path(config['output_root']);private=root/'private';python=os.environ['R10_NEUTRAL_PYTHON']
identity=require(config,profile=True)
def status(stage,**fields):write_json(private/'execution-status.json',dict(stage=stage,identity=identity,**fields))
def child(entry):return subprocess.call([python,os.environ[entry]],env=os.environ.copy())
try:
    status('PROFILE_RUNNING',execution_authorized=False)
    if child('R10_PROFILE_ENTRY')!=0:raise RuntimeError('real profile failed; formal execution stays disabled')
    profile=json.loads((private/'profile.json').read_text());admission=projection(profile)
    if admission!=json.loads((private/'admission.json').read_text()):raise ValueError('admission changed')
    if admission['status']!='PASS':
        status('OVER_CAP',execution_authorized=False,admission=admission)
        raise SystemExit(2)
    config.update(profile=profile,admission=admission,execution_authorized=True)
    auth=dict(config['profile_authorization']);auth.update(scope='FULL_R10_AFTER_REAL_ADMISSION',profile_sha256=digest(profile))
    config['authorization']=auth;require(config)
    path=private/'formal-config.json';write_json(path,config);os.chmod(path,0o600);os.environ['R10_CONFIG']=str(path)
    write_json(private/'formal-authorization.json',auth);status('FORMAL_RUNNING',execution_authorized=True,profile_sha256=digest(profile))
    if child('R10_QUEUE_ENTRY')!=0:raise RuntimeError('formal queue stopped; inspect receipts')
    status('ROUND_TERMINAL',execution_authorized=True)
    if child('R10_REPORT_ENTRY')!=0:raise RuntimeError('round ended but aggregate validation failed')
    status('REPORT_READY',execution_authorized=True,publication='PENDING')
except Exception as exc:
    status('STOPPED',error_type=type(exc).__name__,reason=str(exc));raise
