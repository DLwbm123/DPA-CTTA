"""After successful qualification: one frozen formal replay, then sealed CPU join."""
import os,json,subprocess,time,sys
from pathlib import Path
r=Path('/tmp/n97').resolve();c=json.loads((r/'private/config.json').read_text())
q=json.loads((r/'public/QUALIFICATION.json').read_text());assert q['status']=='PASS' and q['admitted']
assert json.loads((r/'public/QUALIFY_WRAPPER.json').read_text())['exit_code']==0
for stage in ('formal','score'):
 if time.time()>c['deadline']-300:raise TimeoutError('controller deadline')
 (r/'public/RUN_STATE.json').write_text(json.dumps(dict(stage=stage,at=time.time(),deadline=c['deadline'])))
 env=dict(os.environ,MODE=stage)
 p=subprocess.run(['/tmp/q85m/bin/python','/tmp/l97.py'],env=env)
 if p.returncode:
  (r/'public/CONTROLLER_EXIT.json').write_text(json.dumps(dict(status='FAILED',stage=stage,exit_code=p.returncode,at=time.time())))
  sys.exit(p.returncode)
(r/'public/CONTROLLER_EXIT.json').write_text(json.dumps(dict(status='COMPLETE',at=time.time())))
