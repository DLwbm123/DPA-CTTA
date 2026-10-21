"""One launch, fixed qualification -> formal -> CPU score sequence."""
import json,os,subprocess,sys,time
from pathlib import Path
r=Path('/tmp/n105').resolve();c=json.loads((r/'private/config.json').read_text())
with (r/'START_ONCE.json').open('x') as f:json.dump(dict(started=time.time(),T0=c['T0'],deadline=c['deadline']),f)
for stage in ('qualify','formal','score'):
    if time.time()>c['deadline']-300:raise TimeoutError('controller deadline')
    if stage=='formal':
        q=json.loads((r/'public/QUALIFICATION.json').read_text());assert q['status']=='PASS' and q['admitted']
    (r/'public/RUN_STATE.json').write_text(json.dumps(dict(stage=stage,at=time.time(),deadline=c['deadline'])))
    p=subprocess.run(['/tmp/q85m/bin/python','/tmp/l105.py'],env=dict(os.environ,MODE=stage))
    if p.returncode:
        (r/'public/CONTROLLER_EXIT.json').write_text(json.dumps(dict(status='FAILED',stage=stage,exit_code=p.returncode,at=time.time())))
        sys.exit(p.returncode)
(r/'public/CONTROLLER_EXIT.json').write_text(json.dumps(dict(status='COMPLETE',at=time.time())))
