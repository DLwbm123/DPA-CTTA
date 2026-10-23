"""Simulate a stuck storage check using sleep, never a real NFS failure."""
import json, os, time
from pathlib import Path
r = Path(os.environ['WORK_ROOT'])
(r/'probe.pid').write_text(str(os.getpid()))
p = r/'count'; n = int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n))
mode = os.environ['PROBE_CASE']
if mode == 'block_first' or (mode == 'block_after' and n > 1): time.sleep(30)
print(json.dumps({'allocated_bytes':0}),flush=True)
