"""Synthetic fault fixture: no datasets, no models, no network filesystem."""
import json, os, signal, subprocess, sys, time
from pathlib import Path

r = Path(os.environ['CONTROL_ROOT'])
case = os.environ['CASE']
if case in {'ignore','grandchild','legacy'}: signal.signal(signal.SIGTERM, signal.SIG_IGN)
(r/('child.pid' if case == 'grandchild' else 'fixture.pid')).write_text(str(os.getpid()))
if case == 'ignore':
    env = os.environ.copy(); env['CASE'] = 'grandchild'
    subprocess.Popen([sys.executable,'-I','-S',__file__],env=env)
if case == 'save':
    started=[None]
    def checkpoint_signal(*_):
        if started[0] is None: started[0]=time.monotonic()
    signal.signal(signal.SIGTERM,checkpoint_signal)
    while started[0] is None or time.monotonic()-started[0]<.65: time.sleep(.01)
    (r/'SAVED').write_text('synthetic checkpoint completed')
    sys.exit(0)
if case == 'flood':
    for _ in range(32): os.write(1,b'x'*65536)
    sys.exit(0)
if case == 'normal': sys.exit(0)
if case == 'grow':
    time.sleep(.1)
    (r/'work'/'allocation').write_bytes(b'x'*160000)
time.sleep(30)
