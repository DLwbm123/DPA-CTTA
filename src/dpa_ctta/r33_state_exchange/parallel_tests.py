"""CPU-only handoff check: paused parent, uninterrupted children, real exit codes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from .parallel import process_state, alive, retire_legacy
from ..r16_evidence_correction.run import process_identity


def main():
    with tempfile.TemporaryDirectory(prefix='q_', dir='/tmp') as directory:
        root = Path(directory)
        (root/'w.py').write_text('import os,time\nfrom pathlib import Path\nr=Path(os.environ["CHECK_ROOT"])\nwhile not (r/"finish").exists(): time.sleep(.02)\n')
        (root/'p.py').write_text('''import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
r=Path(os.environ["CHECK_ROOT"])
children=[subprocess.Popen([sys.executable,str(r/'w.py')]) for _ in range(3)]
try:
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        fs=[pool.submit(p.wait) for p in children]
        (r/'ready').write_text(json.dumps([p.pid for p in children]))
        for f in concurrent.futures.as_completed(fs): f.result()
except KeyboardInterrupt:
    (r/'reaped').write_text(json.dumps([p.returncode for p in children]))
''')
        p = subprocess.Popen([sys.executable,str(root/'p.py')],env=dict(os.environ,CHECK_ROOT=str(root)),start_new_session=True)
        identity = process_identity(p)
        try:
            until=time.monotonic()+10
            while not (root/'ready').exists():
                assert time.monotonic()<until, 'fixture startup timeout'
                time.sleep(.02)
            assert alive(identity)
            assert process_state(dict(identity,start_ticks='-1')) is None
            os.kill(p.pid,signal.SIGSTOP)
            until=time.monotonic()+5
            while process_state(identity) not in ('T','t'):
                assert time.monotonic()<until
                time.sleep(.02)
            children = json.loads((root/'ready').read_text())
            (root/'finish').touch()
            until=time.monotonic()+5
            while any(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z' for pid in children):
                assert time.monotonic()<until, 'children did not finish while parent stopped'
                time.sleep(.02)
            assert process_state(identity)=='T'
            retire_legacy(dict(legacy_supervisor=identity,legacy_watchdog=dict(pid=999999999,start_ticks='0')))
            assert p.wait(timeout=5)==0
            assert json.loads((root/'reaped').read_text())==[0,0,0]
            print(json.dumps(dict(status='PASS',children_finished_while_parent_stopped=3,real_exit_codes=[0,0,0],stale_pid_identity_rejected=True)))
        finally:
            if p.poll() is None:
                os.kill(p.pid,signal.SIGCONT);p.kill();p.wait()
            (root/'finish').touch()


if __name__=='__main__':
    main()
