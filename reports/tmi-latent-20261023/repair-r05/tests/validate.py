"""Linux synthetic integration tests, exclusively in neutral local /tmp spools."""
import ctypes, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path

ctypes.CDLL(None).prctl(36,1,0,0,0)  # Reap only our orphaned fixture descendants.
PYTHON = '/usr/bin/python3'
GUARD = '/tmp/e695.py'
PROBE = '/tmp/e696.py'
WORKER = '/tmp/e697.py'
BLOCK = '/tmp/e699.py'
RESULT = Path(os.environ['VALIDATION_RESULT'])


def alive(pid):
    try: return Path('/proc',str(pid),'stat').read_text().split()[2] != 'Z'
    except FileNotFoundError: return False


def reap():
    while True:
        try:
            if os.waitpid(-1,os.WNOHANG)[0] == 0: return
        except ChildProcessError: return


def job(case='normal', probe_case=None, old=False):
    r = Path(tempfile.mkdtemp(prefix='n',dir='/tmp')); (r/'work').mkdir()
    c = {'origin_unix':time.time()-(10 if old else 0), 'hard_seconds':4.,'stop_margin':1.6,
         'kill_reserve':.4,'probe_timeout':.5,'probe_interval':.15,'failure_grace':.25,
         'stdout_cap':8192,'storage_soft':4000000,'storage_hard':5000000,'expected_peak_bytes':500000,
         'worker_argv':[PYTHON,'-I','-S',WORKER], 'probe_argv':[PYTHON,'-I','-S',BLOCK if probe_case else PROBE],
         'work_root':str(r/'work'),'worker_env':{'CASE':case}}
    (r/'job.json').write_text(json.dumps(c))
    env={**os.environ,'CONTROL_ROOT':str(r),'PROBE_CASE':probe_case or 'normal'}
    return r,c,env


def run(r,env):
    started=time.monotonic()
    p = subprocess.run([PYTHON,'-I','-S',GUARD],env=env,cwd='/tmp',capture_output=True,timeout=6)
    assert (r/'EXIT.json').exists(),p.stderr.decode()
    d=json.loads((r/'EXIT.json').read_text())
    for name in ['fixture.pid','child.pid']:
        if (r/name).exists(): assert not alive(int((r/name).read_text())),(name,d)
    reap()
    return p,d,time.monotonic()-started


out = {'synthetic_only':True,'scientific_data_reads':0,'model_calls':0,'tests':{}}
r,c,env=job();p,d,t=run(r,env)
assert p.returncode==0 and d['termination_confirmed'] and d['within_deadline']
out['tests']['normal_exit']=d
r,c,env=job('ignore');p,d,t=run(r,env)
assert d['reason']=='TIME_SOFT_STOP' and d['within_deadline'] and d['termination_confirmed']
assert (r/'child.pid').exists()
out['tests']['term_ignored_descendant_killed']=d
r,c,env=job('save');p,d,t=run(r,env)
assert (r/'SAVED').exists() and d['within_deadline'] and d['termination_confirmed']
out['tests']['saving_grace_window_preserved']=d
r,c,env=job('ignore','block_first');p,d,t=run(r,env)
assert d['reason']=='STORAGE_PROBE_TIMEOUT' and not d['worker_started'] and t<2
out['tests']['blocked_preflight_does_not_launch']=d
r,c,env=job('ignore','block_after');p,d,t=run(r,env)
assert d['reason']=='STORAGE_PROBE_TIMEOUT' and d['termination_confirmed'] and d['within_deadline'] and t<3
out['tests']['blocked_postlaunch_check_fails_closed']=d
r,c,env=job('ignore');c.update(storage_soft=150000,storage_hard=500000,expected_peak_bytes=50000)
(r/'job.json').write_text(json.dumps(c));(r/'work'/'allocation').write_bytes(b'x'*160000)
p,d,t=run(r,env)
assert d['reason']=='STORAGE_SOFT_STOP' and not d['worker_started']
out['tests']['storage_limit_preflight_refused']=d
r,c,env=job('grow');c.update(storage_soft=150000,storage_hard=500000,expected_peak_bytes=50000)
(r/'job.json').write_text(json.dumps(c));p,d,t=run(r,env)
assert d['reason']=='STORAGE_SOFT_STOP' and d['worker_started'] and d['within_deadline']
out['tests']['storage_growth_stops_worker']=d
r,c,env=job(old=True);p,d,t=run(r,env)
assert not d['worker_started'] and not (r/'LAUNCH.json').exists()
out['tests']['expired_clock_refused']=d
r,c,env=job('flood');p,d,t=run(r,env)
assert p.returncode==0 and (r/'stdout.log').stat().st_size<=8192 and d['stdout_may_be_truncated']
out['tests']['bounded_stdout_does_not_starve_timer']=d
previous=(r/'EXIT.json').read_bytes();c['origin_unix']=time.time()+100;(r/'job.json').write_text(json.dumps(c))
p=subprocess.run([PYTHON,'-I','-S',GUARD],env=env,capture_output=True,timeout=2)
assert p.returncode==2 and (r/'EXIT.json').read_bytes()==previous
out['tests']['reuse_cannot_reset_clock']={'refused':True,'original_exit_preserved':True}
r,c,env=job('ignore')
p=subprocess.Popen([PYTHON,'-I','-S',GUARD],env=env,cwd='/tmp',stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
limit=time.monotonic()+1.5
while not (r/'LAUNCH.json').exists() and time.monotonic()<limit:time.sleep(.01)
assert (r/'LAUNCH.json').exists(),p.stderr.read().decode()
launch=json.loads((r/'LAUNCH.json').read_text());os.kill(p.pid,signal.SIGKILL);p.wait(timeout=1)
limit=launch['deadline_monotonic']
while time.monotonic()<limit:
    ids=[int((r/n).read_text()) for n in ['fixture.pid','child.pid'] if (r/n).exists()]
    if ids and all(not alive(i) for i in ids): break
    time.sleep(.02)
assert ids and all(not alive(i) for i in ids)
assert not alive(launch['wrapper']['pid'])
reap()
out['tests']['guardian_sigkill_native_timeout_survives']={'termination_confirmed':True,'before_original_deadline':time.monotonic()<=limit}
r,c,env=job('ignore','block_first')
p=subprocess.Popen([PYTHON,'-I','-S',GUARD],env=env,cwd='/tmp',stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
limit=time.monotonic()+1.5
while not (r/'work'/'probe.pid').exists() and time.monotonic()<limit:time.sleep(.01)
assert (r/'work'/'probe.pid').exists()
probe_pid=int((r/'work'/'probe.pid').read_text())
os.kill(p.pid,signal.SIGKILL);p.wait(timeout=1)
limit=time.monotonic()+1.5
while alive(probe_pid) and time.monotonic()<limit:time.sleep(.02)
assert not alive(probe_pid) and not (r/'LAUNCH.json').exists()
reap()
out['tests']['guardian_sigkill_probe_timeout_survives']={'probe_termination_confirmed':True,'worker_started':False}
r,c,env=job('legacy');env['CASE']='legacy'
start=time.monotonic()
p=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=.3s','.5s',PYTHON,'-I','-S',WORKER],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
code=p.wait(timeout=2);reap();assert code==-9,code
out['tests']['legacy_timeout_can_return_minus9']={'exit_code':code,'elapsed_seconds':time.monotonic()-start,'historical_cause_proven':False}
out['passed']=len(out['tests'])==13
RESULT.write_text(json.dumps(out,indent=2))
print(json.dumps({'passed':out['passed'],'tests':list(out['tests'])}),flush=True)
