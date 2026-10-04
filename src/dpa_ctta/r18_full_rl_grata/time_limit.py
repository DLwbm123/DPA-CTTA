"""User-requested wall-time override, independent of frozen scientific config."""
import json,os,signal,time
from pathlib import Path

def deadlines(t0):
    return dict(T0_epoch=t0,compute_deadline_epoch=t0+9600,absolute_deadline_epoch=t0+10800)

def apply(root):
    p=Path(root)/'USER_TIME_LIMIT.json'
    if not p.exists():return
    x=json.loads(p.read_text())
    if os.environ.get('RUN_MODE')=='worker':
        cutoff=x['absolute_deadline_epoch']-60 if os.environ.get('RUN_PHASE')=='score' else x['compute_deadline_epoch']
        os.environ['RUN_DEADLINE']=str(min(float(os.environ.get('RUN_DEADLINE',cutoff)),cutoff))
        # Fail before any data/model access if a later registered job cannot start.
        if time.time()>=cutoff-15:raise SystemExit('USER_THREE_HOUR_LIMIT: no new compute')

def identity(pid):
    p=Path(f'/proc/{pid}')
    try:
        stat=(p/'stat').read_text().split();argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        return dict(pid=pid,start_ticks=stat[21],state=stat[2],argv=argv)
    except FileNotFoundError:return None

def same(record):
    current=identity(record['pid'])
    return current and current['state']!='Z' and current['start_ticks']==record['start_ticks'] and current['argv']==record['argv']

def terminate(record,events,phase):
    if not same(record):return
    os.kill(record['pid'],signal.SIGTERM);events.append(dict(pid=record['pid'],phase=phase,at=time.time(),signal='TERM',reason='USER_THREE_HOUR_LIMIT'))
    until=time.time()+5
    while same(record) and time.time()<until:time.sleep(.2)
    if same(record):os.kill(record['pid'],signal.SIGKILL);events.append(dict(pid=record['pid'],phase=phase,at=time.time(),signal='KILL',reason='USER_THREE_HOUR_LIMIT'))

def enforce(root):
    root=Path(root);x=json.loads((root/'USER_TIME_LIMIT.json').read_text());events=[]
    (root/'time_limit_guard.json').write_text(json.dumps(dict(identity=identity(os.getpid()),state='ARMED',**deadlines(x['T0_epoch']))))
    while time.time()<x['absolute_deadline_epoch']:
        if time.time()>=x['compute_deadline_epoch']:
            for p in (root/'processes').glob('*.json'):
                r=json.loads(p.read_text())
                if r['active'] and r['phase']!='score':terminate(r,events,r['phase'])
        if events:(root/'USER_TIME_LIMIT_EVENTS.json').write_text(json.dumps(events))
        # Do not leave another daemon after the original finite watch has retired.
        if not any(same(r) for r in x['supervision']):break
        time.sleep(2 if time.time()>=x['compute_deadline_epoch'] else min(30,max(1,x['compute_deadline_epoch']-time.time())))
    if time.time()>=x['absolute_deadline_epoch']:
        for p in (root/'processes').glob('*.json'):
            r=json.loads(p.read_text())
            if r['active']:terminate(r,events,r['phase'])
        for r in x['supervision']:terminate(r,events,'supervision')
    (root/'USER_TIME_LIMIT_EVENTS.json').write_text(json.dumps(events))
    (root/'time_limit_guard.json').write_text(json.dumps(dict(identity=identity(os.getpid()),state='RETIRED',ended=time.time(),**deadlines(x['T0_epoch']))))

if __name__=='__main__':
    assert deadlines(100)['absolute_deadline_epoch']==10900
    assert deadlines(100)['compute_deadline_epoch']==9700
