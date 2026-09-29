"""Finite R10 serial DAG with receipt-first crash reconciliation and score priority."""
import os,subprocess,time,signal
from pathlib import Path
from .protocol import graph,digest,RECOVERY_POLICY,CAPS
from .identity import require
from .ledger import Ledger
from .runtime import read,worker,disk
from .recovery import bound_failure
from ..r9_current_first.storage import write_json,lease


def reconcile(root,state,node,record,identity,ledger):
    name=node['id'];a=state['nodes'][name]['attempt']
    if (record.get('schema')!='R10_ATTEMPT_V1' or record['identity']!=identity or record['node']!=name or record['attempt']!=a):raise ValueError('R10 attempt identity')
    entry=read(ledger.root/'state.json')['attempts'][a]
    if record['status'] not in ('COMPLETE','FAILED') or (record['failure'] is None)!=(record['status']=='COMPLETE'):raise ValueError('receipt status')
    ledger.observe(a,entry['token'],record['actual'],settle=True,failed=record['status']=='FAILED',allow_finished=True)
    if read(ledger.root/'state.json')['attempts'][a]['status']!=record['status']:raise ValueError('ledger receipt mismatch')
    if record['status']=='COMPLETE':state['nodes'][name]=dict(status='COMPLETE',attempt=a)
    else:
        failure=record['failure'];retry=False
        chain=[x for x in read(ledger.root/'state.json')['attempts'] if x.rsplit('.',1)[0]==name]
        if failure['class']=='INFRASTRUCTURE' and len(chain)==1 and failure.get('evidence',{}).get('snapshots'):
            failure=bound_failure(record);retry=ledger.claim_recovery(node.get('job',{}).get('id',name),a,digest(record))
        state['nodes'][name]=dict(status='RETRY' if retry else 'FAILED',attempt=a,failure=failure)
        if failure['class'] not in ('INFRASTRUCTURE','NUMERICAL'):
            write_json(root/'queue.json',state);raise RuntimeError('R10 stopped: '+failure['reason'])
    ledger.sync_disk();write_json(root/'queue.json',state)


def wait_owned(process,ledger,attempt,token,gpu):
    started=time.monotonic()
    while True:
        try:return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if not gpu:continue
            e=read(ledger.root/'state.json')['attempts'][attempt]
            if e['status']!='RUNNING':continue
            c=dict(e.get('observed',dict.fromkeys(CAPS,0)));c['gpu_seconds']=max(c['gpu_seconds'],time.monotonic()-started)
            try:ledger.observe(attempt,token,c,allow_finished=True)
            except RuntimeError:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise


def ready_nodes(root,state,nodes):
    """Failed scientific branches stay terminal while independent branches proceed."""
    completed={k for k,v in state['nodes'].items() if v['status']=='COMPLETE'}
    terminal=completed|{k for k,v in state['nodes'].items() if v['status'] in ('FAILED','BLOCKED')}
    ready=[]
    for n in nodes:
        name=n['id']
        if name in terminal:continue
        needs=set(n['needs'])
        if not needs<=terminal:continue
        reason=None
        if n['kind'] not in ('select','lock') and not needs<=completed:reason='dependency failed or blocked'
        if not reason and n['kind']=='train' and n['job'].get('method','').startswith('SELECTED_'):
            if read(root/'selection.json')['selected'][n['job']['method'].split('_')[1]] is None:reason='family has no complete discovery pair'
        if not reason and n['kind']=='online':
            from .factory import resolve
            from .runtime import receipts
            selection=read(root/'selection.json');arm=n['job']['arm']
            if arm.startswith('SELECTED_') and selection['selected']['SUP' if arm.startswith('SELECTED_SUP') else 'GR'] is None:reason='family unavailable'
            else:
                try:resolve(n['job'],selection,receipts(root))
                except KeyError as exc:reason='source unavailable: '+str(exc)
        if reason:
            state['nodes'][name]=dict(status='BLOCKED',reason=reason);terminal.add(name)
        else:ready.append(n)
    return ready


def run(config,entry,python):
    identity=require(config);root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if any(x in (str(entry)+' '+str(python)).lower() for x in ('wangbomin','ctta','r10','use_write')):raise ValueError('neutral entry/interpreter required')
    ledger=Ledger(root/'ledger',identity,[g['physical_id'] for g in config['gpu_assignments']])
    with lease(root/'queue',identity):
        if not (ledger.root/'state.json').exists():
            ledger.create();prior=config['profile']['prior_cost'];token=ledger.reserve('PROFILE.0',config['gpu_assignments'][0]['physical_id'],prior)
            ledger.observe('PROFILE.0',token,prior,settle=True);ledger.sync_disk()
        state=read(root/'queue.json') if (root/'queue.json').exists() else dict(identity=identity,config_sha256=digest(config),nodes={},recovery_policy=RECOVERY_POLICY)
        if state['identity']!=identity or state['config_sha256']!=digest(config):raise ValueError('queue identity')
        nodes=graph();launch=root/'queue'/'launch.json';write_json(launch,config)
        while True:
            completed={k for k,v in state['nodes'].items() if v['status']=='COMPLETE'}
            eligible=ready_nodes(root,state,nodes)
            if not eligible:break
            n=min(eligible,key=lambda x:(x['kind']!='score',nodes.index(x)));name=n['id'];previous=state['nodes'].get(name)
            if previous and previous['status']=='RUNNING':
                path=root/'attempts'/f"{previous['attempt']}.json"
                if not path.exists():raise RuntimeError('owned worker has no final receipt: retain reservation; inspect before resume')
                reconcile(root,state,n,read(path),identity,ledger);continue
            entries=read(ledger.root/'state.json')['attempts'];chain=[a for a in entries if a.rsplit('.',1)[0]==name]
            if chain:
                claim=read(ledger.root/'state.json')['recovery_claims'].get(n.get('job',{}).get('id',name))
                if len(chain)!=1 or not previous or previous['status']!='RETRY' or not claim or claim['attempt']!=previous['attempt']:raise ValueError('recovery allowance exhausted')
            assignment=config['gpu_assignments'][0] if n['resource']=='gpu' else None
            budget=config['profile']['node_budgets'][name].copy();ledger.sync_disk()
            if chain and n['kind']=='online':
                from .recovery import verify_failure
                job=root/'target'/n['job']['id'];verify_failure(root,job,name,identity,previous['failure'])
                prefix=(job/'predictions.bits').stat().st_size
                if not 0<=prefix<=n['job']['arrivals']*2*512*512*4:raise ValueError('prefix occupancy')
                budget['disk_bytes']-=prefix
            if disk(root)+budget['disk_bytes']+8*1024**2>CAPS['disk_bytes']:ledger.stop('projected output cap');raise RuntimeError('aggregate resource cap: output')
            if assignment:
                from .assets import available_memory
                available_memory(assignment,config['profile']['peak_vram_bytes'][name])
            attempt=name+'.'+str(len(chain));token=ledger.reserve(attempt,assignment['physical_id'] if assignment else None,budget)
            packet=dict(config_path=str(launch),node=n,assignment=assignment,attempt=attempt,token=token,failure=previous.get('failure') if previous else None)
            path=root/'queue'/f'{attempt}.json';write_json(path,packet)
            state['nodes'][name]=dict(status='RUNNING',attempt=attempt);write_json(root/'queue.json',state)
            env=os.environ.copy();env.update(R10_PACKET=str(path),R8_SCOPE='FULL',CUDA_VISIBLE_DEVICES=str(assignment['physical_id']) if assignment else '',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/'queue'/f'{attempt}.log').open('ab') as log:
                p=subprocess.Popen([python,entry],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['nodes'][name]['pid']=p.pid;write_json(root/'queue.json',state);wait_owned(p,ledger,attempt,token,assignment is not None)
            receipt=root/'attempts'/f'{attempt}.json'
            if not receipt.exists():raise RuntimeError('worker exited without receipt; no automatic retry')
            reconcile(root,state,n,read(receipt),identity,ledger)
        state['status']='COMPLETE' if len(completed)==len(nodes) else 'INCOMPLETE'
        state['blocked_nodes']=[n['id'] for n in nodes if n['id'] not in state['nodes']]
        write_json(root/'queue.json',state);return state


def worker_main():
    p=read(os.environ['R10_PACKET']);p['config']=read(p.pop('config_path'));r=worker(**p);return r['status']!='COMPLETE'
