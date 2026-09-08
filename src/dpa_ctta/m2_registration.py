"""M2 overlay of immutable M1 registration and saved history; no asset selection."""
import json
from pathlib import Path
import numpy as np
import torch

from .m2_episodes import make_m2_episodes,validate_coverage,coverage_summary
from .source_pilot_release import digest,file_identity,check_registered_files
from .host_diagnostic_run import private_json


def inspect_history(path,task):
    # These are our own M1 torch.save files containing native NumPy memory entries.
    states=torch.load(path,map_location='cpu',weights_only=False)
    shape=(1,3,5,5) if task=='fundus' else (1,3,3,3)
    if not isinstance(states,list) or len(states)!=32:raise ValueError('missing original 32-state history')
    for i,state in enumerate(states):
        if set(state)!={'prompt','adam','memory','counters'}:raise ValueError('history fields')
        p=state['prompt']['data_prompt']
        if p.shape!=shape or p.dtype!=torch.float32 or not torch.isfinite(p).all():raise ValueError('history prompt format')
        if not state['counters'] or {c[0] for c in state['counters'].values()}!={i}:raise ValueError('history index/counter mismatch')
        adam=state['adam'];g=adam['param_groups']
        if len(g)!=1 or g[0]['lr']!=(.05 if task=='fundus' else .01) or g[0]['betas']!=(.9,.99) or g[0]['eps']!=1e-8:raise ValueError('history Adam options')
        if i==0 and adam['state']:raise ValueError('initial history optimizer nonempty')
        if i>0:
            if len(adam['state'])!=1:raise ValueError('history optimizer ownership')
            opt=next(iter(adam['state'].values()))
            if int(opt['step'])!=i:raise ValueError('history Adam index mismatch')
            for k in ('exp_avg','exp_avg_sq'):
                if opt[k].shape!=shape or not torch.isfinite(opt[k]).all():raise ValueError('history moments')
        if not isinstance(state['memory'],dict) or len(state['memory'])>i:raise ValueError('history memory count')
        for key,value in state['memory'].items():
            if not isinstance(key,bytes) or len(key)!=p.numel()*4 or np.asarray(value).size!=p.numel() or not np.isfinite(value).all():raise ValueError('history memory layout')
    return states


def anchor(path):
    path=Path(path);before=path.stat();sha=digest(path);after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('evidence changed while reading')
    return dict(path=str(path),sha256=sha,bytes=after.st_size,mtime_ns=after.st_mtime_ns)


def register(old,out):
    old=Path(old).resolve();out=Path(out).resolve()
    if old==out:raise ValueError('M1 directory must not be overwritten')
    receipt=json.loads((old/'receipt.run.json').read_text())
    if receipt['commit']!='d92ed88e603eb68aea97a57493c96a084e98d91c':raise ValueError('wrong M1 execution')
    if json.loads((old/'verification.json').read_text())['status']!='M1_METHOD_VALIDATION_COMPLETE':raise ValueError('M1 completion missing')
    if digest(old/'registration.json')!=receipt['registration_sha256']:raise ValueError('M1 registration drift')
    original=json.loads((old/'registration.json').read_text());checks=[];changed=[]
    for item in original['identities']:
        st=Path(item['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):
            checks.append(file_identity(item['path'],item['sha256']));changed.append(item['path'])
        else:checks.append(item)
    overlay=dict(old_directory=str(old),old_registration_sha256=receipt['registration_sha256'],old_receipt_sha256=digest(old/'receipt.run.json'),
        asset_checks=checks,changed_metadata_rehashed=changed,tasks={},old_log_identities=[],history_rebuilt_online_steps=0,
        history_identity_scope='M1 did not persist a history-file digest. Current file digest and full small-state structure are anchored at M2 registration; no retrospective byte-equivalence claim.')
    frozen=json.loads((old/'final_artifacts.frozen.json').read_text())
    for task,r in original['tasks'].items():
        history=old/f'history_{task}.pt'
        if not history.exists():raise FileNotFoundError('M1 history missing; authorized exact-history recovery must run before registration')
        inspect_history(history,task)
        rows=make_m2_episodes(r['episodes'],task)
        episode_path=out/f'episodes_{task}.json';private_json(episode_path,rows)
        entry=dict(episodes_path=str(episode_path),episodes_sha256=digest(episode_path),
            history=anchor(history),history_reused=True,
            coverage={'old':coverage_summary(r['episodes']),'new':coverage_summary(rows)},
            final_M1_proxy_checks={})
        for method in ('D','O'):
            p=old/f'{task}_{method}_600.pt';done=json.loads((old/f'train_{task}_{method}.completion.json').read_text())
            if p.stat().st_size!=done['artifact_bytes'] or done['final_sha256']!=frozen[task][method]:raise ValueError('M1 final artifact identity record mismatch')
            entry['final_M1_proxy_checks'][method]=dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,recorded_sha256=frozen[task][method],verification='saved completion/frozen digest agreement and current size only; not loaded as M2 initialization')
        overlay['tasks'][task]=entry
        for stage in ('source','target'):
            for arm in ('N','A','R','D','O'):
                p=old/f'{stage}_{task}_{arm}.jsonl'
                if p.exists():overlay['old_log_identities'].append(anchor(p))
    private_json(out/'registration.json',overlay)
    private_json(out/'coverage.public.json',{t:r['coverage'] for t,r in overlay['tasks'].items()})
    return overlay


def load_registered(out):
    overlay=json.loads((out/'registration.json').read_text());old=Path(overlay['old_directory'])
    if digest(old/'registration.json')!=overlay['old_registration_sha256'] or digest(old/'receipt.run.json')!=overlay['old_receipt_sha256']:raise ValueError('M1 registered evidence changed')
    check_registered_files({'identities':overlay['asset_checks']+overlay['old_log_identities']+[r['history'] for r in overlay['tasks'].values()]})
    original=json.loads((old/'registration.json').read_text())
    for task,r in original['tasks'].items():
        entry=overlay['tasks'][task]
        if digest(entry['episodes_path'])!=entry['episodes_sha256']:raise ValueError('frozen episode list changed')
        rows=json.loads(Path(entry['episodes_path']).read_text());validate_coverage(r['episodes'],rows)
        r['episodes']=rows  # In-memory copy only; M1 registration is never written.
    return overlay,original
