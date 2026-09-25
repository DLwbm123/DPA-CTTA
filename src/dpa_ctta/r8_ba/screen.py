"""Authorized ten-source/forty-target screening scope, sharing the R8 algorithms."""
import json
from pathlib import Path

from .protocol import PROTOCOL_SHA256
from .resources import category
from .source_artifacts import digest, read_completed
from .calibration import select_checkpoint
from .trainer import SAVE_STEPS


def tasks(graph):
    if (graph.get('scope') != 'SCREEN24' or graph.get('protocol_sha256') != PROTOCOL_SHA256 or
            (graph.get('source_training_jobs'),graph.get('target_jobs'),graph.get('target_arrivals')) != (10,40,78040)):
        raise ValueError('R8 screening graph identity')
    work=[]
    def add(key,kind,weights,deps=(),job=None,amplitude=None):
        work.append(dict(id=key,kind=kind,weights=dict(worker_setup=1,**weights),dependencies=list(deps),job=job,amplitude=amplitude))
    for i in range(3):
        add(f'oracle_shard_{i}', 'ORACLE_SHARD', dict(oracle_step=256*256,oracle_query=256*6),amplitude=0.3)
    add('scaler','SCALER',dict(scaler_observation=512*111))
    add('bases_0.3','BASES',dict(basis_vjp=1024),('oracle_0.3',),amplitude=0.3)
    add('capacity_0.3','CAPACITY',dict(capacity_step=2*64*128),('bases_0.3',),amplitude=0.3)
    source=[j for j in graph['jobs'] if not j['arrivals']]
    for job in source:
        route='mlp' if job['stage']=='SOURCE_MLP' else job['arm'].lower()
        weights={f'source_fit_{route}_step':4000,f'source_val_{route}_visit':2*64*32*(1 if route=='mlp' else 2)}
        if route!='mlp':weights[f'source_cal_{route}_step']=2*1024
        add(job['id'],'SOURCE_JOB',weights,('scaler','bases_0.3'),job=job)
    for job in graph['jobs']:
        if job['arrivals']:
            add(job['id']+'_online','TARGET_JOB',{category(job):1951},('artifact_lock',),job=job)
            add(job['id']+'_score','SCORE_JOB',dict(score_visit=1951),(job['id']+'_online',),job=job)
    if len(source)!=10 or len(work)!=96 or len({r['id'] for r in work})!=96:
        raise ValueError('R8 screening queue coverage')
    return work, {
        'oracle_0.3':[f'oracle_shard_{i}' for i in range(3)],
        'grid_selection':[j['id'] for j in source if j['stage']=='SOURCE_GRID'],
        'source_index':[j['id'] for j in source]+['grid_selection'],
        'artifact_lock':['source_index','capacity_0.3']}


def select_grid(graph,configs,roots,code_sha):
    candidates={c['id']:c for c in configs}; points={}; evidence={}
    for candidate in configs:
        points[candidate['id']]={}
        for mode in ('FULL','STATIC'):
            jobs=[j for j in graph['jobs'] if j['stage']=='SOURCE_GRID' and j['config']==candidate['id'] and j['mode']==mode]
            if len(jobs)!=2 or {j['source_seed'] for j in jobs}!={20260924,20260925}:
                raise ValueError('R8 screen paired source seeds')
            rows=[]
            for job in jobs:
                result=read_completed(roots[job['id']],job,candidate,code_sha,graph['spec_sha256'])
                rows.append(result.pop('rows'));evidence[job['id']]=result
            step,scores=select_checkpoint({step:tuple(r[step] for r in rows) for step in SAVE_STEPS})
            points[candidate['id']][mode]=dict(source_step=step,score=scores[step],all_step_scores=scores)
    return dict(schema='R8_SOURCE_ONLY_GRID_SELECTION_V1',scope='SCREEN24',protocol_sha256=PROTOCOL_SHA256,
                code_sha=code_sha,graph_spec_sha256=graph['spec_sha256'],source_jobs=evidence,
                selected_config={c['route']:c['id'] for c in configs},per_config_mode=points,
                selection_kind='predeclared configs; only paired source checkpoint selection')


def lock_targets(index,index_sha,capacity_root):
    root=Path(capacity_root); done=json.loads((root/'worker_complete.json').read_text()); receipt=json.loads((root/'complete.json').read_text())
    identity=done.get('identity',{})
    if (len(index['source_jobs'])!=10 or done.get('schema')!='R8_CAPACITY_WORK_COMPLETE_V1' or
        any(identity.get(k)!=index[k] for k in ('code_sha','protocol_sha256','refs','checkpoint_sha256')) or
        identity.get('amplitude')!=0.3 or done.get('receipt_sha256')!=digest(root/'complete.json') or
        receipt.get('identity')!=identity or receipt.get('completed')!=128 or
        receipt.get('results_sha256')!=digest(root/'results.json') or receipt.get('physical_sha256')!=digest(root/'physical.jsonl')):
        raise ValueError('R8 screening capacity completion identity')
    return dict(index,schema='R8_TARGET_ARTIFACT_LOCK_V1',scope='SCREEN24',source_index_sha256=index_sha,
                gradient_lr_selection={},capacity_worker_sha256={'0.3':digest(root/'worker_complete.json')})
