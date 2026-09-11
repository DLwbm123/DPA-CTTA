"""Reuse completed P2/B3 identities without selection; C0 is canonical once."""
import json
from pathlib import Path
from .b3_orders import build_stream
from .b3_analysis import old_records as b2_records,validate_rows as validate_b3
from .p2_data import expected,ARMS as P2_ARMS
from .p2_analysis import validate_bundle
from .m1_analysis import read_lines
from .source_pilot_release import digest,check_registered_files
from .host_diagnostic_run import private_json
from .b1_run import metadata

BUDGET=dict(records=7357,forwards=32585,backwards=3604,base_adam=3604,perturb=0,restore=0)


def stream(reg,task,order=0):
    return build_stream(reg,order) if task=='fundus' else expected(reg,task,order)


def old_records(reg,task,order):
    if task=='polyp':
        rows={a:read_lines(Path(reg['p2_directory'])/f'polyp_{order}_{a}.jsonl') for a in P2_ARMS}
        validate_bundle(rows,stream(reg,task,order),task,order)
        return {a:rows[a] for a in ('N','A','EA','O2','D4')}
    if order<2:
        rows=b2_records(reg,order);return {a:rows[a] for a in ('N','A','C')}
    old=b2_records(reg,0)['N'];n=map_rows(old,stream(reg,task,order));rows={'N':n}
    for a in ('A','C'):
        rs=rows[a]=read_lines(Path(reg['b3_directory'])/f'fundus_{order}_{a}.jsonl')
        validate_b3(rs,stream(reg,task,order),order,a,n)
    return rows


def map_rows(rows,ordered):
    lookup={r['group_id']:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!={r['group_id'] for r in ordered}:raise ValueError('canonical identity coverage')
    for r in ordered:
        if any(lookup[r['group_id']][k]!=r[k] for k in ('sample_id','domain','subset')):raise ValueError('canonical identity mapping')
    return [lookup[r['group_id']] for r in ordered]


def register(b3,out):
    b3=Path(b3);r=json.loads((b3/'receipt.run.json').read_text())
    if r['commit']!='0fac9b2b1e762ed57b624953edf95058cb1f0380' or digest(b3/'registration.json')!=r['registration_sha256']:raise ValueError('B3 binding')
    if json.loads((b3/'verification.json').read_text())['status']!='B3_FROZEN_ORDER_COMPARISON_COMPLETE':raise ValueError('B3 incomplete')
    reg=json.loads((b3/'registration.json').read_text());p2=Path(reg['p2_directory']);p=json.loads((p2/'registration.json').read_text());receipt=json.loads((p2/'receipt.run.json').read_text())
    if receipt['commit']!='d3ee6901379be293f47abd1687808caaf5b04266' or digest(p2/'registration.json')!=receipt['registration_sha256']:raise ValueError('P2 binding')
    reg['tasks']['polyp']=p['tasks']['polyp'];reg.update(b3_directory=str(b3),formal_budget=BUDGET)
    paths={reg['tasks']['polyp']['checkpoint']['path']}|{row[k] for row in reg['tasks']['polyp']['target'] for k in ('image_path','mask_path')}
    reg['identities'] += [i for i in p['identities'] if i['path'] in paths]
    if len(reg['tasks']['polyp']['target'])!=1802 or paths!={i['path'] for i in p['identities'] if i['path'] in paths}:raise ValueError('Polyp source/pool bindings')
    reg['identities'] += [metadata(p) for p in sorted(p2.glob('polyp_*.jsonl'))]
    reg['identities'] += [metadata(b3/n) for n in ('registration.json','receipt.run.json','verification.json','execution_audit.json','run.environment.json')]
    reg['identities'] += [metadata(p) for p in sorted(b3.glob('fundus_*.jsonl'))]
    check_registered_files(reg)
    for task in ('fundus','polyp'):
        for order in range(4 if task=='fundus' else 2):old_records(reg,task,order)
    private_json(out/'registration.json',reg);private_json(out/'registration.completion.json',dict(status='B4_READY',canonical_groups=dict(fundus=1951,polyp=1802),formal_budget=BUDGET,old_controls_validated=True,exit_code=0))
    return reg
