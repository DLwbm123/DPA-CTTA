"""B4/P2 identity reuse, without selecting or rerunning historical controls."""
import json
from pathlib import Path
from .b4_data import old_records as p2_records,map_rows
from .b4_analysis import validate_rows as validate_b4
from .p2_data import expected
from .m1_analysis import read_lines
from .source_pilot_release import digest,check_registered_files
from .host_diagnostic_run import private_json
from .b1_run import metadata

BUDGET=dict(records=9010,forwards=59466,backwards=7208,base_adam=7208,perturb=0,restore=0)
SMOKE=dict(forwards=100,backwards=12,base_adam=12,perturb=0,restore=0)


def stream(reg,order):return expected(reg,'polyp',order)


def old_records(reg,order):
    arms=p2_records(reg,'polyp',order);out=Path(reg['b4_directory'])
    rows=read_lines(out/f'polyp_{order}_C.jsonl');validate_b4(rows,stream(reg,order),'polyp','C',order,arms['N']);arms['C']=rows
    c0=read_lines(out/'polyp_canonical_C0.jsonl');base=p2_records(reg,'polyp',0)['N'] if order else arms['N']
    validate_b4(c0,stream(reg,0),'polyp','C0',None,base);arms['C0']=map_rows(c0,stream(reg,order))
    return arms


def register(b4,out):
    b4=Path(b4);r=json.loads((b4/'receipt.run.json').read_text())
    if r['commit']!='e3c8f5ff80e626ee7e77bcbf75ebd99c71ffd1cb' or digest(b4/'registration.json')!=r['registration_sha256']:raise ValueError('B4 binding')
    if json.loads((b4/'verification.json').read_text())['status']!='B4_FROZEN_C_TRANSFER_COMPLETE':raise ValueError('B4 incomplete')
    reg=json.loads((b4/'registration.json').read_text());reg.update(b4_directory=str(b4),formal_budget=BUDGET)
    for name in ('registration.json','receipt.run.json','verification.json','execution_audit.json','run.environment.json','polyp_canonical_C0.jsonl','polyp_0_C.jsonl','polyp_1_C.jsonl'):reg['identities'].append(metadata(b4/name))
    if len(reg['tasks']['polyp']['target'])!=1802:raise ValueError('Polyp canonical coverage')
    check_registered_files(reg)
    for order in (0,1):old_records(reg,order)
    private_json(out/'registration.json',reg);private_json(out/'registration.completion.json',dict(status='B5_READY',canonical_groups=1802,old_controls_validated=True,formal_budget=BUDGET,exit_code=0))
    return reg
