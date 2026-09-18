"""Metadata only. Stage I entrypoints fail before opening assets or spawning work."""
import hashlib,json
from pathlib import Path
from ..r1.plan import registration_digest
from ..r3.plan import stream,stream_summary
ROOT=Path(__file__).resolve().parents[3]
INPUT=ROOT/'docs/review/r7/input'
BASE='00ccb401942e9f1d3fed48ccb7fb3777c87714b8'

def science():return {p.stem:json.loads(p.read_text()) for p in sorted((INPUT/'specs').glob('*.json'))}
def science_hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((INPUT/'specs').glob('*.json'))}
def real_entry(scope,*args,**kwargs):
    raise PermissionError('Stage I: every real source/target/smoke/AB/auto-run entrypoint is disabled')

def source_binding():
    return dict(status='SOURCE_BINDING_PENDING',source_manifest_sha256=None,source_split_sha256=None,source_checkpoint_sha256=None,source_assets_sha256=None,source_pretraining_exposure='UNKNOWN',subject_identity='UNKNOWN',reason='Stage I does not read source pixels or checkpoint; no frozen source group manifest supplied')

def matrix(scope,groups=()):
    if scope=='TARGET_SCREEN':arms=['C_BASE','C0']+[g+'_'+v for g in 'ABC' for v in ('FULL','STATIC')];orders=(0,1,4)
    elif scope=='TARGET_MECHANISM':arms=['A_ISO_OBS','B_PRED_ONLY','C_CONST_R'];orders=(0,1,4)
    elif scope=='TARGET_EXTENSION':
        if len(groups)>2 or len(set(groups))!=len(groups) or any(g not in 'ABC' or len(g)!=1 for g in groups):raise ValueError('at most two explicitly selected groups')
        arms=['C_BASE','C0']+[g+'_'+v for g in groups for v in ('FULL','STATIC')];orders=(2,3)
    else:raise ValueError('separate exact scope')
    return [dict(job_id=f'{scope}_{a}_{o}',arm=a,order=o,arrivals=1951,scored_contents=1695,network_forwards=1951*(8 if a=='C_BASE' else 1 if a=='C0' else 2),backwards=1951 if a=='C_BASE' else 0,Adam=1951 if a=='C_BASE' else 0,VJP=0,status='NOT_RUN',device=None) for o in orders for a in arms]

def dry_run(reg):
    common=science()['COMMON_PROTOCOL']
    if registration_digest(reg)!=common['registration_digest']:raise ValueError('registered metadata binding')
    summaries=[]
    for o in range(5):
        rows=stream(reg,o)
        summaries.append(dict(order=o,arrivals=len(rows),unique_contents=len({r['group_id'] for r in rows}),scored_contents=sum(r['subset']=='remaining_dev' for r in rows),sequence_sha256=hashlib.sha256(json.dumps([r['group_id'] for r in rows],separators=(',',':')).encode()).hexdigest()))
    summary=stream_summary(reg)
    if summary['stream_digest']!=common['recurrence_digest']:raise ValueError('recurrence binding')
    return dict(status='REGISTERED_METADATA_DRY_RUN',science_sha256=science_hashes(),screen=matrix('TARGET_SCREEN'),mechanism=matrix('TARGET_MECHANISM'),extension_templates={''.join(g):matrix('TARGET_EXTENSION',g) for g in [('A','B'),('A','C'),('B','C')]},extension_template_note='mutually exclusive hypothetical pairs; no group nominated; at most12 jobs, not36',stream=summaries,recurrence=summary,source=source_binding(),source_budget=dict(oracle_F=6144,oracle_backward=3072,oracle_Adam=3072,A_basis_F=32,A_basis_VJP=1024,fit_F=72000,fit_backward_AdamW=6000,cal_F=6144,cal_backward_Adam=1536,validation_F=6*64*12,additional='source scaler observations, oracle query evaluation, C constantR calibration observations separately metered; budget not a complete measured total'),external_review='NOT_RUN',execution_started=False)
