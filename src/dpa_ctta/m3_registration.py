"""Small M3 overlay referencing completed M2 and original M1; no selection or sampler."""
import json
from pathlib import Path
from .m2_registration import load_registered as load_m2,anchor
from .source_pilot_release import digest,check_registered_files
from .host_diagnostic_run import private_json

M2_EXECUTION='fef00f5bb1ea9c557054215ebe00ca5ee932ab81'


def register(m2,out):
    m2=Path(m2).resolve();out=Path(out).resolve()
    receipt=json.loads((m2/'receipt.run.json').read_text())
    if receipt['commit']!=M2_EXECUTION or json.loads((m2/'verification.json').read_text())['status']!='M2_EPISODE_COVERAGE_COMPARISON_COMPLETE':raise ValueError('M2 completed execution missing')
    if digest(m2/'registration.json')!=receipt['registration_sha256']:raise ValueError('M2 overlay changed')
    old,registration=load_m2(m2)
    overlay=dict(m2_directory=str(m2),m2_registration_sha256=digest(m2/'registration.json'),m2_receipt_sha256=digest(m2/'receipt.run.json'),old_directory=old['old_directory'],tasks=old['tasks'],history_rebuilt_online_steps=0,old_log_identities=[],o2_artifacts={})
    frozen=json.loads((m2/'final_artifacts.frozen.json').read_text())
    for task in registration['tasks']:
        p=m2/f'{task}_O2_600.pt';done=json.loads((m2/f'train_{task}_O2.completion.json').read_text())
        if done['episodes']!=600 or done['final_sha256']!=frozen[task]['O2'] or p.stat().st_size!=done['artifact_bytes']:raise ValueError('M2 O2 artifact mismatch')
        # The required O2T input is first bound to its saved M2 final digest in this task.
        item=anchor(p)
        if item['sha256']!=frozen[task]['O2']:raise ValueError('sealed O2 contents mismatch')
        overlay['o2_artifacts'][task]=item
        for stage in ('source','target'):
            for arm in ('D2','O2'):overlay['old_log_identities'].append(anchor(m2/f'{stage}_{task}_{arm}.jsonl'))
    private_json(out/'registration.json',overlay)
    private_json(out/'coverage.public.json',{t:r['coverage'] for t,r in overlay['tasks'].items()})
    return overlay


def load_registered(out):
    overlay=json.loads((Path(out)/'registration.json').read_text());m2=Path(overlay['m2_directory'])
    if digest(m2/'registration.json')!=overlay['m2_registration_sha256'] or digest(m2/'receipt.run.json')!=overlay['m2_receipt_sha256']:raise ValueError('M2 receipt/overlay drift')
    _,registration=load_m2(m2)
    check_registered_files({'identities':overlay['old_log_identities']+list(overlay['o2_artifacts'].values())})
    return overlay,registration
