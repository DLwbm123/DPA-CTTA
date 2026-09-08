"""M4 references immutable M3/M2/M1 inputs; only score-blind source order is new."""
import json
from pathlib import Path
from .m3_registration import load_registered as load_m3
from .m2_registration import anchor
from .m4_sequences import make_sequences,validate_sequences
from .source_pilot_release import digest,check_registered_files
from .host_diagnostic_run import private_json


def register(m3,out):
    m3=Path(m3).resolve();out=Path(out).resolve()
    receipt=json.loads((m3/'receipt.run.json').read_text())
    if receipt['commit']!='00d1da5b3a333e53b20d869ac9fcbbdcb8b67a8f' or json.loads((m3/'verification.json').read_text())['status']!='M3_CONDITIONED_PROXY_COMPLETE':raise ValueError('completed M3 required')
    if digest(m3/'registration.json')!=receipt['registration_sha256']:raise ValueError('M3 registration changed')
    prior,reg=load_m3(m3);m2=Path(prior['m2_directory'])
    overlay=dict(m3_directory=str(m3),m3_registration_sha256=receipt['registration_sha256'],old_directory=prior['old_directory'],m2_directory=str(m2),tasks={},artifacts={},identities=[])
    frozen=json.loads((m2/'final_artifacts.frozen.json').read_text())
    for task,r in reg['tasks'].items():
        path=out/f'sequence_{task}.json';rows=make_sequences(r['episodes'],task);private_json(path,rows)
        overlay['tasks'][task]=dict(sequence=anchor(path),history=prior['tasks'][task]['history'],visits=600,streams=5,windows=150)
        overlay['artifacts'][task]={}
        for arm in ['D2','O2']:
            item=anchor(m2/f'{task}_{arm}_600.pt')
            if item['sha256']!=frozen[task][arm]:raise ValueError('sealed M2 proxy mismatch')
            overlay['artifacts'][task][arm]=item
        for split in ['source','target']:
            overlay['identities'].append(anchor(m3/f'{split}_{task}_O3.jsonl'))
    private_json(out/'registration.json',overlay)
    return overlay


def load_registered(out):
    overlay=json.loads((Path(out)/'registration.json').read_text());m3=Path(overlay['m3_directory'])
    if digest(m3/'registration.json')!=overlay['m3_registration_sha256']:raise ValueError('M3 registration drift')
    _,reg=load_m3(m3)
    check_registered_files({'identities':overlay['identities']+[v for t in overlay['artifacts'].values() for v in t.values()]})
    for task,r in reg['tasks'].items():
        item=overlay['tasks'][task]['sequence']
        if digest(item['path'])!=item['sha256']:raise ValueError('M4 sequence changed')
        rows=json.loads(Path(item['path']).read_text());validate_sequences(r['episodes'],rows);r['sequence']=rows
    return overlay,reg
