"""Receipt-bound recovery evidence; the journal still validates complete snapshots."""
from pathlib import Path
import json
from .protocol import digest
from ..r9_current_first.storage import _digest


def snapshots(root):
    root=Path(root);result={}
    for pattern in ('**/checkpoint.*.pt','**/checkpoint.*.json','**/latest.json','score_checkpoint.json'):
        for path in sorted(root.glob(pattern)):
            if path.is_file():result[str(path.relative_to(root))]=_digest(path)
    return result


def evidence(identity,node,attempt,jobroot):
    return dict(schema='R10_FAILURE_EVIDENCE_V1',identity=identity,node=node,attempt=attempt,snapshots=snapshots(jobroot))


def bound_failure(record):
    f=record['failure'];e=f['evidence']
    if (f['class']!='INFRASTRUCTURE' or not isinstance(e,dict) or e.get('schema')!='R10_FAILURE_EVIDENCE_V1'
        or not isinstance(e.get('snapshots'),dict) or e.get('identity')!=record['identity'] or e.get('node')!=record['node'] or e.get('attempt')!=record['attempt']):
        raise ValueError('recovery evidence identity/class')
    return dict(f,evidence=dict(e,receipt_sha256=digest(record),receipt=f"attempts/{record['attempt']}.json"))


def verify_failure(root,jobroot,node,identity,failure):
    e=failure.get('evidence',{});attempt=failure.get('attempt','')
    if not attempt or '/' in attempt or '\\' in attempt:raise ValueError('recovery attempt path')
    path=Path(root)/'attempts'/f'{attempt}.json'
    record=json.loads(path.read_text())
    if record.get('status')!='FAILED' or record.get('node')!=node or record.get('identity')!=identity:
        raise ValueError('recovery receipt identity')
    if failure!=bound_failure(record):raise ValueError('recovery receipt seal')
    if e['snapshots']!=snapshots(jobroot):raise ValueError('recovery snapshot evidence changed')
