"""Small private atomic evidence files; the current result pointer is authoritative."""
import json
import os
import tempfile
import uuid
from pathlib import Path


def write(path,value,replace=False):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix='.write-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        if replace:os.replace(name,path)
        else:os.link(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def invalidate(out,status='RECOMPUTING',reason=None):
    out=Path(out);out.chmod(0o700)
    write(out/'current_result.json',dict(status=status,valid=False,reason=reason),replace=True)
    current=out/'current'
    if current.is_symlink():current.unlink()
    elif current.exists():raise ValueError('unexpected current result path')
    # Preserve pre-fix regular outputs; future versions remain in results/.
    for name in ('public_aggregate.json','R1_EXPERIMENT_REPORT.md'):
        p=out/name
        if p.exists() and not p.is_symlink():p.rename(out/('.previous-'+uuid.uuid4().hex+'-'+name))


def publish(out,result,report):
    out=Path(out);versions=out/'results';versions.mkdir(mode=0o700,exist_ok=True)
    version=versions/uuid.uuid4().hex;version.mkdir(mode=0o700)
    write(version/'public_aggregate.json',result)
    report_path=version/'R1_EXPERIMENT_REPORT.md'
    fd=os.open(report_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(report)
    for name in ('public_aggregate.json','R1_EXPERIMENT_REPORT.md'):
        p=out/name
        if not p.is_symlink():p.symlink_to(Path('current')/name)
    temporary=out/('.current-'+version.name)
    temporary.symlink_to(Path('results')/version.name,target_is_directory=True)
    os.replace(temporary,out/'current')
    write(out/'current_result.json',dict(status=result['status'],valid=True,binding=result['binding'],result_directory=str(version.relative_to(out))),replace=True)
