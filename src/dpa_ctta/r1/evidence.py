"""Small private atomic evidence files; the current result pointer is authoritative."""
import json
import os
import re
import stat
import tempfile
import uuid
from pathlib import Path


def output_bytes(root):
    """Count each file once; atomic writes and NFS silly-renames may disappear."""
    total=0
    for path in Path(root).rglob('*'):
        try:info=path.stat(follow_symlinks=False)
        except FileNotFoundError:
            if path.name.startswith('.write-') or re.fullmatch(r'\.nfs[0-9a-fA-F]+',path.name):continue
            raise
        if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total


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
