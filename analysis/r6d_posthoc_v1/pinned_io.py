"""Fixed-version input adaptation. Publication aliases are metadata only."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from core import regular


def state(s):
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,
                mode=stat.S_IMODE(s.st_mode),inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink)


class Reader:
    """No-follow directory-descriptor walk; all opens remain explicitly allowlisted."""
    def __init__(self, root, access=None):
        self.root=Path(root).absolute()
        if self.root.is_symlink() or not self.root.is_dir():raise ValueError('source root must be an ordinary directory')
        self.root=self.root.resolve()
        if not hasattr(os,'O_NOFOLLOW') or not hasattr(os,'O_DIRECTORY'):
            raise RuntimeError('required no-follow directory opening unavailable; no fallback')
        self.access=access if access is not None else {}

    def _open(self, name, full, flags, parent=None):
        # CPython includes O_CLOEXEC in the audit flags, including when supplied.
        flags |= getattr(os,'O_CLOEXEC',0)
        self.access['request']=(str(name),str(full),flags)
        try:return os.open(name,flags,dir_fd=parent)
        finally:self.access.pop('request',None)

    @contextmanager
    def opened(self, relative):
        rel=PurePosixPath(relative)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts or str(rel)!=relative:
            raise ValueError('non-canonical source-relative path')
        fds=[];chain=[];df=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
        try:
            full=self.root
            fd=self._open(str(full),full,df);fds.append(fd)
            chain.append((full,state(os.fstat(fd))))
            for part in rel.parts[:-1]:
                full=full/part
                fd=self._open(part,full,df,fd);fds.append(fd)
                chain.append((full,state(os.fstat(fd))))
            path=self.root/relative
            before=path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:
                raise ValueError('source payload must be regular, non-link and single-hardlink')
            flags=os.O_RDONLY|os.O_NOFOLLOW|getattr(os,'O_NOATIME',0)|getattr(os,'O_NONBLOCK',0)
            datafd=self._open(rel.name,path,flags,fd);fds.append(datafd)
            opened=os.fstat(datafd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink!=1 or state(opened)!=state(before):
                raise ValueError('path/opened payload identity mismatch')
            yield datafd,path,state(opened)
            if state(os.fstat(datafd))!=state(opened) or state(path.lstat())!=state(opened):
                raise ValueError('source payload changed during read')
            for directory,identity in chain:
                if not stat.S_ISDIR(directory.lstat().st_mode) or state(directory.lstat())!=identity:
                    raise ValueError('source directory changed during read')
        finally:
            for fd in reversed(fds):os.close(fd)

    def record(self, relative, target=None, capture=False, on_chunk=None):
        h=hashlib.sha256();parts=[];out=None;total=0
        try:
            with self.opened(relative) as (fd,path,before):
                if target is not None:
                    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
                    out=target.open('xb')
                while True:
                    b=os.read(fd,1024*1024)
                    if not b:break
                    h.update(b);total+=len(b)
                    if capture:parts.append(b)
                    if out:out.write(b)
                    if on_chunk:on_chunk()
                if total!=before['bytes']:raise ValueError('source byte count changed')
            if out:out.close();out=None
            if target is not None:
                regular(target)
                if (target.stat().st_dev,target.stat().st_ino)==(before['device'],before['inode']):
                    raise ValueError('snapshot aliases source inode')
            result=dict(before,sha256=h.hexdigest())
            return (result,b''.join(parts)) if capture else result
        finally:
            if out:out.close()


def record_file(path, target=None):
    path=Path(path)
    return Reader(path.parent).record(path.name,target)


def mapping_names(streams):
    names=['current_result.json','receipt.json','R6_SCOPE.json','matrix.processes.json',
           'processes.started.json','public_aggregate.json','packet.private.json']
    names += [f'o{s}a{a}/{name}' for s in streams for a in range(4)
              for name in ['completion.json','unlabeled.jsonl','evaluation.jsonl']]
    names += [f'device{i}/smoke.completion.json' for i in range(3)]
    return names


def source_mapping(addendum, streams):
    return {n:addendum['aggregate_input']['source_relative_path'] if n=='public_aggregate.json' else n
            for n in mapping_names(streams)}


def layout(reader, addendum):
    """Read ordinary authoritative pointer first; never read alias content."""
    pointer_record,raw=reader.record('current_result.json',capture=True)
    pointer=json.loads(raw)
    if pointer.get('valid') is not True or pointer.get('status')!='R6A_COMPLETE_NO_ADVANCE' or pointer.get('binding')!=addendum['historical_binding']:
        raise ValueError('authoritative pointer binding/status')
    version=pointer.get('result_directory')
    if not isinstance(version,str) or re.fullmatch(r'results/[0-9a-f]{32}',version) is None or version!=addendum['pointer']['required_result_directory']:
        raise ValueError('authoritative fixed version mismatch')
    links={}
    for name,rule in addendum['metadata_only_aliases'].items():
        p=reader.root/name;before=p.lstat()
        if not stat.S_ISLNK(before.st_mode):raise ValueError('publication alias must be symlink metadata')
        text=os.readlink(p);after=p.lstat()
        if state(before)!=state(after) or text!=rule['required_readlink']:
            raise ValueError('publication link mismatch or concurrent change')
        links[name]=dict(state(after),type='symlink',readlink=text,readlink_text_sha256=hashlib.sha256(os.fsencode(text)).hexdigest())
    target=version+'/public_aggregate.json'
    if target!=addendum['aggregate_input']['source_relative_path']:raise ValueError('aggregate physical mapping')
    with reader.opened(target) as (_,__,identity):target_identity=identity
    return dict(pointer_record=pointer_record,pointer=pointer,links=links,target_identity=target_identity),raw


def check_layout(reader, addendum, before):
    after,raw=layout(reader,addendum)
    if after!=before:raise ValueError('pointer/alias/target changed since binding')
    return after
