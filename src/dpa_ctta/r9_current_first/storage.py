"""Atomic phase snapshots with one job-wide infrastructure recovery and host leases."""
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import socket
import tempfile
import time
import torch
from ..r8_ba.journal import _replace,_digest
from .protocol import digest


def write_json(path,value):_replace(Path(path),json.dumps(value,sort_keys=True,allow_nan=False).encode())

def write_torch(path,value):
    b=io.BytesIO();torch.save(value,b);_replace(Path(path),b.getvalue())
    return hashlib.sha256(b.getvalue()).hexdigest()


def load_torch(path,expected):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('R9 snapshot seal mismatch')
    return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)


@contextlib.contextmanager
def lease(root,identity,blocking=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lockdir=Path(tempfile.gettempdir())/('l9_'+str(os.getuid()));lockdir.mkdir(mode=0o700,exist_ok=True)
    if lockdir.is_symlink() or lockdir.stat().st_uid!=os.getuid():raise ValueError('local lock ownership')
    lockpath=lockdir/hashlib.sha256(str(root.resolve()).encode()).hexdigest()
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        path=root/'owner.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['host']!=socket.gethostname() or old['identity']!=identity:raise ValueError('owner identity/host changed')
        token=secrets.token_hex(16)
        write_json(path,dict(host=socket.gethostname(),pid=os.getpid(),identity=identity,token=token,acquired=time.time(),active=True))
        try:yield token
        finally:
            current=json.loads(path.read_text())
            if current['token']!=token:raise ValueError('lease token replaced')
            current['active']=False;write_json(path,current)


class PhaseJournal:
    def __init__(self,job_root,phase,identity):
        self.job=Path(job_root);self.root=self.job/phase;self.identity=identity
        self.log=self.root/'physical.jsonl'

    def checkpoint(self,worker):
        raw=self.log.read_bytes();latest=self.root/'latest.json'
        slot=1-json.loads(latest.read_text())['slot'] if latest.exists() else 0
        payload=dict(identity=self.identity,steps=worker.steps,snapshot=worker.snapshot(),log_bytes=len(raw),log_sha256=hashlib.sha256(raw).hexdigest())
        sha=write_torch(self.root/f'checkpoint.{slot}.pt',payload)
        write_json(self.root/'latest.json',dict(slot=slot,sha256=sha,steps=worker.steps))

    def begin(self,worker,failure=None):
        if not self.root.exists():
            if failure is not None:raise ValueError('cannot recover absent phase')
            self.root.mkdir(parents=True);self.log.touch();self.checkpoint(worker);return
        if (self.root/'complete.json').exists():raise ValueError('phase already complete')
        if (not failure or failure.get('class')!='INFRASTRUCTURE' or not failure.get('reason') or
            not failure.get('evidence') or (self.job/'recovery.json').exists()):raise ValueError('one evidenced equivalent recovery required')
        failures=self.job/'failures.jsonl'
        if failures.exists() and json.loads(failures.read_text().splitlines()[-1])['class']!='INFRASTRUCTURE':
            raise ValueError('noninfrastructure failure cannot recover')
        meta=json.loads((self.root/'latest.json').read_text());p=load_torch(self.root/f"checkpoint.{meta['slot']}.pt",meta['sha256'])
        prefix=self.log.read_bytes()[:p['log_bytes']]
        if p['identity']!=self.identity or hashlib.sha256(prefix).hexdigest()!=p['log_sha256']:raise ValueError('recovery prefix/identity')
        worker.restore(p['snapshot'])
        write_json(self.job/'recovery.json',dict(phase=self.root.name,identity=self.identity,steps=p['steps'],failure=failure,retained_physical_bytes=self.log.stat().st_size))

    def append(self,row):
        with self.log.open('ab') as f:
            f.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())

    def complete(self,result):
        receipt=dict(schema='R9_PHASE_COMPLETE_V1',identity=self.identity,physical_sha256=_digest(self.log),result=result)
        write_json(self.root/'complete.json',receipt);return receipt

    def completed(self):
        p=self.root/'complete.json'
        if not p.exists():return None
        r=json.loads(p.read_text())
        if r['identity']!=self.identity or r['physical_sha256']!=_digest(self.log):raise ValueError('phase receipt changed')
        return r
