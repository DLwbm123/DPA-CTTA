"""Three independent, fully covered oracle partitions; frozen per-anchor math."""
import hashlib
import io
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from . import oracle_journal as journal
from .journal import _replace
from .oracles import Oracles
from .schedule import SIZES
from .scope import SCREEN

FULL_ORDER=journal.ORDER


def binding(identity):
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


@contextmanager
def partition(shard):
    if not SCREEN or type(shard) is not int or shard not in (0,1,2) or journal.ORDER != FULL_ORDER:
        raise ValueError('R8 oracle shard scope/order')
    journal.ORDER=FULL_ORDER[shard*256:(shard+1)*256]
    try:yield
    finally:journal.ORDER=FULL_ORDER


def run(root,segmenter,data,identity,guard,failure=None):
    root=Path(root)
    with partition(identity['shard']):
        work=journal.OracleJournal(root,segmenter,data,binding(identity))
        if failure is None:work.create()
        else:work.recover_once(failure)
        while len(work.results)<256:work.run_next(guard)
        packet=work._validated(0)
        if packet['next_ordinal']!=256:raise ValueError('R8 incomplete oracle shard')
        receipt=dict(schema='R8_ORACLE_SHARD_COMPLETE_V1',identity=identity,anchors=256,
                     snapshot_sha256=hashlib.sha256((root/'checkpoint.0.pt').read_bytes()).hexdigest(),
                     physical_sha256=hashlib.sha256((root/'physical.jsonl').read_bytes()).hexdigest(),
                     physical_counts=packet['counts'])
        _replace(root/'shard_complete.json',json.dumps(receipt,sort_keys=True).encode())
    return receipt


def merge(root,shards,identity,folds):
    if not SCREEN or len(shards)!=3 or identity.get('amplitude')!=0.3:
        raise ValueError('R8 oracle merge scope')
    root=Path(root)
    if root.exists():raise ValueError('R8 merged oracle output already exists')
    all_results=[]; physical=[]; totals=Counter(); evidence=[]
    data=SimpleNamespace(folds=folds);segmenter=SimpleNamespace(amplitude=0.3)
    for shard,directory in enumerate(shards):
        directory=Path(directory);expected=dict(identity,shard=shard)
        outer=json.loads((directory/'worker_complete.json').read_text())
        raw=(directory/'shard_complete.json').read_bytes();receipt=json.loads(raw)
        if (outer.get('schema')!='R8_ORACLE_SHARD_WORK_COMPLETE_V1' or outer.get('identity')!=expected or
            outer.get('shard_receipt_sha256')!=hashlib.sha256(raw).hexdigest() or
            receipt.get('schema')!='R8_ORACLE_SHARD_COMPLETE_V1' or receipt.get('identity')!=expected or receipt.get('anchors')!=256 or
            receipt.get('snapshot_sha256')!=hashlib.sha256((directory/'checkpoint.0.pt').read_bytes()).hexdigest()):
            raise ValueError('R8 oracle shard producer identity')
        with partition(shard):
            work=journal.OracleJournal(directory,segmenter,data,binding(expected));packet=work._validated(0)
        logs=(directory/'physical.jsonl').read_bytes()
        if (packet['next_ordinal']!=256 or packet['physical_bytes']!=len(logs) or
            receipt['physical_sha256']!=hashlib.sha256(logs).hexdigest() or receipt['physical_counts']!=packet['counts']):
            raise ValueError('R8 oracle shard physical coverage')
        all_results.extend(packet['results']);totals.update(packet['counts'])
        for line in logs.splitlines():
            row=json.loads(line);row['ordinal']+=shard*256
            physical.append(json.dumps(row,sort_keys=True)+'\n')
        evidence.append(dict(root=str(directory),worker_sha256=hashlib.sha256((directory/'worker_complete.json').read_bytes()).hexdigest()))
    bank={}
    for fold in SIZES:
        rows=[r for r,(f,i) in zip(all_results,FULL_ORDER) if f==fold]
        bank[fold]=vars(Oracles(fold,0.3,torch.stack([r[0] for r in rows],1),tuple(r[1] for r in rows),
            tuple(r[2] for r in rows),tuple(tuple(r[3]) for r in rows)).validate(data))
    inner=dict(binding=binding(identity),amplitude=0.3);buffer=io.BytesIO()
    torch.save(dict(schema='R8_ORACLES_V1',identity=inner,folds=bank),buffer)
    logs=''.join(physical).encode();root.mkdir(parents=True,exist_ok=False)
    _replace(root/'physical.jsonl',logs);_replace(root/'oracles.pt',buffer.getvalue())
    receipt=dict(schema='R8_ORACLE_COMPLETE_V1',identity=inner,anchors=768,
      oracles_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),physical_bytes=len(logs),
      physical_sha256=hashlib.sha256(logs).hexdigest(),physical_counts=dict(totals),shard_evidence=evidence)
    raw=json.dumps(receipt,sort_keys=True).encode();_replace(root/'oracle_complete.json',raw)
    _replace(root/'worker_complete.json',json.dumps(dict(schema='R8_ORACLE_WORK_COMPLETE_V1',identity=identity,
        oracle_receipt_sha256=hashlib.sha256(raw).hexdigest(),shard_evidence=evidence),sort_keys=True).encode())
    return receipt
