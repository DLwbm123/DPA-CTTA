"""Host-local transaction lock, configurable GPU bindings, aggregate-only hard caps."""
import json
import math
import secrets
import socket
import time
from pathlib import Path
from .protocol import CAPS
from .storage import lease,write_json


def cost(row):
    if set(row)!=set(CAPS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row.values()):raise ValueError('physical ledger cost')
    return row


class Ledger:
    def __init__(self,root,identity,gpus):
        self.root=Path(root);self.identity=identity;self.gpus=set(gpus)
    def create(self):
        if (self.root/'state.json').exists():raise ValueError('ledger exists')
        with lease(self.root,self.identity,blocking=True):
            write_json(self.root/'state.json',dict(schema='R9_LEDGER_V1',identity=self.identity,host=socket.gethostname(),caps=CAPS,gpus=sorted(self.gpus),attempts={},stop=None))
    def _read(self):
        d=json.loads((self.root/'state.json').read_text())
        if d['identity']!=self.identity or d['caps']!=CAPS or d['gpus']!=sorted(self.gpus) or d['host']!=socket.gethostname():raise ValueError('ledger identity')
        if d['stop']:raise RuntimeError('R9 GLOBAL STOP: '+d['stop'])
        return d
    @staticmethod
    def total(d):
        total=dict.fromkeys(CAPS,0)
        for a in d['attempts'].values():
            total['disk_bytes']-=a.get('released_disk_bytes',0)
            for k,v in cost(a['actual'] if a['actual'] is not None else a['reserved']).items():total[k]+=v
        return total
    def _cap(self,d):
        if any(v>CAPS[k] for k,v in self.total(d).items()):
            d['stop']='aggregate resource cap';write_json(self.root/'state.json',d);raise RuntimeError('R9 GLOBAL STOP: aggregate resource cap')
    def reserve(self,name,gpu,budget):
        cost(budget)
        if gpu not in self.gpus and gpu is not None:raise ValueError('unbound GPU')
        with lease(self.root,self.identity,blocking=True):
            d=self._read()
            if name in d['attempts']:raise ValueError('attempt reuse')
            token=secrets.token_hex(16)
            d['attempts'][name]=dict(token=token,physical_gpu=gpu,reserved=budget,actual=None,status='RUNNING',started=time.time())
            self._cap(d);write_json(self.root/'state.json',d);return token
    def observe(self,name,token,observed,settle=False,failed=False,allow_finished=False):
        cost(observed)
        with lease(self.root,self.identity,blocking=True):
            d=self._read();a=d['attempts'][name]
            if a['token']!=token:raise ValueError('attempt owner')
            if a['status']!='RUNNING':
                if allow_finished:return
                raise ValueError('attempt finished')
            observed={k:max(v,a.get('observed',{}).get(k,0)) for k,v in observed.items()}
            # All per-job estimates are soft; actual aggregate usage remains bounded.
            for k,v in observed.items():a['reserved'][k]=max(a['reserved'][k],v)
            a['observed']=observed.copy();self._cap(d)
            if settle:
                a['status']='FAILED' if failed else 'COMPLETE'
                if not failed:a['actual']=observed.copy()
            write_json(self.root/'state.json',d)
    def release_disk(self,name,bytes_,evidence):
        if type(bytes_) is not int or bytes_<0 or not evidence:raise ValueError('disk release evidence')
        with lease(self.root,self.identity,blocking=True):
            d=self._read();a=d['attempts'][name]
            if evidence in a.get('disk_release_evidence',[]):return
            if a['status']!='COMPLETE' or a.get('released_disk_bytes',0)+bytes_>a['actual']['disk_bytes']:raise ValueError('disk release bound')
            a['released_disk_bytes']=a.get('released_disk_bytes',0)+bytes_
            a.setdefault('disk_release_evidence',[]).append(evidence);write_json(self.root/'state.json',d)
    def stop(self,reason):
        with lease(self.root,self.identity,blocking=True):
            d=json.loads((self.root/'state.json').read_text());d['stop']=reason;write_json(self.root/'state.json',d)
