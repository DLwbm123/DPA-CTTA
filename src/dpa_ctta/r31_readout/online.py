"""Reuse full-stream I/O and prequential state auditing without label access."""
import torch
from .method import Host, ARMS
from ..r30_state_history.online import stream as full_stream, access_guard


def stream(c, job, guard, permit, count=None, parity=False):
    return full_stream(c, job, guard, permit, count=count, parity=parity, host_class=Host)


def profile(c, job, guard, permit):
    results = {}
    for arm in ARMS:
        j = dict(id=job['id']+'_'+arm, arm=arm, seed=c['seeds'][0], order=0)
        results[arm] = stream(c, j, guard, permit, count=8, parity=True)
    return dict(status='PASS', profiles=results, peak_reserved_bytes=torch.cuda.max_memory_reserved())
