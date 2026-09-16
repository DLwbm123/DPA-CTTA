"""GT and scalar metrics enter only after transaction and risk-buffer commit."""
import time
from ..r1.assets import target
from ..p1_analysis import evaluate
from ..p1_run import sync


def score(payload,mask):
    try:
        return {name:evaluate(payload[name+'_logits'].sigmoid(),mask,'fundus') for name in ('pre','trial','emit')}|{'q':evaluate(payload['q'],mask,'fundus')}
    finally:payload.clear()


def current(host,entry):
    started=time.monotonic();pixels,rgb_io=target(entry,'image');sync()
    logits,trace=host.step(pixels);sync()
    if logits.requires_grad or host.phase!='EVALUATION_RELEASE' or host.pending:raise ValueError('uncommitted prediction')
    payload=host.take_evaluation();begin=time.monotonic()
    try:
        mask,mask_io=target(entry,'mask');metrics=score(payload,mask)
        return trace,dict(metrics=metrics,transaction_before_GT=True,evaluator_seconds=time.monotonic()-begin,pipeline_seconds=time.monotonic()-started,asset_io=dict(image=rgb_io,mask=mask_io))
    finally:payload.clear()
