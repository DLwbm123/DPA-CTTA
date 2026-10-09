"""Rollback learning state while preserving consumed arrivals and random draws."""
from ..r19_model_only.method import clone, equal

BLOCK = 32
HORIZONS = (64, 256)
PIVOTS = tuple(range(64, 1952, 128))


def rollback(before, after, block=BLOCK):
    if before['context'] != after['context'] or after['visits']-before['visits'] != block:
        raise ValueError('rollback origin/arrival mismatch')
    if after['steps']-before['steps'] != block or before['steps'] < 1:
        raise ValueError('rollback must restore a preceding non-source learning state')
    if any(before[k] is not None or after[k] is not None for k in ('adapter','teacher','teacher_adapter')):
        raise ValueError('registered native C state only')
    if not equal(before['modes'], after['modes']) or not equal(before['buffers'], after['buffers']):
        raise ValueError('native C mode/buffer inventory changed')
    result = clone(before)
    for key in ('visits', 'native_rng', 'rng', 'counts'):
        result[key] = clone(after[key])
    return result
