"""Synthetic existence certificate; no model or research-data dependencies."""
import json
import math
import random
import time


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(a):
    length = math.sqrt(dot(a, a))
    return None if length == 0 else [x / length for x in a]


def q(features, mask):
    vectors = [unit(v) for v in features]
    if any(v is None for v in vectors):
        return None
    if min(sum(mask), len(mask) - sum(mask)) < 3:
        return None
    sums = [[sum(v[k] for v, m in zip(vectors, mask) if m == side)
             for k in range(len(features[0]))] for side in (True, False)]
    a, b = [unit(v) for v in sums]
    return None if a is None or b is None else 1 - dot(a, b)


def delta(features, before, after):
    old, new = q(features, before), q(features, after)
    return None if old is None or new is None else new - old


def run():
    started = time.perf_counter()
    rng = random.Random(20261020)
    before = [True] * 4 + [False] * 4
    after = [True] * 3 + [False] * 5
    witness = None
    for trial in range(1, 2001):
        features = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(8)]
        if any(v[:2] == [0, 0] for v in features):
            continue
        scaled = [[x, y, 8 * n] for x, y, n in features]
        d1, d8 = delta(features, before, after), delta(scaled, before, after)
        if d1 is None or d8 is None or d1 * d8 >= 0 or min(abs(d1), abs(d8)) <= .001:
            continue
        # This fixed W ignores the last coordinate: exact integer logit equality.
        logits = [v[:2] for v in features]
        assert logits == [v[:2] for v in scaled]
        projected = [[x, y, 0] for x, y in logits]
        projection_delta = delta(projected, before, after)
        logit_delta = delta(logits, before, after)
        assert projection_delta is not None and logit_delta is not None
        assert abs(projection_delta - logit_delta) <= 1e-14
        for mask in (before, after):
            assert abs(q(features, mask) - q(features, [not x for x in mask])) <= 1e-14
        assert q([[0, 0, 0]] * 8, before) is None
        assert q(features, [True] * 2 + [False] * 6) is None
        witness = dict(trial=trial, synthetic_features=features,
                       before_mask=before, after_mask=after,
                       null_coordinate_scales=[1, 8],
                       partition_deltas=[d1, d8], logits_identical=True,
                       projection_delta=projection_delta, logit_delta=logit_delta)
        break
    return dict(status='COUNTEREXAMPLE_FOUND' if witness else 'NOT_FOUND_WITHIN_FROZEN_SEARCH',
                witness=witness, trials=trial, maximum_trials=2000, seed=20261020,
                synthetic_only=True, real_data_model_label_reads=0,
                gpu_forwards=0, body_seconds=time.perf_counter() - started,
                scope='Existence only; does not establish frequency or cause of E8 failure')


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
