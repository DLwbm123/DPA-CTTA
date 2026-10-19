"""Data-free checks of the published probe score; no model or labels loaded."""
import time
started = time.perf_counter()
import json
import math
import random


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def disagreement(views):
    if not views:
        raise ValueError("at least one probe required")
    if any(len(p) != len(views[0]) or any(x < 0 for x in p)
           or not math.isclose(sum(p), 1) for p in views):
        raise ValueError("categorical probability vectors required")
    mean = [sum(p[c] for p in views) / len(views) for c in range(len(views[0]))]
    return entropy(mean) - sum(entropy(p) for p in views) / len(views)


def js(p, q):
    return disagreement([p, q])


def utility(before, after, source, beta=1):
    dc = disagreement(before) - disagreement(after)
    da = sum(js(q, s) - js(p, s) for p, q, s in zip(before, after, source)) / len(before)
    return dc - beta * max(0, da), dc, da


rng = random.Random(20261019)
checks = []
for classes in (2, 3, 10):
    for _ in range(32):
        def distribution():
            p = [rng.random() + .01 for _ in range(classes)]
            return [v / sum(p) for v in p]
        before, after, source = ([distribution()] for _ in range(3))
        score, dc, da = utility(before, after, source)
        assert dc == 0 and score <= 0 and not (score > .05)
        checks.append((score, dc))
# In the multi-probe formula D_after and the anchor penalty are nonnegative.
# Thus score <= D_before: a strict .05 margin cannot accept D_before <= .05.
bound_checks = []
for _ in range(32):
    values = [.5 + rng.uniform(-.02, .02) for _ in range(4)]
    before = [[v, 1-v] for v in values]
    values = [rng.uniform(.1, .9) for _ in range(4)]
    after = [[v, 1-v] for v in values]
    score, _, _ = utility(before, after, before)
    d = disagreement(before)
    assert d < .05 and score <= d + 1e-12 and not (score > .05)
    bound_checks.append(d)
assert disagreement([[.2, .8]]) == 0
assert disagreement([[1., 0.], [0., 1.]]) > .69
# A non-degenerate multi-probe consistency gain is possible; M=1 does not
# invalidate the M=4 definition or prove that the reported main result is false.
p = [[.6, .4], [.4, .6], [.6, .4], [.4, .6]]
q = [[.5, .5] for _ in range(4)]
score, dc, da = utility(p, q, p)
assert dc > 0 and score > 0
print(json.dumps(dict(status="PASS", scope="published_formula_algebra_only", real_images=0,
    real_labels=0, model_loads=0, gpu_seconds=0, single_probe_checks=len(checks),
    single_probe_delta_consistency_max=max(x[1] for x in checks),
    single_probe_max_risk_adjusted_utility=max(x[0] for x in checks),
    single_probe_commits_at_margin_005=0, single_probe_hcur=None,
    low_disagreement_bound_checks=len(bound_checks),
    four_probe_nonzero_utility_example=score,
    timed_computation_seconds=time.perf_counter()-started), indent=2))
