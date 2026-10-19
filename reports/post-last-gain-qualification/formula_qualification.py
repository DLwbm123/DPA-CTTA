"""Own data-free Bernoulli checks of GAIN's evidence path; no model/data access."""
import json
import math
import time


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def path(s, q, lam):
    return sigmoid((1 - lam) * logit(s) + lam * logit(q))


def numerical_optimum(s, q, evaluator):
    if s == q:
        return s
    delta = logit(q) - logit(s)
    def derivative(lam):
        return delta * (evaluator - path(s, q, lam))
    if derivative(0) <= 0:
        return s
    if derivative(1) >= 0:
        return q
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            lo = mid
        else:
            hi = mid
    return path(s, q, (lo + hi) / 2)


def run():
    start = time.perf_counter()
    count, error = 0, 0.0
    for s in [0.05, 0.2, 0.5, 0.8, 0.95]:
        for q in [0.05, 0.3, 0.5, 0.7, 0.95]:
            for evaluator in [0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99]:
                p = numerical_optimum(s, q, evaluator)
                projected = min(max(evaluator, min(s, q)), max(s, q))
                error = max(error, abs(p - projected))
                assert math.isclose(p, projected, abs_tol=1e-12)
                count += 1
    # A supported model-relative correction can still hurt a different true posterior.
    s, q, evaluator, truth = 0.8, 0.4, 0.5, 0.9
    selected = numerical_optimum(s, q, evaluator)
    def risk(target, p):
        return -target * math.log(p) - (1 - target) * math.log(1 - p)
    surrogate_gain = risk(evaluator, s) - risk(evaluator, selected)
    true_gain = risk(truth, s) - risk(truth, selected)
    assert surrogate_gain > 0 and true_gain < 0
    # Identical sigmoid classifier, distinct class-center embeddings after common gauge shift.
    x = 0.75
    source = sigmoid(x)
    proposals = [sigmoid(x - offset - 0.5) for offset in [0, 2]]
    assert proposals[0] > 0.5 > proposals[1]
    # This is a prior, NOT a claim that the final posterior must invert prevalence.
    supports = [10, 90]
    inv = [1 / (3 + n) for n in supports]
    foreground_prior = inv[0] / sum(inv)
    assert foreground_prior > 0.5
    # Repeated pixels change conditional pseudo-support despite unchanged image information.
    support_scale = {str(n): 1 + 1 / (3 + n * 0.9) for n in [1, 1024]}
    assert support_scale['1024'] < support_scale['1']
    return {
        'status': 'PASS', 'scope': 'data_free_transfer_qualification_only',
        'binary_projection_cases': count, 'maximum_projection_error': error,
        'working_vs_true_gain_example': {'selected': selected,
            'working_logrisk_gain': surrogate_gain, 'true_logrisk_gain': true_gain},
        'classifier_gauge_example': {'source_probability_both': source,
            'prototype_proposals': proposals},
        'inverse_support_prior': {'foreground_mass': 10, 'background_mass': 90,
            'foreground_prior': foreground_prior, 'kappa0': 3},
        'repeated_pixel_predictive_h': support_scale,
        'new_gpu_runs': 0, 'model_forwards': 0, 'image_label_weight_reads': 0,
        'body_seconds': time.perf_counter() - start,
        'limitations': 'No Dice or efficacy estimate; no assertion all valid transfers fail.'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
