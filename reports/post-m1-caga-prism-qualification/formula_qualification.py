"""Data-free necessary-condition checks, not a TTA or clinical evaluation."""
import json
import math
import time


def qualify():
    started = time.perf_counter()
    arrivals, window, trend_windows = 40, 50, 10
    decisions = [t for t in range(1, arrivals + 1) if t % window == 0]
    assert not decisions  # Algorithm 1 initializes restore=False, changes it only here.
    checks = []
    for n in (1, 8, 40, 512):
        for total in (.001, .1, 1., 100.):
            for within in (.001, .1, 1.):
                # r=1: the sole eigenvalue equals its spectral mean.
                f = math.sqrt(total / total)
                a = f  # No strictly below-mean eigenvalue.
                c = min(math.sqrt(within / within), 1.)
                g = math.sqrt(c) * a * math.sqrt(c)
                assert g == 1.
                checks.append(g)
    z, shift = (-1., -2.), 3.
    softmax = lambda xs: [math.exp(x-max(xs))/sum(math.exp(y-max(xs)) for y in xs) for x in xs]
    a, b = softmax(z), softmax([x+shift for x in z])
    assert all(abs(x-y) < 1e-15 for x, y in zip(a, b))
    assert [x >= 0 for x in z] != [x+shift >= 0 for x in z]
    h = lambda p: -p*math.log(p)-(1-p)*math.log1p(-p)
    prevalence_ratio = h(.1)/h(.2)
    assert prevalence_ratio < .9
    return {
        'status': 'PASS_NECESSARY_CONDITIONS_ONLY',
        'caga': {'arrivals_per_trajectory': arrivals, 'paper_window': window,
                 'monitoring_decisions': decisions, 'restoration_events': 0,
                 'full_trend_data_lower_bound_arrivals': window*trend_windows,
                 'lower_bound_is_not_first_trigger_guarantee': True},
        'prism': {'rank_one_positive_variance_examples': len(checks),
                  'T_N_G_equal_identity': True,
                  'zero_variance_and_numerical_floor_cases_not_certified': True,
                  'common_shift_preserves_softmax_but_changes_sigmoid_masks': True,
                  'remaining_centering_and_fusion_may_still_change_predictions': True},
        'bernoulli_marginal_entropy': {'perfectly_correct_prevalence_example': [.2, .1],
                                      'entropy_ratio': prevalence_ratio,
                                      'no_actual_images_or_patient_claim': True},
        'new_gpu_runs': 0, 'model_forwards': 0, 'model_backwards': 0,
        'image_reads': 0, 'label_reads': 0, 'weights_loaded': False,
        'scope': 'Paper formula and protocol arithmetic, not empirical utility or full algorithm reproduction',
        'check_body_seconds': time.perf_counter()-started,
    }


if __name__ == '__main__':
    print(json.dumps(qualify(), indent=2))
