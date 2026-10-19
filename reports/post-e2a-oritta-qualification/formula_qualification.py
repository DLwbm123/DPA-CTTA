"""Own data-free checks of OriTTA geometry transfer; no third-party execution."""
import json
import math
import time


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def softmax(x):
    exp = [math.exp(v - max(x)) for v in x]
    return [v / sum(exp) for v in exp]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def angular(u, axes, target):
    # Author distance includes +1e-12 under the square root; tau=1 here.
    distance = [math.sqrt(max(0., dot(u, u) - dot(u, v) ** 2) + 1e-12)
                for v in axes]
    return -math.log(softmax([-d for d in distance])[target])


def run():
    started = time.perf_counter()
    binary_error = 0.
    for margin in [-8., -2., -.3, 0., .7, 3., 9.]:
        p = softmax([-margin / 2, margin / 2])[1]
        binary_error = max(binary_error, abs(p - sigmoid(margin)))
        assert abs(p - sigmoid(margin)) < 1e-14
    # Rank-one projected u lies on either opposite axis. Loss stays log(2).
    axes2 = [[1., 0.], [-1., 0.]]
    losses2 = [angular([t, 0.], axes2, c)
               for t in [-4., -.7, 0., .7, 4.] for c in range(2)]
    assert max(abs(v - math.log(2)) for v in losses2) < 1e-14

    # Independent OD/OC => exact four-state distribution, no pseudo exclusivity.
    joint_error = 0.
    for a in [-7., -1., .3, 4.]:
        for b in [-5., -.2, 1.4, 8.]:
            prob = softmax([0., a, b, a + b])  # 00, 10, 01, 11
            error = max(abs(prob[1] + prob[3] - sigmoid(a)),
                        abs(prob[2] + prob[3] - sigmoid(b)))
            joint_error = max(joint_error, error)
            assert error < 1e-14
    s = math.sqrt(.5)
    axes4 = [[-s, -s], [s, -s], [-s, s], [s, s]]
    # These four centered rows span two dimensions, not min(K-1,D)=3 for D>=3.
    assert all(abs(axes4[0][j] + axes4[3][j]) < 1e-14 for j in range(2))
    assert all(abs(axes4[1][j] + axes4[2][j]) < 1e-14 for j in range(2))
    assert abs(axes4[0][0] * axes4[1][1] - axes4[0][1] * axes4[1][0]) > .5
    u = [2., .4]
    loss00, loss11 = angular(u, axes4, 0), angular(u, axes4, 3)
    assert abs(loss00 - loss11) < 1e-14
    assert abs(angular(u, axes4, 3) - angular([-x for x in u], axes4, 3)) < 1e-14
    # Four-state angular term is NOT constant: it can separate opposite pairs.
    delta = 1e-5
    slope = (angular([2., .4 + delta], axes4, 3) -
             angular([2., .4 - delta], axes4, 3)) / (2 * delta)
    assert abs(slope) > .01
    assert softmax([-math.sqrt(max(0., dot(u, u) - dot(u, v)**2) + 1e-12)
                    for v in axes4])[3] <= .5

    # A possible ray-distance change resolves sign, but is a NEW loss, not a result.
    ray_sq = lambda v: dot(u, u) - max(0., dot(u, v)) ** 2
    assert ray_sq(axes4[0]) > ray_sq(axes4[3])
    # Dropping anatomically invalid 01 changes both marginals in general.
    p = softmax([0., .7, -.4, .3])
    valid = 1 - p[2]
    conditional_od = (p[1] + p[3]) / valid
    conditional_oc = p[3] / valid
    assert abs(conditional_od - sigmoid(.7)) > .01
    assert abs(conditional_oc - sigmoid(-.4)) > .01

    # Binary angular degeneracy does not erase radial or alignment gradients.
    radius = 1. + .333 * 2.  # s=1, d_nn=2s, matches released default.
    radial = max(0., 3. - radius) ** 2
    align = (-1. - 1.) ** 2
    assert radial > 0 and align == 4.
    # Removing common head directions is unsafe for independent sigmoids.
    assert abs(sigmoid(.4 + 1.) - sigmoid(.4)) > .1
    # Default B=1 schedule probes at 200, reset grace 1000; no action in 40.
    probes = [i for i in range(1, 41) if i % 200 == 0]
    eligible = [i for i in range(1, 41) if i % 200 == 0 and i >= 1000]
    assert not probes and not eligible
    assert .5 > .25  # Balanced K2 bank would exceed threshold IF armed.
    assert len([i for i in range(1, 41) if i % 30 == 0]) == 1

    return dict(status="PASS", scope="algebra_only_not_model_or_scientific_effect",
                binary_probability_max_error=binary_error,
                binary_angular_loss_range=[min(losses2), max(losses2)],
                joint_four_state_marginal_max_error=joint_error,
                joint_axes_rank_for_independent_heads=2,
                joint_opposite_class_losses=[loss00, loss11],
                joint_loss_local_derivative=slope,
                joint_angular_constant=False,
                ray_distance_requires_new_loss=True,
                three_state_marginals=[conditional_od, conditional_oc],
                original_marginals=[sigmoid(.7), sigmoid(-.4)],
                radial_example=radial, alignment_example=align,
                default_40_arrival_probes=len(probes),
                default_40_arrival_reset_opportunities=len(eligible),
                default_40_arrival_alignment_events=1,
                model_forwards=0, image_label_weight_reads=0, gpu_jobs=0,
                body_seconds=time.perf_counter()-started)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
