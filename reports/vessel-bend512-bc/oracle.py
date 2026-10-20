"""Exact finite block Dice oracle qualification. Synthetic integers only."""
from fractions import Fraction
from itertools import product
import json
import time


def solve(area, truth, intersection, changes):
    """Changes are disjoint blocks: (prediction area delta, TP delta).

    Caller must ensure each subset is a realizable mask. This optimizer neither
    computes graph components nor constrains ASSD/HD95/vCDR.
    """
    if truth == 0:
        selected = tuple(a < 0 for a, t in changes)
        if area + sum(a for keep, (a, t) in zip(selected, changes) if keep) == 0:
            return Fraction(1), selected, 0
        return Fraction(0), (False,) * len(changes), 0
    n, d = 2 * intersection, area + truth
    seen = set()
    while True:
        value = Fraction(n, d)
        assert value not in seen
        seen.add(value)
        selected = tuple(2 * t * d - n * a > 0 for a, t in changes)
        nn = 2 * intersection + sum(2*t for keep, (a,t) in zip(selected, changes) if keep)
        dd = area + truth + sum(a for keep, (a,t) in zip(selected, changes) if keep)
        assert dd > 0 and nn*d >= n*dd
        if nn*d == n*dd:
            return value, selected, len(seen)
        n, d = nn, dd


def check():
    start = time.perf_counter()
    blocks = ((0, 1), (2, 3), (4,))
    pairs = oracle_subsets = max_iterations = 0
    values = list(product((0, 1), repeat=5))
    for source, candidate, gt in product(values, repeat=3):
        area, truth = sum(source), sum(gt)
        intersection = sum(p*y for p,y in zip(source,gt))
        changes = tuple((sum(candidate[i]-source[i] for i in b),
                         sum((candidate[i]-source[i])*gt[i] for i in b)) for b in blocks)
        score, chosen, it = solve(area, truth, intersection, changes)
        best = Fraction(-1)
        for selection in product((False, True), repeat=3):
            pred = list(source)
            for keep, b in zip(selection, blocks):
                if keep:
                    for i in b: pred[i] = candidate[i]
            den = sum(pred) + truth
            val = Fraction(2*sum(p*y for p,y in zip(pred,gt)), den) if den else Fraction(1)
            best = max(best, val)
            if selection == chosen: assert val == score
            oracle_subsets += 1
        assert score == best
        pairs += 1
        max_iterations = max(max_iterations, it)
    # Joint two-head installation preserves nesting pointwise when both endpoints
    # are nested; independent head actions do not have this guarantee.
    nested = ((0,0), (1,0), (1,1))
    joint_checks = 0
    for source, candidate in product(nested, repeat=2):
        for accept in (False, True):
            od, oc = candidate if accept else source
            assert oc <= od
            joint_checks += 1
    # Pixel C-H can prefer a block that decreases Dice: remove 1 TP and 2 FP.
    assert solve(10,10,1,((-3,-1),))[0] == Fraction(1,10)
    assert Fraction(2*(1-1),10+10-3) < Fraction(2,20)
    return dict(status='PASS',synthetic_source_candidate_truth_triples=pairs,
                exhaustive_subset_comparisons=oracle_subsets,max_iterations_observed=max_iterations,
                joint_nesting_point_checks=joint_checks,exact_rational=True,
                real_images_labels_weights_reads=0,model_forwards=0,gpu_seconds=0,
                cpu_body_seconds=time.perf_counter()-start)


if __name__ == '__main__':
    print(json.dumps(check(),indent=2))
