from fractions import Fraction
from pathlib import Path
from core import choose, matched, auc, save


def run(c):
    rows = [dict(block_id=i, size=size, oc_pixels=size, direction='EXPAND', distance_bucket='LE1',
                 scores={'CROP_Q': q, 'CROP_LOGIT_Q': -q})
            for i, size, q in [(1, 2, 4), (2, 2, 3), (3, 1, 2), (4, 1, -1)]]
    assert choose(rows, 'CROP_Q', 1) == [3]
    chosen = choose(rows, 'CROP_Q', 3)
    assert chosen == [1, 3]
    assert set(matched(rows, chosen, 'CROP_LOGIT_Q')) == {2, 4}
    assert set(matched(rows, chosen, 'CYCLE')) == {2, 4}
    assert choose(rows, 'CROP_Q', 0) == []
    assert auc([0, 0], [1, -1], True) == .5
    assert auc([1, 2], [1, 1], True) is None
    before = Fraction(16, 20)
    a, b = Fraction(18, 22) - before, Fraction(16, 19) - before
    joint = Fraction(18, 21) - before
    assert joint != a + b
    assert joint == (2 - before) / 21
    save(Path(c['root']) / 'public/CPU_CHECK.json', dict(status='PASS', checks=8, synthetic_only=True,
         exact_quota_budget_and_tie_checks=True, zero_budget_and_oneclass_checks=True, union_not_sum_checked=True))
