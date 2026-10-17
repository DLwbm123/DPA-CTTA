"""Describe frozen R63 choice errors without fitting or selecting a new policy."""
import json
import os
from pathlib import Path
import time
import numpy as np


def summarize(margin, advantage, selected):
    correct = selected == (advantage > 0)
    groups = {
        'beneficial_switch': selected & (advantage > 0),
        'harmful_switch': selected & (advantage < 0),
        'correct_hold': ~selected & (advantage < 0),
        'missed_switch': ~selected & (advantage > 0),
        'target_tie': advantage == 0,
    }
    result = {k: dict(count=int(v.sum()), advantage_sum_pp=float(advantage[v].sum()),
                     advantage_mean_pp=float(advantage[v].mean()) if v.any() else None)
              for k, v in groups.items()}
    delta = np.where(selected, advantage, 0)
    regret = np.maximum(advantage, 0) - delta
    assert sum(v['count'] for v in result.values()) == advantage.size
    assert np.isclose(delta.sum(), result['beneficial_switch']['advantage_sum_pp'] + result['harmful_switch']['advantage_sum_pp'])
    assert np.isclose(regret.sum(), result['missed_switch']['advantage_sum_pp'] - result['harmful_switch']['advantage_sum_pp'])
    result.update(pairs=int(advantage.size), selector_minus_CW_pp=float(delta.mean()),
                  oracle_regret_pp=float(regret.mean()),
                  margin_advantage_pearson=float(np.corrcoef(margin, advantage)[0, 1]) if margin.std() > 0 and advantage.std() > 0 else None,
                  correct_abs_margin=float(np.abs(margin[correct]).mean()) if correct.any() else None,
                  incorrect_abs_margin=float(np.abs(margin[~correct]).mean()) if (~correct).any() else None)
    return result


def main():
    tick = time.time()
    # Synthetic decomposition exercises both mistake types before reading results.
    test = summarize(np.array([1., 2., -1., -2.]), np.array([2., -3., -4., 5.]), np.array([True, True, False, False]))
    assert test['selector_minus_CW_pp'] == -.25 and test['oracle_regret_pp'] == 2
    root = Path(os.environ['DIAG_INPUT'])
    records = [json.loads(s) for s in (root/'private/choices.jsonl').read_text().splitlines()]
    assert [r['pair'] for r in records] == list(range(96))
    values = np.load(root/'scorer/scores.private.npy', allow_pickle=False)
    assert values.shape == (48, 2, 3, 2) and np.isfinite(values).all()
    scores = np.array([r['scores'] for r in records]).reshape(2, 48, 2).transpose(1, 0, 2)
    selected = np.array([r['selected'] for r in records]).reshape(2, 48).T
    assert np.isfinite(scores).all() and np.isin(selected, [0, 1]).all()
    assert np.array_equal(selected, scores.argmax(2))
    assert np.array_equal(values[:, :, 2], np.take_along_axis(values[:, :, :2], selected[:, :, None, None], axis=2)[:, :, 0])
    margin = scores[:, :, 1] - scores[:, :, 0]
    advantage = (values[:, :, 1] - values[:, :, 0]).mean(2)
    domains = np.array([r['domain'] for r in json.loads((root/'private/images.json').read_text())])
    assert domains.shape == (48,) and len(set(domains)) == 4 and all((domains == d).sum() == 12 for d in set(domains))
    result = dict(status='COMPLETE', synthetic_check='PASS', overall=summarize(margin.ravel(), advantage.ravel(), selected.astype(bool).ravel()), cells=[])
    for order in (0, 1):
        for domain in sorted(set(domains)):
            ix = domains == domain
            result['cells'].append(dict(order=order, domain=domain, **summarize(margin[ix, order], advantage[ix, order], selected[ix, order].astype(bool))))
    result.update(worker_wall_seconds=time.time()-tick, gpu_seconds=0, new_image_or_label_reads=0, independent_validation=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
