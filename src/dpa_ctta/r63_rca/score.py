"""Array-only RCA score; no data loader, target labels or experiment launcher."""
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier


def features(rgb):
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.float32 or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('expected finite 512px RGB float32 in [0,1]')
    image = rgb.reshape(128, 4, 128, 4, 3).mean((1, 3))
    fields = [image]
    for size in (3, 9, 27):
        mean = uniform_filter(image, size=(size, size, 1), mode='reflect')
        square = uniform_filter(image * image, size=(size, size, 1), mode='reflect')
        fields.extend((mean, np.sqrt(np.maximum(square - mean * mean, 0))))
    return np.concatenate(fields, axis=2).reshape(16384, 21)


def proxy_mask(mask):
    if mask.shape != (2, 512, 512) or mask.dtype != np.bool_:
        raise ValueError('expected two boolean 512px channels')
    # Use the pixel nearest each area-bin centre; no label interpolation.
    return mask[:, 2::4, 2::4].reshape(2, 16384)


def joint_score(channel_dice):
    scores = np.asarray(channel_dice)
    if scores.shape != (8, 2) or not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError('eight valid two-channel reference scores required')
    return float(scores.mean(axis=1).max())


def candidate_score(current_features, candidate_mask, reference_features, reference_masks):
    x = np.asarray(current_features)
    refs = np.asarray(reference_features)
    labels = np.asarray(reference_masks)
    if x.shape != (16384, 21) or refs.shape != (8, 16384, 21) or not np.isfinite(x).all() or not np.isfinite(refs).all():
        raise ValueError('fixed finite feature matrix required')
    if labels.shape != (8, 2, 16384) or labels.dtype != np.bool_:
        raise ValueError('eight boolean source-reference masks required')
    y = proxy_mask(candidate_mask)
    dice = np.empty((8, 2), dtype=np.float64)
    for ch in range(2):
        if np.all(y[ch] == y[ch, 0]):
            pred = np.full((8, 16384), y[ch, 0], dtype=bool)
        else:
            forest = RandomForestClassifier(n_estimators=50, max_depth=30, max_features='sqrt',
                                            bootstrap=True, class_weight=None, random_state=20261017, n_jobs=2)
            forest.fit(x, y[ch])
            pred = forest.predict(refs.reshape(-1, 21)).reshape(8, 16384)
        truth = labels[:, ch]
        dice[:, ch] = (2 * (pred & truth).sum(1) + 1e-6) / (pred.sum(1) + truth.sum(1) + 1e-6)
    return joint_score(dice), dice


def choose(scores):
    scores = np.asarray(scores)
    if scores.shape != (2,) or not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError('two candidate scores required')
    return int(scores[1] > scores[0])  # Exact ties retain CW_LSO.


def self_check():
    rgb = np.zeros((512, 512, 3), np.float32)
    rgb[:, 256:] = 1
    mask = np.stack([rgb[..., 0] > .5] * 2)
    x = features(rgb)
    refs = np.repeat(x[None], 8, 0)
    labels = np.repeat(proxy_mask(mask)[None], 8, 0)
    good, detail = candidate_score(x, mask, refs, labels)
    bad, _ = candidate_score(x, ~mask, refs, labels)
    again, repeat = candidate_score(x, mask, refs, labels)
    assert good > .99 and bad < .01 and good == again and np.array_equal(detail, repeat)
    assert choose([good, bad]) == 0 and choose([bad, good]) == 1 and choose([good, good]) == 0
    assert joint_score([[1, 0], [0, 1]] * 4) == .5
    empty = np.zeros_like(mask)
    assert candidate_score(x, empty, refs, np.zeros_like(labels))[0] == 1
    assert np.array_equal(proxy_mask(mask), mask[:, 2::4, 2::4].reshape(2, -1))
    for action in (lambda: features(rgb[:1]), lambda: proxy_mask(mask.astype(float)),
                   lambda: choose([np.nan, 1]), lambda: joint_score(np.ones((2, 2))),
                   lambda: candidate_score(x, mask, refs, labels.astype(float))):
        try:
            action()
        except ValueError:
            continue
        raise AssertionError('invalid input accepted')
    return dict(status='PASS',fixture_only=True,candidate_dependence=True,deterministic=True,
                ties=True,constant_masks=True,joint_reference_max=True,invalid_input_rejection=True,
                good_score=good,bad_score=bad,source_images_read=0,target_images_read=0)
