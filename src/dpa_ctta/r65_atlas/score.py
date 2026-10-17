"""Image-only registration, followed by reverse candidate-mask propagation."""
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.color import rgb2gray
from skimage.registration import optical_flow_tvl1
from ..r63_rca.score import proxy_mask, joint_score


def features(rgb):
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.float32 or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('finite float32 RGB in [0,1] required')
    return rgb2gray(rgb.reshape(128, 4, 128, 4, 3).mean((1, 3))).astype(np.float32)


def warp(mask, flow):
    if mask.shape != (2, 128, 128) or mask.dtype != np.bool_ or flow.shape != (2, 128, 128) or not np.isfinite(flow).all():
        raise ValueError('finite flow and two boolean proxy masks required')
    coordinates = np.indices((128, 128), dtype=np.float32) + flow
    return np.stack([map_coordinates(m.astype(np.uint8), coordinates, order=0, mode='constant', cval=0, prefilter=False).astype(bool) for m in mask])


def candidate_score(current, mask, references, labels):
    if current.shape != (128, 128) or references.shape != (8, 128, 128) or not np.isfinite(current).all() or not np.isfinite(references).all():
        raise ValueError('fixed finite grayscale images required')
    if labels.shape != (8, 2, 16384) or labels.dtype != np.bool_:
        raise ValueError('eight source masks required')
    predicted = proxy_mask(mask).reshape(2, 128, 128)
    scores = []
    for reference, truth in zip(references, labels):
        flow = optical_flow_tvl1(reference, current, attachment=15, tightness=.3,
                                num_warp=5, num_iter=10, tol=.0001, prefilter=False, dtype=np.float32)
        propagated = warp(predicted, flow).reshape(2, -1)
        scores.append((2*(propagated & truth).sum(1)+1e-6)/(propagated.sum(1)+truth.sum(1)+1e-6))
    detail = np.asarray(scores)
    return joint_score(detail), detail


def self_check():
    from scipy.ndimage import gaussian_filter, shift
    rng = np.random.default_rng(20261017)
    reference = gaussian_filter(rng.random((128, 128)).astype(np.float32), 2)
    moving = shift(reference, (2, -3), order=1, mode='reflect')
    flow = optical_flow_tvl1(reference, moving, attachment=15, tightness=.3, num_warp=5, num_iter=10, tol=.0001, prefilter=False, dtype=np.float32)
    mask = np.zeros((2, 128, 128), bool); mask[:, 36:92, 36:92] = True
    moved = shift(mask.astype(np.uint8), (0, 2, -3), order=0).astype(bool)
    aligned = warp(moved, flow)
    dice = 2*(aligned & mask).sum()/(aligned.sum()+mask.sum())
    assert dice > .98 and np.array_equal(warp(mask, np.zeros_like(flow)), mask)
    assert not warp(mask, np.full_like(flow, 1000)).any()
    image = np.repeat(np.repeat(reference, 4, 0), 4, 1)
    rgb = np.repeat(image[:, :, None], 3, 2).astype(np.float32)
    large = np.repeat(np.repeat(mask, 4, 1), 4, 2)
    x = features(rgb); refs = np.repeat(x[None], 8, 0)
    labels = np.repeat(proxy_mask(large)[None], 8, 0)
    good, _ = candidate_score(x, large, refs, labels)
    bad, _ = candidate_score(x, ~large, refs, labels)
    assert good > .99 and bad < .01
    return dict(status='PASS', synthetic_only=True, translation_dice=float(dice),
                identity=True, outside_zero=True, candidate_dependence=True)
