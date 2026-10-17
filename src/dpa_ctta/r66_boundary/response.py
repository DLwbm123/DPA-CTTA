"""Label-free response to fixed one-pixel mask perturbations on the native grid."""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from .geometry import proposal


def mask_features(masks):
    return np.stack((~masks[0], masks[0] & ~masks[1], masks[1])).astype(np.float32)


def response(features, masks):
    if masks.shape != (2, 128, 128) or masks.dtype != bool or (masks[1] & ~masks[0]).any():
        raise ValueError('nested native masks required')
    minus = np.stack([binary_erosion(m, structure=np.ones((3,3)), border_value=0) for m in masks])
    plus = np.stack([binary_dilation(m, structure=np.ones((3,3)), border_value=0) for m in masks])
    candidates = (minus, masks, plus)
    denominator = (minus != plus).sum((1,2))
    result = dict(input_spread=denominator.tolist(), arms={})
    for name in ('IMAGE', 'MASK_ONLY'):
        outputs = []; counts = []
        for candidate in candidates:
            f = features if name == 'IMAGE' else mask_features(candidate)
            output, _, audit = proposal(f, candidate)
            outputs.append(output); counts.append(audit['channels'])
        numerator = (outputs[0] != outputs[2]).sum((1,2))
        result['arms'][name] = dict(output_spread=numerator.tolist(),
            spread_ratio=[float(n/d) if d else None for n,d in zip(numerator,denominator)],
            counts=counts, center_edit_pixels=(outputs[1] != masks).sum((1,2)).tolist())
    return result


def self_check():
    xx = np.indices((128,128))[1]
    p = 1/(1+np.exp(np.clip(4*(xx-63.5),-60,60)))
    mask = xx < 64; masks = np.stack((mask,np.zeros_like(mask)))
    r = response(np.stack((p,1-p)), masks)
    assert r['arms']['IMAGE']['spread_ratio'][0] < r['arms']['MASK_ONLY']['spread_ratio'][0]
    assert response(np.ones((2,128,128)), masks)['arms']['IMAGE']['spread_ratio'][0] == 1
    empty = response(np.ones((2,128,128)), np.zeros_like(masks))
    assert empty['input_spread'] == [0,0] and empty['arms']['IMAGE']['spread_ratio'] == [None,None]
    return dict(status='PASS', informative_image_contracts_more_than_mask_control=True,
                constant_image_is_identity=True, empty_denominator_missing_not_stable=True, straight_fixture=r)
