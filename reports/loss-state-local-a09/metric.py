"""Existing r78 metric, reused without model imports."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def metric(pred, gt, base):
    values = []
    for p, y, b in zip(pred, gt, base):
        n = p.sum()+y.sum()
        ps = p ^ binary_erosion(p, border_value=0)
        ys = y ^ binary_erosion(y, border_value=0)
        assd = hd95 = None
        if p.any() and y.any():
            distances = np.concatenate((distance_transform_edt(~ys)[ps], distance_transform_edt(~ps)[ys]))
            assd = float(distances.mean());hd95 = float(np.percentile(distances, 95))
        boundary = distance_transform_edt(~ys) <= 3 if ys.any() else np.zeros_like(y)
        corrected = (b != y) & (p == y)
        spoiled = (b == y) & (p != y)
        regions = {}
        for name, region in [('all', np.ones_like(y)), ('foreground', y), ('boundary3px', boundary)]:
            denominator = int(region.sum())
            regions[name] = dict(denominator=denominator,
                 corrected_fraction=float((corrected & region).sum()/denominator) if denominator else None,
                 spoiled_fraction=float((spoiled & region).sum()/denominator) if denominator else None)
        values.append(dict(dice_percent=float(200*(p & y).sum()/n) if n else 100.,
                           assd_pixel=assd, hd95_pixel=hd95,
                           pred_empty=not bool(p.any()), gt_empty=not bool(y.any()), regions=regions))
    def ratio(mask):
        heights = []
        for m in mask:
            yy = np.nonzero(m)[0]
            heights.append(int(yy.max()-yy.min()+1) if len(yy) else 0)
        return heights[1]/heights[0] if heights[0] else None
    p_ratio, y_ratio = ratio(pred), ratio(gt)
    area_ratio = float(pred[1].sum()/pred[0].sum()) if pred[0].any() else None
    return values, dict(predicted_vcdr=p_ratio, true_vcdr=y_ratio,
                        vcdr_abs_error=abs(p_ratio-y_ratio) if p_ratio is not None and y_ratio is not None else None,
                        predicted_area_ratio=area_ratio)
