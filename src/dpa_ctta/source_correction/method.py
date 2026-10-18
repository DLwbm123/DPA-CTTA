"""ECSC-inspired signed logit correction on a fixed reference, not full S3-Adapt."""
import torch
from torch.nn import functional as F
from ..r78_cross_patient.objective import direct_logits


def correction(logits, ref, head, arm):
    if arm == 'D_RETRAIN':
        return direct_logits(logits.detach(), ref, head(ref.inputs))
    inputs = ref.inputs
    if arm == 'R_SHUFFLE':
        inputs = torch.cat((inputs[:, :4].roll((128, 128), (-2, -1)), inputs[:, 4:]), 1)
    field = head(inputs, geometry_only=arm == 'R_GEOM')
    residual = field[:, 0].unsqueeze(0)
    if arm == 'R_BAND':
        residual = residual*ref.support
    if arm == 'R_NEG':
        residual = -residual
    residual = F.interpolate(residual, logits.shape[-2:], mode='bilinear', align_corners=False)
    return logits.detach()+residual


def match_logit_rms(corrected, base, control):
    """Match per-image scalar functional RMS without labels; downscale only."""
    a = corrected-base
    desired = (control-base).double().square().mean().sqrt()
    actual = a.double().square().mean().sqrt()
    if not torch.isfinite(actual+desired):
        raise ValueError('nonfinite residual')
    if actual <= 1e-12 and desired > 1e-12:
        raise ValueError('degenerate learned residual cannot match nonzero control')
    # shortcut: when the control is larger, retain the bounded candidate and report
    # unmatched status rather than violate the fixed one-logit residual budget.
    factor = (desired/actual.clamp_min(1e-12)).clamp_max(1)
    value = base+a*factor.to(a)
    achieved = (value-base).double().square().mean().sqrt()
    return value, dict(rms_match_target=float(desired), rms_match_actual=float(achieved),
                      rms_match_factor=float(factor), rms_match_exact=bool(torch.isclose(achieved, desired, rtol=1e-4, atol=1e-7)))
