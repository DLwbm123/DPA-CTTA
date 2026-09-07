# Fixed-mask medical loss contract

Implementation: [medical_losses.py](../src/dpa_ctta/medical_losses.py).
Prepared data boundary: [proxy_loss.py](../src/dpa_ctta/proxy_loss.py).
Status: synthetic mathematics and full random-model integration verified; no real-data result.

## Inputs and provenance

Logits are finite floating `[image, independent Bernoulli class, H, W]` tensors.
Masks must match shape/dtype/device exactly, contain only 0/1, and have no gradient.
They are fixed known source annotations or procedural test masks. Fundus OD and
OC are independent Bernoulli channels, not exclusive softmax classes. There is no
anatomical single-connectivity/area/shape constraint on Polyp.

`FixedProxy` explicitly distinguishes `known_source_annotation` and
`procedural_geometry_fixture`; the online image API cannot accept evaluator labels.
The present tests create all proxy pixels/masks in code. No coreset, DD, patient
mask, or real source provenance pipeline has been prepared. An enum cannot certify
that a caller's provenance assertion is truthful; that remains a required future
source-ingestion contract, not a claim of target-label isolation by naming alone.

## Exact reduction

For each image and each independent channel, separately:

```text
positive_BCE = mean BCEWithLogits over positive known pixels
negative_BCE = mean BCEWithLogits over negative known pixels
balanced_BCE = (positive_BCE + negative_BCE)/2 when both sets exist
             = BCE of the one existing set otherwise
softDice = (2*sum(p*y) + 1e-6)/(sum(p) + sum(y) + 1e-6)
L_region = mean_over_images_and_channels(balanced_BCE + 1 - softDice)
L_boundary = mean_over_defined_image_channel_pairs(mean_pixels(dbar*(p-y)))
L_total = L_region + beta_boundary*L_boundary
```

`medical_loss` returns `MedicalLoss(total, region, boundary, defined_boundary_pairs)`.
The batch is never silently combined into one large object. Empty masks retain
background BCE/Dice gradients; full masks retain foreground BCE/Dice gradients.
Only their undefined signed-distance terms are skipped. With no defined boundary
pairs, the boundary is a differentiable zero. beta_boundary must be finite and
nonnegative; .1 is an engineering starting value, not a tuned medical result.

## Distance sign, geometry and missing distances

Distances are supplied by the caller as fixed finite tensors, matching mask
shape/dtype/device. For each nonempty/nonfull channel, values must be strictly
negative inside and strictly positive outside, divided by `hypot(H,W)` in pixel
units. Absolute normalized values must not exceed one. A reversed or zero sign
on a defined pixel raises an error. This implementation uses pixel-center
distance to the opposite class, so interface pixels have nonzero signed distances.
A future generator using zero-valued boundary pixels must register that different
convention rather than silently pass incompatible maps.

The runtime checks sign, shape, range, and finiteness; it cannot certify the exact
Euclidean distance or normalization divisor supplied by a caller. The geometric
fixture has an analytically exact distance to the opposite half-plane's nearest
pixel center. A real mask-distance generator is NOT_IMPLEMENTED. No new distance
library was installed. Required nonempty/nonfull distances cannot be omitted when
beta_boundary > 0; beta_boundary=0 permits a declared region-only loss.

All masks and distances arrive on the final task grid; the loss does not resize.
No unknown target GT or target pseudo-mask is used to construct a distance map.
No boundary head, style transfer, DD outer loop, inclusion loss or topology prior.

## Checks that ran

[tests/test_medical_losses.py](../tests/test_medical_losses.py) compares against an
explicit per-image/per-channel hand reference, including both-class, empty and
full channels in one batch. It verifies that boundary gradient descent increases
foreground for an inside miss and decreases it for an outside false positive,
rejects sign reversal, checks empty/full region gradient directions, and validates
fixed-mask/provenance inputs.

[tests/test_vptta_host.py](../tests/test_vptta_host.py) uses the same objective through
the actual random ResUNet34/PraNet and shared native prompt. Changing procedural
proxy masks changes nonzero finite prompt gradients. Combined loss is optimized
by one native Adam step without updating source weights or auxiliary counters.
This proves computational connectivity and lifecycle behavior, not DD efficacy
or segmentation accuracy.
