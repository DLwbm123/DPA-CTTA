# R72: support-corrected crop-context pseudo-targets

Registered before target inference. Independent component transfer following closed R70/R71 configurations; no fCE or gamma rescue. Existing R11/R12/R20–R23/R62 and the online source tree were checked: their teacher/view, prototype and weighting mechanisms do not implement crop-image prediction followed by support-corrected inverse fusion. R62's original-view substitution is a different target intervention.

## Hypothesis and fixed mechanism

Changing the field of view/scale can expose current-image predictions different from rotations/flips alone. Test whether these predictions improve CW_LSO beyond merely resampling its existing pseudo-target. This introduces a new evaluation of the current image by the same model, not independent ground truth or a guaranteed correction source.

The inspiration is the crop/inverse stage of [TeSLA's RefinedPseudoLabels](https://github.com/devavratTomar/TeSLA/blob/69870cc7d6f1caee48a6264ad54135224cb54c69/networks/augmentation/segoptaug.py). This is not full TeSLA: no EMA, learned policy, marginal entropy, fCE or random crop sweep. Scale .8 is taken from that implementation; two deterministic normalized-coordinate affine matrices are frozen:

```
A1 = [[ .8, 0, -.1], [0, .8, 0]]
A2 = [[-.8, 0,  .1], [0, .8, 0]]
```

A1 samples a left-shifted crop; A2 a horizontally flipped right-shifted crop. Both use bilinear `affine_grid/grid_sample`, `align_corners=False`. Cropped RGB is passed through the existing per-image normalization and current BN model. Two independent sigmoid outputs are inverse-warped to original coordinates. A map of ones is inverse-warped identically to represent valid coverage. Padding is missing support, never an observed background label; OD/OC are never softmax-normalized together.

Let q be the original six-view mean, b_j the inverse-warped crop probability (including zero padding), and s_j its support. The modified BCE target is `(6*q + b_1 + b_2)/(6+s_1+s_2)`: each original weak view retains one vote. The native strong update augmentation, original six-view W reliability, native LSO target/coefficient .1, BN learning rate and final original-image prediction all remain unchanged. Only the weighted BCE target changes. This isolates target transfer; it does not test simultaneous replacement of the structural target.

Three independently source-initialized arms:

- NATIVE: unmodified CW_LSO.
- RESAMPLE: apply each crop and inverse to q itself, then use the same fusion. This controls coordinate interpolation/smoothing without new crop predictions in the target.
- CROP: use crop-image model predictions as b_j.

RESAMPLE still executes the same two crop-model previews as CROP and discards their probabilities, matching model-forward count. Before each modified arrival, preview uses existing read-only forward and full host snapshot/restore. Exact state equality is checked after preview; the two physical preview forwards are reported separately, although native counters are restored. No label, future image, crop ROI annotation, or historical prediction bank is accessed. View geometry is fixed without using domain identities or outcomes.

## Qualification and design

CPU self-check: affine/inverse matrix identity; known-coordinate ramps recover on full support; two constant Bernoulli fields remain constant after coverage-corrected fusion including outside-crop support; nonempty overlap and uncovered regions; BCE replacement value/gradient equality; no gradient to target; exact zero replacement. This is engineering validation, not segmentation evidence.

Real-model qualification repeats each arm from identical full state on the first image of each order and requires output/full-state equality; NATIVE must match retained Host exactly in both orders. Check frozen parameters, finite targets and budget admission. Formal cost estimate is max measured step duration x1152 x1.5 plus consumed qualification cost.

Reuse exactly R23's exposed SEARCH 192 contents (D22/O57/F57/RV56), two original orders, seed 20260907. Three arms x two independent 192-step loops = 1152 outputs; no previous trajectory substitutes for the new NATIVE control. Each arm starts from source weights with empty moments.

Only after successful GPU-worker retirement, CPU scoring requires all six completion receipts and all 1152 predictions. Primary metric is equal-domain/equal-OD-OC/equal-order mean Dice. Content-paired domain-stratified bootstrap, 2000 draws, seed 20261016. Publish all 48 contrast/domain/order/channel cells, all arm scores, and target-change/flip/support/preview-cost diagnostics. Target flip counts do not establish label corrections.

## Frozen allocation gate and budget

Require CROP-NATIVE >=+0.2 pp, 95% lower bound >0, both orders positive, image-weighted delta >=0 and worst cell >=-2 pp; AND CROP-RESAMPLE >=+0.1 pp, lower bound >0 and both orders positive. RESAMPLE-NATIVE is descriptive. No post-hoc switch of primary, target weight, crop scale/translation, crop count, channel/domain subset, seed or horizon.

GPU process cap 5400 seconds including qualification; CPU scoring cap 900 seconds; output allowance 2 GiB. No new source fitting/model/data. Preserve setup and failed costs, no scientific retry. Close the fixed configuration if its conjunction fails. SEARCH/REVIEW exposure remains: this is development evidence, not independent patient/generalization validation.
