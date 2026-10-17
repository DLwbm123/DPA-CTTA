# R70: fixed short-stream f-CE component test

Registered before new inference or label scoring. Primary FCE replaces only weighted soft BCE in retained CW_LSO. Hypothesis: the reverse cross-entropy gradient avoids attraction to uncertain soft probabilities and improves the retained short-stream segmentation, beyond simply hardening the target. This is an integration test of an existing literature component, not novel feedback or a full TeSLA reproduction.

## Source and overlap

[TeSLA paper](https://arxiv.org/pdf/2303.09870), equations 3–9, and [official segmentation implementation](https://github.com/devavratTomar/TeSLA/blob/69870cc7d6f1caee48a6264ad54135224cb54c69/test_time_trainers/tesla_seg.py) were inspected. The implementation's reverse CE uses student probabilities multiplying log detached pseudo probabilities with epsilon 1e-6. The complete method additionally uses marginal entropy, EMA/refined teacher and adversarial augmentation. We isolate reverse CE; none of those other components is claimed reproduced. Their existence does not establish this component works on Fundus.

R20 already tests EMA teacher blending; R21/R22 cover historical prototypes/neighbors; R23 covers disagreement masking/intervals; R61 removes strong augmentation and R62 substitutes the original-view target. None of the inspected registered implementations reverses the CE arguments in CW_LSO. Native strong augmentation is random, while TeSLA learns entropy-maximizing policies with feature-mean regularization; this remains a distinct untested alternative, not a failed R61 synonym. It is not added simultaneously because that would confound the component test.

## Fixed arms and formula

For each nested OD/OC Bernoulli independently, p=sigmoid(strong logits), q=detached native six-view mean, W=unchanged native reliability:

- NATIVE: mean W*[-q log p -(1-q)log(1-p)], implemented with stable BCE logits.
- HARD: same BCE with target 1[q>=0.5]. This is the hardening comparator, not a fallback selected after results.
- FCE (primary): mean W*[-p log(q+1e-6) -(1-p)log(1-q+1e-6)].

Replace the existing BCE term, retain exactly the native 0.1 LSO term and its soft target. No marginal-entropy term: imposing balanced foreground/background area in these nested channels is a different intervention. Thus this does not test TeSLA's complete information-maximization objective. FCE gradient is p(1-p)log((1-q+eps)/(q+eps)); it can oppose soft BCE for overshooting pixels but shares the HARD sign except q=0.5. It is not a guaranteed correction of wrong pseudo-labels. Gradient/update magnitudes are not matched; changes cannot be attributed solely to direction.

## Cohort and closed-loop matrix

Reuse the exact 192-image R23 SEARCH cohort and both existing ordered manifests (22 Drishti,57 ORIGA,57 REFUGE,56 REFUGE_Valid), with no new selection and no new labels. Seed 20260907. Six independent source-initialized trajectories (3 arms x 2 orders), 192 arrivals each, 1152 post-update hard predictions. Adaptation states persist across the fixed short stream; no replay, multi-pass, horizon scan or per-image restoration in formal execution. Recompute the current CW_LSO control rather than compare the older R23 CW configuration. Source checkpoint, BN-affine trainability, LR 1e-4, W tau .1/boundary 2, native augmentation/RNG and readout remain fixed. No source training or source images.

## Qualification and safeguards

CPU formula checks: exact analytic Bernoulli f-CE gradient, detached pseudo target, finite endpoints/extreme logits, zero gradient for q=.5, weighted replacement algebra and gradients, known soft-BCE sign reversal. Before formal trajectories, one registered first image per order: NATIVE subclass versus retained host exact output/full-state parity, and repeated output/full-state equality for all three arms from restored initial state. These are separate qualification executions, not committed formal visits. Frozen non-BN weights remain unchanged. Check actual peak memory and estimate all 1152 arrivals within finite budget before admission. Private online manifests contain image-only records. Access guards reject labels and source images. Independent CPU scorer runs only after complete successful GPU worker retirement.

## Fixed outcomes and decision

Primary macro hard-Dice delta FCE-NATIVE, domain/channel/order equally weighted. Also FCE-HARD and HARD-NATIVE, all 16 cells per contrast, image-weighted outcomes and per-arm time. Domain-stratified content bootstrap 2000 draws, seed20261016, keeps both orders and channels paired. It is conditional development uncertainty, not independent-patient inference.

Allocate a later full trajectory only if FCE-NATIVE >=+0.2pp, its bootstrap lower bound >0, both orders >0, worst domain/channel/order >=-2pp, image-weighted delta>=0; additionally FCE-HARD >=+0.1pp with lower bound>0 and both orders>0. Conjunction required. HARD cannot become the primary after inspection. No coefficient/epsilon/LR sweep, seeds expansion, retry for low scores, favorable-domain selection or outcome-dependent stopping. Engineering failures preserve receipts/cost and require diagnosis. Passing is short-stream development evidence only; failing does not prove every full TeSLA component fails.

Budget: one GPU worker <=3600s including qualification; separate CPU scorer <=900s; artifact cap2GiB; total controller <=80min. Record preparation CPU cost separately and include it in the cumulative ledger. Do not wait for full execution in chat; one startup check, then the existing half-hour monitor handles completion and source-only anonymous public delivery.
