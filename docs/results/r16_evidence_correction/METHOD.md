# R16: four evidence/correction directions (Inspired / Adapted)

One finite DEVELOPMENT experiment, based on R15 commit 726a66ec6062963bf3244285d1a68468bbfae57b. R15 remains closed. No RL or automatic successor is authorized.

## Bound task and data

Fundus ResUNet34 at 512x512, source checkpoint digest 88b7d8902d23fb1c15b27668e0bf02599f8298aa45c38ae90279ae5a1d42bdf0. Original RGB bicubic resize, per-image float32 min-max normalization, batch 1, current-statistics eval BN with frozen affine and absent running buffers. Two independent sigmoid channels: OD=(gray<255), OC=(gray==0), so OD contains OC. Hard decision is sigmoid(float32 logits)>=0.5. Existing scorer uses the entire grid, with no FOV exclusion and no public morphological postprocessing. ASSD is undefined if either mask is empty; empty-empty Dice is one.

Source split is the existing 111 fit / 23 calibration / 25 validation images. D trains only on fit; source validation uses 16 registered simulator episodes x32 visits with seed 20261003. Fit images use identity appearance; validation retains the existing deterministic simulator. Augmentations never cross the image split. Patient independence is unknown. Target is the previously exposed four-domain development cohort, full 1951 arrivals/1695 principal images per order, or the sole cost-only downgrade to existing 1024/888 protocol. Warmup is retained for stateful methods.

## A: current-image prototypes

Capture detached up3 (256x128x128) from the original forward. Confidence seeds for background/disc-without-cup/cup use the frozen .1/.9 rules. Exclude native boundaries at radius 6, erode with a 5x5 square (radius 2), then project seeds by nearest interpolation. Require at least 16 feature-grid seeds per class. L2 normalize each feature and each mean prototype; cosine-softmax temperature .1. OD probability is disc+cup; OC is cup. Missing/nonfinite/near-zero prototypes cause a recorded whole-image fallback to DS.

DS is the existing decision-support logit rule: quarter flip logits only on native/quarter hard disagreement; native logits elsewhere. Edit band M is the Euclidean radius-6 band around DS boundaries, excluding reliable internal seeds, independently per channel. P=where(M,DS+.25*clip(logit(clamp(q,1e-6,1-1e-6))-DS,-1,1),DS). No iteration or cross-image prototype memory. A fixed seed-20261003 spatial permutation is a source-only diagnostic, with all extra forwards charged.

## B: connected structure tracking and verification

S generates local add/delete components at thresholds .40/.45/.50/.55/.60 relative to DS, inside M. Connectivity is 8-neighbor. Foreground parents match across levels by best spatial IoU>=.5; ties choose the lower raster-order component label. This is component tracking, not persistent homology or OT. P_VERIFY/P_SIMPLE share exactly P's changes; D_VERIFY uses only D_CONTEXT changes.

VERIFY requires valid prototype support (mean new-minus-old probability>=.05), >=3 matching levels, no reliable-seed deletion, no increase in OC-outside-OD violations, no increase in foreground fragments or enclosed holes versus the currently accepted merged state, and cumulative changed spatial pixels<=2% of the full grid. Sort by descending semantic support, level count, channel, first raster location, delete/add direction, then threshold. Same-pixel conflicts accept only the first accepted candidate; each merged state is checked once greedily. Rejections restore original DS logits. A threshold candidate writes DS-logit(threshold), once on its support.

SIMPLE uses per-channel largest component then hole fill of P's proposed hard mask, with raster-order ties. It accepts only P candidates consistent with that reference, while retaining M, protected seeds, containment and edit-budget checks; it does not use semantic or threshold stability. This morphology is extra method processing, not added to other baselines. Candidate supports, parents, levels and decisions are private audit evidence. Prototype verification is dependent evidence, not a second independent model.

## C: two-forward zero-order feature adaptation

A separate up3 affine adapter has w=(a,b) of dimension 512, initially exactly zero. Feature RMS is per channel over spatial dimensions of the same pre-injection feature, with epsilon 1e-6. F'=(1+a)F+b*RMS(F); explicit zero bypass preserves native output. No flip teacher or source-model output is used.

Each image probes center +/-mu*u with two original-resolution forwards, emits sigmoid(mean logits) before updating, and restores the center injection. Channel spatial-RMS normalization puts both probe logits on the common mean-logit scale; stopped foreground/background groups from that mean are equally weighted within each independent Bernoulli channel. Group/channel denominators exclude empty groups. No fixed area or previous-patient matching. Gradient estimate is finite difference times direction, clipped to L2 norm 1. PROBE never updates. CORE updates w by eta. BOUND replaces every fourth direction with a randomized anchor direction when w is nonzero, decays by .999, and projects RMS norm<=.05. Directed probes are not claimed unbiased.

Directions are keyed by seed/order/step; all three branches share the same underlying random stream. Source evaluates exactly four shared pairs: (.001,.0001),(.001,.0005),(.003,.0001),(.003,.0005), across CORE/BOUND and the balanced source episodes. Select by their mean hard Dice; exact ties lower eta then mu. Target seeds are 20261003 and 20261004 for both orders, with independent state and no domain reset. These seeds are algorithm randomness, not patient replication. Journals restore adapter, keyed random cursor, prediction prefix and visit count; failed work is not erased.

## D: source-only residual heads

Both heads have 269 inputs, 1x1 Conv->32/ReLU, depthwise 3x3->32/ReLU, 1x1->2, with zero final layer; equal capacity under 100k. Eight prediction slots are DS logits, original-minus-inverse-flip logits, Bernoulli entropy, normalized signed boundary distances. CONTEXT adds 256 up3 features, RGB and forward x/y luminance gradients at 128x128. LOGIT zeroes those extra slots after source-fit normalization. Both source cache and target inputs apply the same float16 rounding followed by float32 computation. Feature resizes use bilinear align_corners=False; seed mapping uses nearest.

Output is where(M,DS+.25*tanh(bilinear(head)),DS). Backbone/BN freeze throughout. Source fit-only means/stds are frozen, with std floor 1e-6. AdamW lr .001, wd .0001, batch8, 2000 updates per head; same seeded batch schedule and identical initialization. Loss is foreground/background-balanced BCE per channel plus soft Dice plus .1*mean probability change on originally correct pixels. Evaluate 500/1000/1500/2000 on source hard Dice; ties retain earlier checkpoint. Final selected source outputs include D_VERIFY. Target heads are frozen with no optimizer.

Source feature cache is bounded under 8GiB and retired after selected heads/statistics are sealed, before target journals. It stores the exact source image/style features used, not old features for a new augmentation. Target features are released after the current image; no complete target feature library exists.

## Matrix, accounting and decisions

C0/H025/DS/P/S/P_SIMPLE/P_VERIFY/D_LOGIT/D_CONTEXT/D_VERIFY share two backbone forwards per current image. Their second order is a verified content reordering; no duplicate static model run. Z_PROBE/CORE/BOUND have 2 orders x2 seeds and independent two-forward trajectories. Compatible C0/H025/G hard scalar references are reused with exact seals; C0/H025 additionally must match every new shared pixel count. DS hard equals H025 by construction; real probabilities cannot be aliased. G's complete historical trajectory is never stitched from short/complement parts.

Store packed hard masks and private candidate audits; target soft Dice/Brier/ECE are NA. Independent CPU scorer reads labels only after all registered prediction jobs are terminal. Algorithm workers have an audit hook that rejects target-mask file reads. Source-only fitting occurs before a target lock containing code/checkpoint/data/choice digests. Report all arms, orders, seeds, domains, OD/OC, adverse paired tails, resource costs and failures. Z averages seeds within each order first. Development signal requires >0 versus DS and nonnegative in both orders. New-data-confirmation candidate requires >=.5 pp versus C0, >0 versus DS and no >2 pp worst domain/channel drop; these are descriptive rules, not significance or clinical claims.

Original T0 is 2026-10-04 00:00 Beijing, conservatively earlier than first preparation. Wall cap24h, cumulative GPU-worker12h (normal10h+recovery2h), at most2 simultaneous GPU workers in authorized pool4/5/6/7. Stage normal caps: preflight1h, source3h, target6h. CPU and I/O while a worker owns a GPU remain charged. Every failure/profile counts. Profile projections use factor1.3 and preserve all qualified directions; no performance-based arm removal, resolution reduction, parameter sweep or further campaign.

## Primary sources and differences

Read versions: [CLIP controlled study v2](https://arxiv.org/html/2606.14299v2), [TopoOT v1](https://arxiv.org/html/2601.20333v1), [EVA-0 v1](https://arxiv.org/html/2605.18867v1), [ORCA v1](https://arxiv.org/html/2606.14222v1). A uses fundus decoder features rather than a CLIP classifier/update reproduction. B omits persistence/OT/online head training. C adapts EVA-0's two-forward idea to two Bernoulli segmentation channels and a low-dimensional affine feature adapter. D cannot observe future target error as ORCA does; no target residual buffer, Bayesian routing or online head fitting.

Historical R3/R4 use teacher/transport and boundary-geometry paths; M3 conditions synthetic proxy pixels and rehearses online. New A is image-internal feature evidence, B is explicit spatial component correspondence, C is derivative-free decoder adaptation, D is a separate source-supervised frozen correction head. They are not renamed old mechanisms. Historical negative carrier/RL outcomes remain unchanged.

Baseline checkpoint training-seed/membership details are not established here; the same bound checkpoint is reused. Head fit/val image isolation does not imply an independently unseen backbone validation cohort.
