# DSR dense module transfer: one frozen small pilot

Registered before target inference, 2026-10-18. User authorized one new ICLR module, rationale first, a small controlled pilot without RDC/ENGRAM, completed analysis and GitHub delivery. No tuning or successor is authorized.

## Rationale and source status

Borrow the dynamic hard-supervision module from [DSR](https://openreview.net/forum?id=dMBuehC5eR), equations2–7, appendices A.5/A.6 and Table6. The current OpenReview page was read on2026-10-18 and explicitly says **ICLR2027 Conference Desk Rejected Submission**. This is a public submission mechanism, not an accepted or currently reviewed paper. Abstract metadata from an older local retrieval did not establish this current status. Full method/appendix was downloaded from its public PDF link and read before implementation.

Why worth one test: our OD/OC are independent nested sigmoid outputs; the original module uses independent Bernoulli predictions rather than mutually exclusive softmax classes. Current CW_LSO primarily has soft consistency; DSR adds explicit high-confidence positive and negative guidance only when temporal teacher/student discrepancy is elevated. This may counter conservative updates or excessive foreground predictions. This is a hypothesis, not established efficacy. Prior error analyses involved different historical correction arms and do not prove this pilot's baseline has the same error mechanism.

Compared with ENGRAM, no image-to-domain memory key is needed. Compared with RDC/Soft-PSCL, it does not infer pair relationships across spatial locations with highly unequal classes. HRAC's composition correction requires task-specific tumor composition bounds that do not transfer safely to cup/disc pathology; those bounds are not borrowed. We select DSR for mechanism compatibility, not conference prestige or a guaranteed gain.

## Fixed arms

All arms retain CW_LSO, source checkpoint, existing BN-only Adam, learning rate, W, LSO, six weak and one strong consistency views, prediction timing, and update order. No RDC/ENGRAM invocation or new trainable parameters.

- NATIVE: exact existing CW_LSO.
- SOFT: same host, replace only its soft BCE target by a clean EMA-teacher prediction; no hard term.
- ALWAYS: same as SOFT, add0.25 ASL on eligible predictions on all arrivals after20 warm-up arrivals.
- DSR: same as SOFT, add0.25 ASL only after warm-up when D_t>1.2 reference_t.

SOFT isolates the EMA target change. ALWAYS separates hard supervision from the dynamic exposure rule. The host's LSO retains its native weak-view target. EMA teacher exists only in the three non-NATIVE arms, predicts one current clean image before host update, and updates its BN affine coordinates after the host step with momentum0.999. Frozen coordinates remain fixed. All teacher forwards restore buffers and RNG; no target BN-EMA, stochastic restoration, full-model AdamW, teacher-output evaluation, or source calibration from the paper is added. These are explicit host-transfer differences, not full DSR reproduction.

## Dense transfer and fixed settings

D_t is the mean Bernoulli KL from clean teacher q to clean preupdate student p, over both channels and all current-image pixels. Student clean prediction reuses the host's first weak forward. Clamp probabilities to[1e-6,1−1e-6] for this diagnostic. D_t only controls the gate, does not receive gradients or estimate true reliability.

First20 arrivals: hard off in all arms; keep positive thresholds0.8 and set discrepancy reference to the mean of20 observed D_t. After warm-up, gate uses previous reference, then reference←0.999 reference+0.001D_t.

Each channel keeps a4096-value queue. Append64 teacher values from fixed image positions32,96,…,480 on an8×8 grid; keep the most recent4096. This dense sampling preserves cross-image history; it replaces the paper's per-class image-score history. After warm-up, compute queue95th percentile clipped[0.60,0.95], then threshold←0.9threshold+0.1candidate. Current image is appended before this update. Positive q≥threshold; negative q≤0.30; all other pixels ignored. Hard ASL averages over eligible pixel/channel entries; zero if none. gamma_pos0, gamma_neg2, negative probability clip0.05; focal factor is differentiated. These settings are fixed from Table6, not tuned on target labels.

Background domination, anatomical correlation, and calibration of source segmenter probabilities may make this transfer fail. Dense Bernoulli KL is not a segmentation-quality estimate. Fixed sampling is not a patient-independence guarantee.

## Cohort, stream, seed and endpoints

Take the first22 contents from each of four domains in the previous completed pilot's frozen order0 list; exactly88 already-exposed images. Each domain's first11 form cycle0 and remaining11 cycle1; membership shared across orders. Preserve original within-domain order for each order. Each cycle follows original domain order, with order1 reversed. No image repeats within a trajectory. Offline preparation/scoring may read domain/cycle metadata; online rows contain only registered image identity/path/size. No labels, future-image reads or domain-boundary hints online.

Four arms × two orders ×88 arrivals = **8 trajectories /704 predictions**, seed20260907. Fresh source initialization per trajectory, persistent state between cycles. No old scores reused as current controls; no direct absolute comparison to earlier192-image streams.

Primary: full-stream DSR−NATIVE, equal domains/channels/orders. All fixed conditions must pass: mean≥+0.2 Dice pp; paired content-stratified95% lower bound>0; each order>0; worst domain/order/channel≥−2pp; DSR−SOFT mean>0; DSR−ALWAYS mean≥0. SOFT−NATIVE and ALWAYS−SOFT plus per-cycle outcomes are descriptive. No post-hoc primary switch.

2000 bootstrap draws, seed20261016, domain-stratified contents, repeated orders and OD/OC coupled. Single seed, fixed orders, exposed development images, unknown patient linkage: no independent/generalization validation. Report all negative cells and mechanism diagnostics, including hard-active fraction, q threshold/positive/negative fractions, KL reference, BN norms and timing. ALWAYS is an identical per-active-step coefficient control; it is not exposure-budget matched. A DSR win alone cannot prove gate quality or distinguish gating from reduced cumulative hard dose.

## Engineering, budget, stop and delivery

CPU checks: warm-up/gate/reference/queue updates, threshold bounds, snapshot restore, correct positive/negative hard-gradient signs, ignored-pixel zero gradient, empty-valid loss. Real-model checks: repeat state/output for all arms and both orders; NATIVE exact retained-host parity; first warm-up step SOFT/ALWAYS/DSR parity; forced-hard qualification path exercised separately. Frozen weights and host operation contracts enforced. Qualification inputs never scored as scientific predictions.

GPU cap1800 seconds including qualification, CPU scorer300 seconds, output cap2GiB; existing detached controller enforces caps and preserves failure/cost receipts. Admit full matrix only if slowest qualification step×704×1.5 plus used qualification fits. No scientific retry, threshold scan, seed expansion or automatic next experiment.

Score only after GPU worker exit0, complete8 trajectory receipts and704 predictions. Verify scorer/controller retirement and coverage, publish own code/protocol/aggregate tables/cost/check summaries and analysis through mandated GitHub proxy. Do not publish images, masks, checkpoints, per-image identifiers/scores/predictions, private server paths, or third-party PDF.
