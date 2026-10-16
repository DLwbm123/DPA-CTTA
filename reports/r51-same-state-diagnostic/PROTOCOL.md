# R51: fixed late-state direction diagnostic

Authorized scope: restore/gradient qualification followed, only on success, by
96 fixed state–image pairs. No replay, continued stream, tuning, retry, added
sample, source training, or follow-up experiment. The diagnosis is confined to
two existing CW_LSO states at visit 1900, seed 17011, orders 0 and 1 of R50.
These are two states, not 96 independent states or new cohorts.

## Inputs and frozen selection

Use the original source model plus existing journal snapshots (parameters,
buffers, Adam moments, global/native RNG, modes and module state). Never mutate
the original journals. Select 12 contents per domain from the historically
exposed SEARCH partition, sorted by existing content identity before a single
NumPy Generator(seed=20261016) permutation in alphabetic domain order. Both
states use the same 48-image panel. No labels, predictions, or previous effects
enter selection. Model state may already have encountered panel contents.

## Updates and matching

Use unchanged CW_LSO, LSO coefficient 0.1, native Adam, and the existing R46
shared filter (strength 1, ridge 0.01). Compute full gradients of weighted CW
consistency and unweighted LSO over every eligible BN parameter from the same
strong view and detached weak target. Their weighted sum must match the native
combined gradient (relative norm tolerance 1e-5). No gradients are approximated
by loss scalars or norms. Preserve native Adam moments.

Eight predictions per pair: no update, native, shared J, radial contraction
with the J update norm, and four geometry-matched nulls (seeds 1103, 2207, 3301,
4409). In Adam metric H=sqrt(v_hat)+eps, write w=H^.5 delta_native,
v=H^.5 delta_J, h0=H^-.5 grad(CW), hs=H^-.5 grad(LSO). Let p be the projection
of v onto span(w,h0,hs), r=v-p. Each null is p+||r||q, with q a unit random
direction in the orthogonal complement. Therefore final H-norm, H-distance
from the native proposal, and both loss directional derivatives match J.
This tests the whole J direction choice, not low rank separately from its
orientation. Euclidean norms are reported but not matched.

The normalized-column SVD rank cutoff is 1e-10. If ||r||/||v|| <=1e-6, nulls
equal J and the pair is marked degenerate; no replacement or sample expansion.
Before write, double-precision relative matching tolerance is 1e-10. After
float32 parameter write, norm and modification relative errors must be <=0.005;
gradient-dot discrepancy divided by ||H^-.5 g|| ||v|| must be <=0.0005.
No tolerance relaxation, clipping, repair, resampling of directions, or
re-normalization after a failed write. Numerical failure stops the campaign.

All prediction branches share nonparameter buffers, modes and evaluation RNG.
Each pair starts from its anchor. Shadow state is never fed into another pair.
Full gradients and sensitivity/geometry vectors stay private.

## Qualification gate

CPU generated fixtures test matching, degenerate cases, invalid-input rejection,
gradient decomposition, native-step parity and restore repeatability. Before
the matrix, for each anchor use its next historical image once as a qualification
case (not a sequential replay): check exact restored snapshot, repeat native
one-step equality, original saved binary prediction equality, and equality of
instrumented native parameters, gradients, Adam moments and native RNG. Check
all representable-write matching tolerances. No labels are read during this gate.
These two cases are qualification only and never counted toward 96 formal pairs.
All their cost is billed. Same-round projected full work must fit the caps before
formal work. A failed gate is NOT_EVALUABLE, not a scientific negative result.

## Outcomes and analysis

Save all eight masks and label-free diagnostics for all 96 pairs before a
separate CPU scorer is allowed to read labels. Primary: J minus the mean of the
four null Dice values (not a voted mask). Auxiliary: J minus radial, J minus
native, and native minus no-update. Equal weights for domains, OD/OC and the
two fixed anchors. Retain every positive/negative domain/channel/anchor cell.
Bootstrap 2000 times, seed 20261016, stratified by domain and clustered by content;
all anchors/channels/nulls of a content stay coupled. These intervals are
conditional on two fixed late states and cannot quantify state, cohort or
patient-level generalization. Do not fit a high-dimensional predictor with 48
content clusters.

Only discuss further investigation if primary mean >=0.1pp, its 95% lower bound
>0, both anchors positive, J minus radial mean >0 and J minus native mean >=0.
Opposite-anchor signs, null superiority, or radial-only gains do not support the
direction claim here. Wide intervals are inconclusive, not permission to expand.
No conclusion about early/middle states, long-term stream effects, anatomy
protection, or independent generalization follows from this diagnostic.

## Resource and delivery

One worker; total charged GPU-worker wall time <=1800 seconds, including imports,
initialization, qualification, diagnostics and failures. Qualification <=360s,
formal pair phase <=1080s; remaining <=360s is overhead, not retries. Launch-time
deadline never resets. Stop early enough to preserve receipts within total cap.
CPU scoring/analysis <=900s; new artifacts <=1GiB. Existing mandatory asset
integrity checks remain; no new whole-archive checksums or bulk scans.
GPU admission requires at least 6GiB free and matching assigned UUID; do not
interrupt other processes. A supervisor enforces timeouts and records failures.

Publish own source, this protocol, aggregate results (including negatives),
qualification summary, costs and completion audit to the project repository.
Never publish input images, labels, raw masks, per-content scores or identities,
full gradients, weights, private paths or raw logs. Failed attempts remain in
the ledger. No automatic retry or monitoring automation.
