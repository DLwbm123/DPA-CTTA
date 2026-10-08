# R30: delayed effects and cross-arrival state value

Authorized 2026-10-08 after review of the proposed A/B experiment. This freezes
the complete experiment before profiling or outcome access. No RL, reward fitting,
new residual actions, strength/period search, source training images, or new models.
Historical SEARCH and REVIEW are both development-exposed. Patient linkage and
ROI provenance remain UNKNOWN. Old REVIEW numbers are not current SEARCH baselines.

## Budget and admission

Use available A100 GPUs 0/1/2 without modifying other jobs. T0 begins with real
profiling. Absolute wall cap 24 hours; online cutoff 22 hours; all profile,
initialization, failed work and recovery count toward 36 GPU-worker hours. New
private output cap 64 GiB. Keep at least 20% recovery reserve within the admission
estimate. Profile all eight B arms on 34 real arrivals on each GPU, including the
first periodic reset and extra diagnostic/reference forwards; profile one A
snapshot with 8 future arrivals per GPU. No labels are available to profiling.
Admit A and B together or mark BUDGET_BLOCKED; no outcome-dependent pruning or
budget reset. No automatic scientific retry. Repairs retain T0 and failure costs.

## Unchanged data and scoring

Original source checkpoint, 512x512 input, original preprocessing and OD/OC hard
Dice. Each complete stream has 1,951 arrivals: 1,017 SEARCH, 678 legacy REVIEW,
256 context. Seeds 20260907, 17011, 29009; two existing orders. All arrivals are
processed; only SEARCH and old REVIEW can be scored. Context is never given a
zero score or included in score denominators. No labels, historical scores,
domain identities, or future inputs enter the update decision. Domain/role data
are scorer-only. The online runner has image-only manifests and fixed integer
probe positions. All 48 B streams and six A jobs must retire before CPU label
release. Failed/incomplete matrices have no scientific-negative interpretation.

## State and mechanics

C uses current-image BN batch statistics: running_mean/running_var are absent.
Persistent learning state is BN affine and Adam m/v/local step; ANCHOR also has
adapter weights/moments. Preserve model buffers, frozen parameters, hooks, global
arrival position, native random stream and physical accounting. Inventory modes,
Dropout and BN configuration. S_SOURCE versus S_BATCH is a configuration contrast;
attribute it solely to BN only if the mode inventory rules out other differences.

Adam reset clears its state and native optimizer-local step; it does not recreate
the optimizer, alter hyperparameters or reset global arrival/physical counters.
Parameter restoration copies source BN affine only, preserving optimizer state.
Generated-input and real-GPU checks cover unchanged C/ANCHOR reference replay,
frozen states, isolated parameter/optimizer reset, reset timing, true skip,
read-only diagnostic isolation, native RNG preservation and mask coverage.

## B: complete-stream interventions (primary resource decision)

Eight arms x three seeds x two orders = 48 streams / 93,648 algorithm arrivals.
Every stream starts independently from the source checkpoint.

| Arm | Cross-arrival behavior |
|---|---|
| S_SOURCE | Frozen source, native eval inference |
| S_BATCH | Frozen source parameters, C current-image BN and model mode |
| C_CONT | Unchanged consistency Adam, retain affine and optimizer |
| C_EPISODIC | Before each image restore source affine and clear Adam |
| C_OPT32 | Only clear Adam before arrivals 33,65,97,... |
| C_PARAM32 | Only restore source affine at those arrivals |
| C_BOTH32 | Restore affine and clear Adam at those arrivals |
| ANCHOR | Unmodified current-image-conditioned adapter reference |

C-derived arms keep six weak views, one strong view, one Adam update, fixed
BN LR 1e-4, final student readout and all other native settings. ANCHOR retains
adapter LR 3e-4. Period 32 counts all arrivals, not scored cases or true domain
boundaries. No period scan. Formal post-update output is the primary prediction.
Read-only pre-update masks at the existing 16 fixed positions/order are auxiliary;
for reset arms they are recorded after the registered reset and before adaptation.
Log affine drift/update norm, Adam m/v norms, local/global clocks, reset events,
and fixed-position before/after mask change. Charge diagnostics separately from
algorithm operations. Pre-update predictions must not change method RNG/state.

Re-run C_CONT and ANCHOR. Compare S_BATCH-S_SOURCE, C_EPISODIC-S_BATCH,
C_CONT-C_EPISODIC, each periodic intervention-C_CONT; report all arms against both
C_CONT and ANCHOR. Interaction is BOTH32-PARAM32-OPT32+CONT. Adam reset changes
moments and bias-correction clock together; it cannot isolate stale first moments.
Parameter-only restoration deliberately retains potentially mismatched moments.

Primary: SEARCH Dice equally averaged over domain/channel/seed/order. Image-weighted
Dice is auxiliary; REVIEW is separately descriptive. A candidate among EPISODIC,
OPT32, PARAM32, BOTH32 is worth further review only if it beats the single stronger
whole-SEARCH macro reference (C_CONT or ANCHOR) by >=0.3 pp, both orders improve,
at least 5/6 seed/order groups improve, and worst seed-averaged domain/channel/order
cell is >=-2 pp. All engineering/coverage checks must pass. No per-cell reference
switching. These are allocation gates, not significance/clinical thresholds.
Report all effects, six group effects and descriptive paired content bootstrap
intervals (2000 draws; domain-stratified, seed/order coupled). Patient independence
and selection-adjusted inference are not established. No fabricated soft metrics.

## A: delayed consequence of one action

Reuse all 96 R28 pre-arrival snapshots; no score-driven selection or new states.
FULL, ZERO, HALF act only on the current image. ZERO truly skips backward/Adam;
HALF halves both final learning rates for that one update. Every branch then
resumes original ANCHOR for up to 64 successive real arrivals, retaining its own
learned state. Do not restore the reference state at each future image.

FULL runs first and records its per-arrival native RNG input. Alternative branches
receive the same exogenous RNG at the corresponding arrival. Skipping backward
must not shift subsequent augmentation draws. No image-dependent or label-dependent
reselection. Formal FULL masks must exactly match both saved R28 current masks and
the newly completed B_ANCHOR full stream at every visited position. B therefore
runs before A, with no intermediate score release.

H=0 means current image; H=8/32/64 means only t+1..t+H. H=64 is primary. A window
is eligible only when the whole H-arrival future exists and contains at least one
SEARCH position. Determine eligibility, denominators and domain composition from
manifests before launch. Process truncated tail branches but report them separately
as ineligible for that H; do not wrap, extend, replace or score context. Each window
mean gives equal weight to authorized images and OD/OC; the outer mean equally
weights origin domain/seed/order. Report denominators and future-domain composition
so this local estimand is not confused with full-stream B.

Report ZERO-FULL, HALF-FULL, window-level oracle including FULL, oracle-FULL and
oracle-best-fixed-action. A legal oracle selects one entire first-action branch
per window, never switches per future image. Ties choose FULL then ZERO then HALF.
Report future mean effects and cumulative sums separately. H=64 oracle >=0.3 pp,
both orders positive and >=5/6 positive groups flags allocation-scale delayed
headroom only. It does not demonstrate a usable selector. Report all negative
channel cells. Overlapping windows and repeated content/seeds are not independent
patients; no iid-window confidence claim. No A outcome cancels admitted B.

## Decision, publication and stop

B promising: recheck the simple state strategy first. A promising/B weak: delayed
signal without a demonstrated full-stream strategy; no automatic RL. Both weak:
stop sweeps of these actions/host/period; a future change in supervision or adaptive
representation needs a new justified protocol. Incomplete work is BLOCKED, not
negative science. No R31, hourly monitor or other automatic experiment is created.

Deliver registration, protocol, inventory, mechanical/profile receipts, horizon
and full-stream results, paired contrasts, negative cells, cost, completion audit,
report and next decision. Only source/protocol/aggregates/audit summaries may be
published; no images, masks, labels, identities, snapshots, weights, credentials
or private paths. Verify GitHub remote commit and anonymous access before claiming
public delivery. Keep all failed attempts and original experiment history.
