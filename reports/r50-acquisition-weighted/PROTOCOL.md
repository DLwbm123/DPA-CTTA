# R50 acquisition weighted structural filtering

This round tests the CPU-qualified acquisition-weighted filter on C. It is frozen
before same-round profiles, formal adaptation or label scoring. Only this round
is authorized. No automatic next round, scientific/technical retry, tuning,
scan, extra seed, retrospective primary switch or relaxed admission/gate.

## Fixed conditions and update

| ID | Background | Filter | Physical calls per image |
|---|---|---|---|
| C | original C | none | 8F/1B/1Adam |
| C_JS_SHARED | original C | original R46 shared matrix | 10F/5B/1Adam |
| C_JS_ACQ_WEIGHTED | original C | acquisition-weighted structure matrix | 10F/5B/1Adam |
| CW_LSO | original C+W, original LSO0.1 | none | 8F/1B/1Adam |

Primary: C_JS_ACQ_WEIGHTED minus C. The matched mechanistic comparator is
C_JS_SHARED, not the W/LSO-enabled original R46 condition. CW_LSO remains the
unchanged strong control. No W or LSO is added to the new candidate.

Use the implementation qualified at e734e349d19f4fc6d6b407290f5b3e87647efb63.
For actual native Adam proposal d, H=sqrt(v_hat)+eps, X=H^(-1/2) acquisition,
Y=H^(-1/2) structure sensitivities, Q=orth([X/||X||_F,Y/||Y||_F]),
A=Q^T X X^T Q/||X||_F^2, S=Q^T Y Y^T Q/||Y||_F^2 and R=A+0.01I.
The new T=R^(1/2) S R^(1/2), K=(I+T)^(-1),
d_submitted=H^(-1/2)[w+Q(K-I)Q^T w], w=H^(1/2)d.
The original shared condition uses R^(-1/2) instead. This changes both effective
strength and direction selection; it is not an exact subspace intersection.

Retain rank<=4, strength1, ridge0.01, original Gram cutoff
max(1e-12,1e-6*largest eigenvalue), Sensor/evidence, two fixed photo probes,
four auxiliary VJPs, six weak views, original strong augmentation, source model,
BN-affine-only parameters, Adam LR1e-4/moments/step count, preprocessing and final
prediction. Rank0/missing blocks preserve native parameters exactly; zero strength
delegates to the exact C path. No R48 descent constraint or repair fallback,
SCNP/SE/GC/GP/RL, new model, loss, optimizer step, memory, label/domain action or
output fusion. Read-only Adam hooks record original and actual-write loss
directional derivatives in all four arms without controlling any update.

## Fresh matrix and resources

Use original seeds20260907/17011/29009 and orders0/1, each1951 arrivals:
24 new complete trajectories,46824 independent CPU score rows. No previous
control trajectory or isolated-profile prefix is reused as a result.
Scoring is CPU-only after all formal online workers have retired.

Host jiangsuiyang, physical GPU4/5/6/7 only, at most4 GPU-workers. Bind physical
UUIDs; sharing allowed, do not modify other tasks. Use the isolated NAS code,
qualified Python3.10.16/PyTorch2.6.0+cu124 environment and migrated source model.
Do not claim cross-server bitwise numerical identity with earlier experiments.
Source and image identity checks built into the retained runtime remain active.

Four same-round12-arrival profiles (4 warmup+8 timed) must all PASS. The entire
24-job matrix must pass retained conservative budget admission; no pruning.
Formal worker free memory must be >=max(1.2peak,peak+512MiB) using this same round,
same condition's PASS peak_reserved_bytes. Missing/invalid profiles refuse.
Cold profiles retain the original5GiB peak guard and its usual margin.

Independent original T0 immediately before first same-round GPU work;22h online,
24h wall,48GPU-worker h,32GiB. Initialization, profiles and failed attempts count;
no reset. Previous14 closed/stopped rounds cost117.10121225502758GPU-worker h;
older independent profiles668.9714405536652s. The new isolated acquisition-weighted
profile cost194.9522364139557s. Total before R50=117.34119105418526GPU-worker h.
Old T0/ledgers/results remain unchanged. Record this preparation's CPU costs and
any failure independently. All visible parent/worker entry points are neutral.

## Frozen scientific evaluation

Primary SEARCH comparison against C requires >=0.1pp macro Dice, paired95% CI
lower>0, both order means positive, >=5/6 seed/order trajectories positive,
imageweighted>=0 and worst seed-mean domain/channel/order cell>=-2pp.
This continues the R49 standalone test criterion. No threshold relaxation or
primary replacement after results. Apply exactly the same gate separately to
new minus C_JS_SHARED and new minus CW_LSO before claiming improvement over each.
Passing against C does not establish improvement over either retained control.

Prespecified paired contrasts in SEARCH and descriptive REVIEW: new-C,
new-C_JS_SHARED, new-CW_LSO, C_JS_SHARED-C, CW_LSO-C. Report macro/imageweighted,
all seed/order/domain/OD/OC effects, both positive and negative cells and paired
intervals. Use retained domain-stratified2000-repeat content-cluster bootstrap,
seed20261005, coupling all seeds/orders of each content. Domains/channels are
equally weighted before seeds/orders. Patient dependence UNKNOWN. Secondary
intervals are descriptive, without multiplicity correction.

Specifically assess REFUGE_Valid improvements and Drishti_GS/ORIGA losses in each
channel, without tuning the method to any observed domain. SEARCH and REVIEW have
historical exposure; REVIEW is descriptive and cannot select or retune. No
independent-generalization, clinical, anatomical-protection or causal claim from
structure proxies, contraction, weaker filtering or pseudo-loss derivatives.
Dice and empty-mask conventions remain unchanged. ASSD/Brier/soft Dice are not
computed. Incomplete/unscored cells are NOT_EVALUABLE, never zero.

## Diagnosis and completion delivery

Retain all native/submitted directional derivatives and differences, joint rank,
K eigenvalues, acquisition/anatomy capture, proposal span energy, relative metric
update change/norms, OD/OC structure evidence, base/LSO loss, BN gradients/updates
and existing region-logit diagnostics where available. Missing C-only region hooks
remain missing. Report aggregate diagnostics by domain as well as condition.
The isolated profile found a much weaker intervention; this is not Dice evidence.

Reuse the background watchdog/supervisor, receipts, current-image/model-only
filesystem boundary, same-round profiled memory admission, complete online
retirement barrier, independent CPU scorers and anonymous report generation.
Stop on new numeric/resource/technical failure; preserve all partial evidence and
cost. No automatic technical retries or formal recovery from saved checkpoints.

User-authorized heartbeat monitoring every30min. Healthy/routine progress stays
quiet. Notify completion/failure/budget stop/delivery blockage/required choice.
At termination verify watch/supervisor/online/scorers all retired,24 unique full
jobs,24 independent CPU scorers each1951 rows,46824 total rows and52 attempts
(4 profiles+24 online+24 score). Preserve every failure and every cost.

Publish own source/protocol and anonymized aggregate positive/negative results,
CI, cost and audit under reports/r50-acquisition-weighted on branch
experiment/r50-acquisition-weighted-v1 in DLwbm123/DPA-CTTA. Every GitHub operation
uses explicit local proxy with NO_PROXY/no_proxy cleared; no direct fallback,
force or unrelated overwrite. Verify remote commit SHA and anonymous REPORT HTTP
before local/remote DELIVERY and CAMPAIGN closure. Startup delivery is not a
scientific result. Never publish images, labels, per-content scores/predictions,
patient/content identifiers including hashes, model states, private paths/logs or
third-party PDFs. Delete the monitor after verified delivery regardless of gate;
if delivery is blocked retain it only for necessary delivery retries. No R51.
