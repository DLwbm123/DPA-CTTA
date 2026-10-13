# R47: descent-constrained shared imaging/anatomical sensitivity

Frozen before same-round GPU profiles, online execution and scoring. User
authorized the descent-constrained version with C/W/CW_LSO/R46 SHARED controls,
physical GPU0/1 only, quiet30-minute monitoring, full positive/negative analysis
and GitHub delivery. No automatic next round, scientific retry or parameter scan.

## Hypothesis and retained method

R46 SHARED improved several REFUGE_Valid deficits but changed SEARCH macro Dice
by-0.025136pp versus CW_LSO and-0.018238pp versus original JS. It could attenuate
helpful correction as well as harmful drift. A local first-order CW_LSO descent
constraint may preserve useful adaptation while retaining shared sensitivity
protection. This is a testable hypothesis, not demonstrated anatomical protection.

Use the CPU-qualified method in JOINT_SENSITIVITY_DESCENT_CPU.md. Same Sensor,
two acquisition probes, four VJPs, rank<=4 shared basis, strength1/ridge0.01, Gram
cutoff, W and LSO0.1, source ResUNet, fixed native Adam and BN-affine-only update.
For H=sqrt(v_hat)+epsilon, delta_A=theta_after_Adam-theta_before,

    w_A=H^1/2 delta_A; v_A=Q^T w_A; K=(I+T)^-1
    g=gradient from existing CW_LSO loss backward; c=Q^T H^-1/2 g
    v_0=K v_A; violation=c^T(v_0-v_A)
    if violation>0: mu=violation/(c^T K c); v_*=v_0-mu K c
    else: retain exact original SHARED update
    delta_*=H^-1/2 [w_A+Q(v_*-v_A)].

This minimizes0.5||v-v_A||^2+0.5v^T T v subject to c^T(v-v_A)<=0.
Original g^T delta_A>=0, missing sensitivity/rank0, or a nonpositive numerical
constraint denominator keeps the actual Adam proposal. No line search, new loss,
extra network pass or optimizer step. Cache g only transiently; Adam moments and
step count unchanged. Finite-precision committed derivatives are audited with the
CPU-frozen tolerance1e-5*max(abs(proposal_dot),1e-10). A violated audit stops the
attempt; no tolerance tuning or automatic retry. SHARED retains its original
metric contraction audit. DESCENT no longer requires global metric contraction.
These are numerical checks, not efficacy guarantees or clinical thresholds.

Both joint versions remain10F/4 VJP+1 loss backward/1Adam. CW_LSO8F/1B/1Adam.
No GC/GP/GDP/SCNP/SE/RL, new model, permanent memory, label/domain decisions or
output fusion. Disabled constraint preserves frozen R46 exactly; disabled joint
module preserves CW_LSO exactly on CPU qualification.

## Fixed matrix, resources and admission

C/W/CW_LSO/CW_LSO_JS_SHARED/CW_LSO_JS_SHARED_DESCENT; primary is DESCENT.
Original seeds20260907/17011/29009, two original orders,1951 arrivals each:
30 fresh trajectories and58,530 independent CPU-scored rows. No old trajectory
reuse or profile promotion, no extra images or seeds. Existing historically
exposed SEARCH/REVIEW split. Only physical GPU0/1,max2 GPU-workers; allow sharing,
never intervene in other tasks. R46 and independent resource profile must retire.

Independent immutable T0 before R47 GPU work;22h online/24h wall/48GPU-worker h/
32GiB. All same-round profiles, failures and attempts charged. Every condition
needs a PASS12-arrival profile, followed by whole30-job conservative dual-worker
admission. Formal worker uses the same-round/same-condition measured PASS peak,
free>=max(1.2*peak,peak+512MiB); missing/invalid peak refuses. Cold profile5GiB
guard unchanged. No pruning or resetting T0/budget. Prior11 closed experimental
rounds cost91.68475513378781 GPU-worker h; independent JS profile74.37265658378601s
and descent profile123.62959313392639s bring aggregate91.73975575870939h.

Prior DESCENT/SHARED profiles used fresh source models on each of physical0/1,
12 arrivals per job,4warmup/8timed, with reverse condition order. Four PASS,
same peak allocated816.875MiB, reserved896→898MiB; wall ratios1.009750/0.994854.
These shared-GPU short measurements establish resource feasibility only; they
are not same-round R47 profiles or a segmentation result. The prior profile's
pre-GPU CPU metadata-dispatch failure is retained and does not reset old costs.

## Frozen evaluation and full report

Primary SEARCH versus C/W each>=0.3pp; versus CW_LSO/R46 SHARED each>=0.1pp.
All four require both orders positive,>=5/6 positive trajectories,imageweighted
>=0,worst seed-mean domain/channel/order cell>=-2pp. No primary replacement or
relaxation. Report full paired intervals and every positive/negative cell even
if failed. Specifically compare REFUGE_Valid OD/OC preservation against new
Drishti_GS/ORIGA deficits, separating individual-seed and seed-mean results.

All online workers retire before independent CPU labels/scoring and REVIEW
release. SEARCH and legacy REVIEW are historically exposed; REVIEW descriptive,
never a tuning signal or independent generalization claim. Report constraint
trigger/fallback rates, raw/unconstrained/filtered/actual directional derivatives,
unconstrained gap/multiplier, shared rank/eigenvalues/capture/energy/update change,
OD/OC evidence, BN/region gradients, every attempt and total costs. Existing K
eigenvalues describe the unconstrained filter, not the affine-corrected operator.
Proxy loss descent and local sensitivity contraction do not prove anatomy or
Dice protection; finite-step loss, ASSD/Brier and patient generalization untested.

Only necessary30-minute checks while healthy; stay quiet for routine progress.
Notify completion, failure, budget stop, delivery blockage or required user action.
No restart, repeat launch, retuning or automatic next round. On actual completion
verify retirement/30 unique jobs/30 scores/58,530 coverage, analyze and deliver
own source/protocol and anonymous aggregate reports to reports/r47-shared-descent
on experiment/r47-shared-descent-v1 in DLwbm123/DPA-CTTA. Every GitHub operation
uses explicit local proxy; verify remote SHA and anonymous report access, update
local/remote DELIVERY and cumulative CAMPAIGN, then delete this monitor regardless
of scientific success. Keep failed attempts and costs; delivery blockage permits
only necessary delivery retries. No images, masks, per-content scores/predictions,
patient/content identifiers including hashes, model state, private paths/logs or
third-party PDFs are public.
