# R48: repaired descent-constrained shared sensitivity

Frozen before same-round GPU profiles, full trajectories or scoring. User
authorized the repaired formal round, C/W/CW_LSO/R46 SHARED controls, only physical
GPU0/1, original tolerance, quiet30-minute monitoring and all-result GitHub delivery.
No automatic next round, scientific retry, scan, extra seed or data relaxation.

## Hypothesis and exact retained algorithm

R46 SHARED traded some REFUGE_Valid improvements against Drishti_GS/ORIGA losses;
SEARCH macro change versus CW_LSO was-0.025136pp. R47 stopped on committed-derivative
checks before any complete candidate or score. Its scientific result is NOT_EVALUABLE,
not a segmentation loss. The retained hypothesis is that preserving the native
CW_LSO first-order descent while filtering shared sensitive directions can reduce
the tradeoff. This is not established anatomical protection or novelty evidence.

Use the repaired method from SHARED_DESCENT_NUMERICS_CPU.md and source3b01d6c.
CW_LSO losses/views/W parameters/LSO0.1, source model, BN-affine-only native Adam,
LR/moments/step counts, Sensor/two probes/four VJPs, rank<=4 shared basis,
strength1/ridge0.01 and original Gram cutoff remain exactly as R47.

    H=sqrt(v_hat)+epsilon; delta_A=theta_after_Adam-theta_before
    w_A=H^1/2 delta_A; v_A=Q^T w_A; K=(I+T)^-1
    g=existing CW_LSO loss gradient; c=Q^T H^-1/2 g
    v_0=K v_A; violation=c^T(v_0-v_A)
    if violation>0: mu=violation/(c^T K c); v_*=v_0-mu K c
    else: retain exact original SHARED arithmetic
    delta_*=H^-1/2[w_A+Q(v_*-v_A)].

This is the registered shared-space constrained quadratic update. Native
non-descent proposals, missing sensitivity/rank0 and degenerate denominators
keep the exact actual Adam proposal. For a representable filtered commit, retain
the ORIGINAL guard when proposal_dot<0:

    actual_dot-proposal_dot <= 1e-5*max(abs(proposal_dot),1e-10).

If the candidate commit is nonfinite or violates that check, restore the saved
post-Adam parameters bitwise, measure final derivative again and retain the guard.
No tolerance relaxation, precision-mode change, extra network call, line search,
coefficient sweep or another Adam step. Unresolved final failures stop the attempt.
The fallback can remove a useful joint correction and is not a new KKT optimum.
It must be reported, not hidden. Gradient/optimizer/RNG/persistent state unchanged.

Both joint arms10F/5B/1Adam; CW_LSO8F/1B/1Adam. No GC/GP/GDP/SCNP/SE/RL, new model,
memory bank, labels/domain decisions or output fusion. Disabled constraint remains
bitwise R46; disabled joint remains bitwise CW_LSO on qualified CPU fixtures.
Actual finite-step loss, Dice, clinical benefit and anatomy are not guaranteed.

## Fixed matrix and resources

Five conditions C/W/CW_LSO/CW_LSO_JS_SHARED/CW_LSO_JS_SHARED_DESCENT (repaired
source in this independent round); primary DESCENT. Three original seeds
20260907/17011/29009, two original orders,1951 arrivals per trajectory:
30 fresh complete trajectories,58,530 independent CPU-scored rows. Every arm
starts from the original source checkpoint; do not borrow24 completed R47 controls
or promote the64-image resource prefixes. No scientific failure retry.

Only physicalGPU0/1,max2 workers, sharing allowed; no intervention in other tasks.
Require all R47 and isolated repair-profile processes retired and previous verified
delivery. Separate original R48 T0 before first GPU work;22h online/24h wall/
48GPU-worker h/32GiB, including initialization, same-round profiles and failures.
Never reset this T0/budget or rewrite earlier ledgers. All five conditions need a
same-round12-arrival PASS profile and whole30-job conservative admission. No
profile pruning. Formal worker uses matching PASS peak, free>=max(1.2*peak,
peak+512MiB); missing/invalid refuses. Cold profile5GiB guard unchanged.

Previous12 closed/stopped experimental rounds cost98.81037320819168 GPU-worker h,
including R39/R47 failures. Three isolated profiles total668.9714405536652s;
pre-R48 aggregate98.99619860834548h. The latest isolated repaired profile had8 PASS,
512 arrivals,256 repaired-condition updates,254 negative proposals,0 final
violations,16 commit repairs6.25% and2 native non-descent fallbacks. First repairs
matched visits16/42. Timing overhead versus SHARED+0.075% to+2.289%, allocated
816.875MiB unchanged, reserved896→898MiB. These are bounded numerical/resource
observations, not population fallback rates or segmentation results. They do not
replace this round's matching profiles/admission or guarantee all future steps.

## Frozen evaluation and diagnosis

Primary SEARCH versus C/W each>=0.3pp; versus CW_LSO/R46 SHARED each>=0.1pp.
All four comparisons require both orders positive,>=5/6 positive trajectories,
imageweighted>=0,worst seed-mean domain/channel/order cell>=-2pp. No primary
replacement or relaxed threshold. Complete negative results remain delivered.
Report paired content-cluster bootstrap intervals, all domain/channel/order/seed
cells and REFUGE_Valid OD/OC preservation versus Drishti_GS/ORIGA costs. Independent
CPU masks/scoring only after every online worker retires. Existing SEARCH/REVIEW
were historically exposed; REVIEW descriptive only, never tuning or independent
generalization. Patient linkage remains unknown; ASSD/Brier/soft Dice uncomputed.

Aggregate constraint triggers, native and commit-repair fallback counts/reasons,
proposal/returned-filter/candidate-committed/final derivatives, gap/tolerance,
unconstrained derivative(proposal+gap), multiplier, rank/K spectra/capture/energy/
relative update/norms, OD/OC evidence, BN and regional gradients, calls and costs.
K spectra describe the unconstrained filter; pre-repair norms describe its
candidate and are not always the final committed norm. Failures persist numerical
fields and the pre-call visit in physical journals and receipts; no unavailable
derivative is fabricated. Mechanism association/proxy descent is not anatomy proof.

## Monitoring and actual-completion delivery

Necessary30-minute checks only: RUN_STATE/attempts/processes/progress/RESOURCE_LEDGER
and neutral ps/nvidia-smi. Stay quiet during healthy progress. Notify completion,
failure, budget stop, delivery blockage or required choice. Do not restart healthy
queues, repeat launch, retry science, tune or start R49. Retain partial failed
outputs and all costs; no automatic technical recovery or tolerance relaxation.

At actual termination verify all online/scoring/watch/supervisor retired. If
complete, verify30 unique complete jobs and30 CPU scores x1951=58,530; if stopped,
explicitly distinguish incomplete/unscored cells from zeros and scientific failure.
Deliver own source/protocol plus anonymous all-positive/all-negative aggregates,
audits, costs and report to reports/r48-repaired-shared-descent on
experiment/r48-repaired-shared-descent-v1 in DLwbm123/DPA-CTTA. Every GitHub action
uses explicit local proxy, no bypass/direct fallback, force push or overwrite.
Verify remote SHA and anonymous report HTTP before local/remote DELIVERY and
CAMPAIGN updates. No private images/labels/per-content predictions or scores,
patient/content identifiers including hashes, model states, private paths/logs
or third-party PDFs. Delete the monitor after verified terminal delivery regardless
of scientific gate; delivery blockage keeps only necessary delivery retries.
