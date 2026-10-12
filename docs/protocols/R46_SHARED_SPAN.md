# R46: shared acquisition/anatomical sensitivity span

Frozen before same-round GPU profiles, online execution and scoring. User
authorized evaluating the CPU-qualified shared-span adjustment. Primary
CW_LSO_JS_SHARED; controls C/W/CW_LSO/original CW_LSO_JS. No automatic next round.

## Hypothesis and retained algorithm

R45 original JS had rank2 on every SEARCH step, but changed accuracy by-0.006898pp
versus CW_LSO and did not protect the worst W cell (-5.138166pp). Its anatomy
coverage and proposal overlap were not recorded; small metric-norm shrinkage
alone cannot identify failure cause. The new hypothesis is that anatomy-sensitive
directions outside the acquisition-only span need to be represented in the joint
filter. Capturing them may also suppress useful correction; no benefit guaranteed.

Use the CPU-qualified method in JOINT_SENSITIVITY_SHARED_SPAN_CPU.md. Original
X=H^-1/2 acquisition_VJPs,Y=H^-1/2 anatomical_VJPs and H=sqrt(v_hat)+1e-8.
Only the basis construction changes:

    Z=[X/||X||_F,Y/||Y||_F]; Q=orth(Z), rank<=4
    A_Q=(Q^T X)(Q^T X)^T/||X||_F^2
    S_Q=(Q^T Y)(Q^T Y)^T/||Y||_F^2
    R=(A_Q+0.01 I)^-1/2; K=(I+R S_Q R)^-1
    delta=H^-1/2 [H^1/2 delta_Adam+Q(K-I)Q^T H^1/2 delta_Adam]
    theta_new=theta_before+delta.

Strength1/ridge0.01; Gram cutoff max(1e-12,1e-6*largest_eigenvalue) on normalized
concatenation. Missing either sensitivity, rank0 or disabled keeps actual Adam
proposal exactly. Small matrices float64; moments/step count unchanged. Default
original JS remains bitwise CPU-equivalent to frozen R45. No parameter scan.

Same six weak views, W temperature0.1/boundary boost2, LSO0.1, LR1e-4, native Adam
betas(0.9,0.999)/epsilon1e-8/weight_decay0 and BN-affine-only update. Same detached
structure evidence, pooling4/smoothing3/Sobel orientation, boundary radius3,
variance reliability, nesting-valid mask, directional-magnitude cutoff2e-6,
displacement floor1e-3 and private Rademacher seed0. Fixed contrast1.10 and channel
logit shift(0.05,0,-0.05), logit clamp1e-6 with endpoints retained. Four auxiliary
VJPs,10F/5B/1Adam in both JS variants; CW_LSO8F/1B/1Adam. No GC/GP/GDP/SCNP/SE/RL,
new model, persistent bank, labels/domain actions, output fusion or optimizer step.

## Matrix and admission

Five fixed conditions C/W/CW_LSO/CW_LSO_JS/CW_LSO_JS_SHARED. Three original seeds
20260907/17011/29009, two original orders,1951 arrivals each:30 fresh trajectories,
58530 independent CPU-scored rows. Do not borrow old trajectories or promote
profiles. Same historically exposed SEARCH/REVIEW split, no new data access.

Only physical GPU0/1, maximum2 GPU workers, no intervention in other processes;
sharing is permitted and timing estimates can change. All five same-round
12-arrival profiles must PASS, then the entire30-job matrix must fit conservative
dual-worker admission. No pruning, scientific retry, retuning, seed expansion or
changing a healthy queue. Worker memory uses same-round condition PASS peak with
free>=max(1.2*peak,peak+512MiB); malformed/missing peak refuses. Cold profiles retain
the original5GiB guard. Neutral process names, paths and visible arguments.

Immutable T0 before any R46 GPU work:22h online/24h wall/48GPU-worker hours/32GiB,
all profiles and failed work charged. Old ledgers/T0 unchanged. Ten prior closed
rounds cost79.5537663931317 GPU-worker h (including unscorable R39); isolated JS
profile74.37265658378601s brings prior aggregate79.57442546440497h. Successful
shared-span local CPU qualification21 tests/10.718s, earlier20-test pass9.839s;
server preparation and failures recorded separately. No infinite technical retry.

## Frozen outcomes and report

Primary SEARCH versus C/W each>=0.3pp and versus CW_LSO/original JS each>=0.1pp.
All four comparisons require both orders positive,>=5/6 positive trajectories,
imageweighted>=0 and worst seed-mean domain/channel/order cell>=-2pp. No primary
substitution or relaxed gates; negatives retained regardless of success.

All online workers retire before independent CPU labels/scoring and REVIEW release.
Review is descriptive only: SEARCH/REVIEW historically exposed, no independent
generalization or clinical protection claim. Report all condition means, paired
intervals, domain/channel/order/seed deficits, attempts/costs. Report shared rank,
filter eigenvalues, acquisition/anatomy capture, proposal span energy, relative
metric update change, proposal/filtered norms, evidence OD/OC, BN and region
gradients. Contraction/capture does not prove anatomical correctness or causality.

On actual completion publish own source/protocol and anonymous aggregates/report
to reports/r46-shared-span on experiment/r46-shared-span-v1 in DLwbm123/DPA-CTTA.
Explicit local GitHub proxy required; verify remote SHA and anonymous report HTTP,
then local/remote DELIVERY and cumulative CAMPAIGN. No images/masks, content or
patient identifiers including hashes, per-content predictions/scores, model state,
private paths/logs or third-party PDFs. Background startup is not scientific
completion. No follow-up experiment or new periodic monitor is installed by this
launch; user can request monitoring separately.
