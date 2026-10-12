# R45: joint acquisition/anatomical sensitivity on retained CW_LSO

Frozen before this round's GPU profiles, online execution and scoring. User
authorization: evaluate the already implemented and profiled joint-sensitivity
module. Primary CW_LSO_JS; controls C/W/CW_LSO. No automatic next round.

## Hypothesis and fixed method

Jointly sensitive BN update components may respond to imaging changes while
also disrupting reliable predicted structure. Filter only these components of
the actual CW_LSO Adam proposal, leaving its metric-orthogonal complement intact.
This is a testable hypothesis, not a guarantee of anatomical protection.

Use the exact CPU-qualified and real-model-profiled implementation documented in
JOINT_SENSITIVITY_CPU.md, without changing method parameters. Only existing BN
affine parameters adapt; original model/checkpoint, preprocessing, six weak views,
strong pseudo loss, W temperature0.1/boundary boost2, LSO0.1 and native C-host Adam
LR1e-4/betas(0.9,0.999)/epsilon1e-8/weight decay0 remain unchanged.

Estimate independent OD/OC structure evidence from detached six-view mean and
variance, radius3 boundary support pooled4, existing structure tensor and nesting
validity. Evidence = boundary*(1-4*pooled_variance)_+*normalized_directional_magnitude
*nesting_valid*[magnitude>2e-6], normalized by boundary support rather than total
evidence. Fields use pooling4, smoothing3 and doubled-angle Sobel orientation;
displacement denominator floor1e-3, private Rademacher seed0. No labels, domain
decisions, source images or shape atlas inform these estimates.

Two fixed acquisition probes before native normalization: sigmoid(1.10*logit(x))
and sigmoid(logit(x)+(0.05,0,-0.05)), logit clamp1e-6 and exact endpoint retention.
Original OD/OC sensor VJPs form B; each photo sum-channel VJP minus the original
sum-channel VJP forms a column of A. Four auxiliary VJPs do not accumulate Adam
gradients; probe RNG is restored before native strong augmentation.

Let delta_A be the one native Adam proposal and H=sqrt(v_hat)+1e-8 using its
post-step second moment. X=H^-1/2 A, Y=H^-1/2 B, Q=orth(X), rank<=2. Keep Gram
eigenvalues strictly above max(1e-12,1e-6*lambda_max). Define

    A_Q=(Q^T X)(Q^T X)^T/||X||_F^2
    S_Q=(Q^T Y)(Q^T Y)^T/||Y||_F^2
    R=(A_Q+0.01 I)^-1/2; K=(I+R S_Q R)^-1
    delta=H^-1/2 [H^1/2 delta_A+Q(K-I)Q^T H^1/2 delta_A]
    theta_new=theta_before+delta.

Strength1, ridge0.01; no scan. Zero anatomy norm gives S_Q=0; rank0 preserves the
actual Adam proposal exactly. Small matrices float64, final parameter dtype;
Adam moments/step count unchanged. Joint disabled delegates exactly to CW_LSO.
No GC/GP/GDP/SCNP/SE/RL, new persistent model/memory or additional optimizer step.
Per candidate image10 forwards/5 reverse traversals (4 VJPs+1 base backward)/1
Adam versus CW_LSO8/1/1. These are acquisition probes, not output fusion.

## Matrix, admission and costs

C/W/CW_LSO/CW_LSO_JS; seeds20260907,17011,29009; two original orders;1951 arrivals
each,24 fresh complete trajectories,46824 independent CPU-scored rows. Do not
borrow previous trajectories or promote profile predictions. Same historically
exposed SEARCH/REVIEW split; no new evaluation set. All online workers retire
before CPU mask scoring and REVIEW release. No scientific retries or retuning.

Only physical GPU0/1, maximum2 workers, no intervention in other tasks. R44 and
the prior isolated profile are already retired and delivered. The prior profile
measured JS/base latency1.44-1.50x and reserved peak896MiB, but is not admission for
this round: all four fresh12-arrival same-round profiles must PASS, then the full
24-job matrix must fit conservative fixed budget admission. No profile pruning.
Formal workers use same-round same-condition measured peak with free memory
>=max(1.2*peak,peak+512MiB); missing/abnormal profile fails closed. Cold profiles
retain the original5GiB guard. Neutral entry paths and arguments for all processes.

New immutable T0 before any R45 GPU work:22h online,24h wall,48GPU-worker hours,
32GiB. All R45 profiles/failures count. Keep old round ledgers unchanged. Prior
nine full rounds consumed71.52042149424554GPU-worker hours; the isolated JS profile
adds74.37265658378601GPU-worker seconds (CPU packaging failures also retained).
CPU preparation is separate and recorded. No budget reset or automatic new round.

## Frozen evaluation and delivery

Primary SEARCH versus C/W each>=0.3pp and versus CW_LSO>=0.1pp. Each comparison
requires both orders positive,>=5/6 trajectories positive,imageweighted>=0 and
worst seed-mean domain/channel/order cell>=-2pp. Keep all negative cells; no
primary substitution or relaxed gates. These are development investment gates.
REVIEW is descriptive only, not a tuning source or independent validation.

Report full paired/domain/channel/order/seed Dice aggregates, content-cluster
bootstrap intervals, negatives, all attempts/costs, structural evidence by channel,
joint rank/filter eigenvalues/proposal and filtered metric norms, BN and existing
region gradients. Evidence is a proxy; contraction is not proof of anatomy or
accuracy preservation. No clinical calibration or patient-level claim.

On actual completion publish own source/protocol and anonymous aggregate results,
costs and audit on experiment/r45-joint-sensitivity-v1. Explicit local GitHub proxy
required; verify remote commit and anonymous REPORT HTTP, then local/remote
DELIVERY and cumulative CAMPAIGN. Exclude content/patient identifiers including
hashes, images/masks, per-content scores/predictions, model state, private paths/
logs and third-party PDFs. Background launch is not scientific completion.
No new periodic monitor or follow-up experiment is authorized by this protocol;
existing user authorization for resource-safe background execution is retained.
