# R39: gradient-conflict selective BN trust cap

User authorized evidence-driven non-RL increments on2026-10-09. R36/R37/R38
completed with verified public deliveries before R39. Freeze this round before
GPU profiles, formal execution or target scoring. No changes to prior ledgers.

## Evidence and single hypothesis

R38 SEARCH cap-minus-C/W/CW_LSO was-0.935696/-0.333784/-1.288808pp, all0/6
positive. The fixed0.0015 cap reduced the worst W cell from-5.155135 to-1.477074pp,
but suppressed necessary adaptation: ORIGA OD/OC lost about6.00-7.33pp versus
CW_LSO and order0 Drishti OD/OC lost4.28/5.56pp. Cap activated97.7712% of steps,
mean scale0.496073. R38 retained CW_LSO still gained0.353112pp over C and
0.955024pp over W, while failing the same worst-cell guard. All negatives remain.

Hypothesis: retaining aligned adaptation while restricting steps whose current
BN gradient opposes recent BN gradients may retain mean gains and reduce drift.
R38 motivates selective activation; no benefit of this conflict signal is yet
measured. Conflict may instead signal a necessary distribution shift, so this
module can fail. No scored gradient-cosine threshold fitting or domain rule is
used. Own bounded heuristic, not a reproduction or proven Bayesian mechanism.

## Fixed method and matrix

Exactly C/W/CW_LSO/CW_LSO_GC, seeds[20260907,17011,29009], both original orders,
1,951 arrivals each:24 trajectories/46,824 formal rows. Fresh original model,
BN/Adam/RNG/gradient memory per job. Controls exactly match R38 definitions.
Primary CW_LSO_GC keeps CW_LSO loss, views, W parameters and LSO coefficient0.1.
Keep the R38 radius0.0015, but activate only when cosine(current gradient,
preceding raw-gradient EMA)<0. A zero norm gives cosine0. The first step has no
reference and is uncapped. No threshold/radius/EMA scan.

Capture the ordinary C-host pre-Adam BN-affine gradient vector g. Compute cosine
against preceding m, then set m=g on first step or m=0.9*m+0.1*g thereafter.
Ordinary Adam runs with unchanged gradients, moments, step counter and learning
rate. After Adam, if cosine<0 and global BN displacement>0.0015, project the
displacement to0.0015; otherwise leave it unchanged. Final native forward follows
projection. The EMA stores gradients, not patient IDs or examples; it is included
in snapshot/restore. No extra forward/backward/optimizer step, labels, domains,
source retraining, teacher, fusion or RL. No online feedback from scoring.

## Gate, diagnostics and cost

Same hard Dice and both-empty convention, equal domain/channel means, both
orders/three seeds, SEARCH1017; legacy REVIEW678 descriptive only. All workers
retire before separate CPU masks. Primary must exceed C/W each by>=0.3pp,
both orders positive,>=5/6 positive trajectories, imageweighted nonnegative,
worst seed-mean domain/channel/order cell>=-2pp. Also primary-minus-CW_LSO mean
must be>=-0.05pp. This extra comparison is mean noninferiority only; all its
negative cells remain public. Never relax C/W gates or replace primary later.
Both sets historically exposed; no independent generalization claim.

Report conflict/EMA-ready frequency, cosine distribution, cap activation/scale,
raw and actual BN displacement, all cells and coupled-content bootstrap2000.
Patient linkage UNKNOWN; no ASSD/Brier/soft Dice. Test unchanged controls,
disabled-cap and uncapped-first-step full-state/output parity, forced-conflict
bound, frozen parameters, finite output and EMA snapshot continuation. Reuse
existing strong-control/noninferiority and publication checks. Profile all four
12-arrival conditions and admit the whole matrix with20% timing reserve.

Three workers maximum, prescribed mounted storage, neutral entrypoints. Original
T0 before profiles;22h online/24h total/48 GPU-worker h/32GiB, all failures/profile
charged. Preparation CPU separately recorded. Prior closed cost29.205472535888354
GPU-worker h across3 rounds; retain original ledgers and cumulative cost.
No pruning, scientific retries, restarts of healthy workers, extra seeds or tuning.

## Delivery and continuation

Publish own code/protocol and all anonymous positive/negative aggregates, costs,
audits and report on experiment/r39-conflict-trust-v1 in DPA-CTTA. Every GitHub
operation uses explicit local proxy; verify remote commit and anonymous HTTP
before updating DELIVERY. No private data, content/patient IDs or hashes, paths,
model state, per-content predictions/scores/logs or third-party PDFs are public.
Continue authorized hourly necessary checks; notify meaningful changes only.
Success requires complete verified delivery before pausing monitoring. Failure
permits only a separately frozen evidence-driven hypothesis, never a blind rerun.
