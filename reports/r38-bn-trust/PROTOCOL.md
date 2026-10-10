# R38: one fixed BN displacement trust cap

User authorized autonomous evidence-driven non-RL incremental improvements on
2026-10-09. R37 completed and its full positive/negative aggregates were delivered
before R38. Freeze this entire round before any execution or target-label scoring.

## Development hypothesis

R37 CW_LSO SEARCH gained0.354066pp over C,0.955978pp over W and0.191136pp over CW,
all6/6 trajectories positive. Complete gate failed because worst seed-mean cell
versus W was-5.155135pp (order0 REFUGE_Valid OC); OD also lost4.908927pp and
order1 REFUGE OC lost3.528396pp. These cells are preserved, not excused by means.

SEARCH-linked model diagnostics: mean BN displacement CW_LSO0.003228 versus
W0.001402, mean LR1e-4 versus3.695513e-5. The C/G difference also includes entropy,
perturbation and dynamic learning rate; association does not prove step-size causality.
Hypothesis: a post-Adam parameter displacement cap reduces large-update drift while
retaining CW_LSO's positive mean. Single radius0.0015 is chosen from these exposed
development diagnostics, not REVIEW; no radius or coefficient scan is authorized.
This is an own simple trust module, not a reproduction or Bayesian-optimality claim.

## Frozen matrix and method

Exactly C, W, CW_LSO, CW_LSO_TR; three seeds [20260907,17011,29009], both original
orders [0,1], full1,951 arrivals each:24 trajectories/46,824 formal arrivals.
Fresh original checkpoint/BN/Adam/RNG state each job. C/W/CW_LSO are exactly R37
definitions. Primary CW_LSO_TR adds only a global BN-affine displacement cap0.0015.

For each ordinary C-host Adam step, save the BN-affine parameter vector immediately
before Adam. After Adam, compute delta and its global L2 norm. If norm>0.0015,
write theta_before+delta*(0.0015/norm); otherwise leave the step untouched.
Keep the ordinary Adam moment update and step counter, gradients, learning rate,
weak views, current detached target, W weights and LSO coefficient0.1 unchanged.
The native final forward occurs after projection. No image query or additional
backward, optimizer step, adaptive radius, memory, teacher, output fusion or RL.
Adam moments describe the unprojected gradient; this is projected Adam, not an
exact smaller-learning-rate replica. Only adaptive BN affine parameters may change.

No pruning, promotion, retuning, extra seeds or scientific retry. Profile all four
12 arrivals and admit the whole24-trajectory queue using existing timing20% reserve.
Three GPU workers maximum, one per GPU after memory/storage checks. Radius zero
must recover exact retained output, full state and RNG.

## Scoring and gate

All online workers retire before separate CPU mask reads. Same historically exposed
SEARCH1017/legacy REVIEW678, full coverage, hard Dice2TP/(prediction+GT), both-empty1,
equal domains×OD/OC then seeds/orders. Report means, all cells/negative cells,
imageweighted score, empty predictions, containment, content-cluster bootstrap2000
and trust-cap activation/scales/raw displacements. Patient linkage UNKNOWN.
No REVIEW-guided choices, online GT, new evaluation set or independent-generalization
claim. ASSD/Brier/soft Dice not calculated in this bounded round.

Primary succeeds only if SEARCH CW_LSO_TR exceeds both C and W by>=0.3pp; each
comparison has both orders positive,>=5/6 trajectories positive, imageweighted
gain>=0 and worst seed-mean domain/channel/order cell>=-2pp. Additionally preserve
CW_LSO mean within0.05pp (TR-minus-CW_LSO>=-0.05pp). This is a robustness module:
the CW_LSO comparison is mean noninferiority, not required positive on every order.
Report all of its negative cells anyway. The original C/W gates are not relaxed.
Frozen primary only; no changing radius or threshold based on this round's scores.

## Budget, verification and delivery

Tests: exact disabled-cap output/full-state parity, unchanged controls, actual BN
delta bound with forced activation, finite post-cap forward, frozen parameters and
snapshot-next-step recovery. Reuse unchanged LSO geometry and public redaction
checks. Original per-round T0 before profiles;22h online,24h total,48 GPU-worker h,
32GiB. Count all profile/formal/failed attempts. Preserve R36/R37 T0/ledgers;
cumulative prior GPU-worker cost22.140055h, plus this round's actual attempts.
Record preparation CPU tests separately. No healthy worker restart or budget reset.

Hourly monitoring only as authorized. At actual completion, retain complete positive
and negative anonymous aggregates, own code/protocol/report/audits, push DPA-CTTA
branch experiment/r38-bn-trust-v1 using explicit local proxy, verify remote commit
and anonymous report before updating DELIVERY. No private content/patient hashes,
images, labels, predictions, weights, model states, paths/logs or third-party PDFs.
Use neutral visible entrypoints/arguments and prescribed mounted storage. Failed
gate permits only a new evidence-driven pre-registered hypothesis, never blind reruns.
