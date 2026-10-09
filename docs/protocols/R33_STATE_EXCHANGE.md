# R33: matched-age parameter and Adam-state exchange

## Question and authority
Continue the user-authorized hourly, evidence-grounded diagnostic campaign. R32 is complete and publicly delivered; never rerun or rescore it. This round asks whether changing Adam memory, while holding a specified learned parameter state fixed, changes subsequent adaptation on the registered query tails. It does not assume harmful Adam memory is the cause. R32 HOLD differences already establish prediction-relevant parameter-history differences. R30 optimizer-only period32 resets harmed REFUGE_Valid; parameter resets helped that domain but sharply harmed ORIGA. Those periodic interventions changed trajectories and optimizer age and are not equivalent to the matched-age exchange here.

R32 SOURCE_UPDATE also lost on REFUGE_Valid, so foreign optimizer history is not necessary for loss. Parameter/optimizer compatibility is an explicit alternative explanation: off-diagonal state losses may reflect mismatched components rather than universally bad memory. No RL, selector training, objective change, LR/period scan or deployment claim is authorized by this diagnostic.

## Frozen assets and matrix
Retain the exact R32 source model, native C updater (BN affine only, Adam LR1e-4, betas0.9/0.999),512px preprocessing, image orders and roles. Seeds20260907,17011,29009; both original orders; query domains ORIGA and REFUGE_Valid. SAME64 is source plus the first64 images of the query domain; CROSS64 is source plus the last64 images of the immediately preceding domain. Both use64 updates, identical augmentation RNG draws at matched history positions, and Adam step64. Neither history overlaps query image contents; patient linkage remains UNKNOWN.

The four initial states are SS,SC,CS,CC: first letter selects BN affine parameters, second selects Adam m/v; S=SAME64,C=CROSS64. All Adam per-parameter clocks remain64. Copy the full registered Adam state/group metadata only after exact key, shape, parameter-group and step checks. Hold the parameter donor's other host fields fixed; model buffers and GraTa persistent state must agree across donors. No parameter interpolation, step reset, m-only/v-only split or extra history length.

All four states apply native C once per query, then retain it. Reuse R32 frozen SAME64_HOLD,CROSS64_HOLD,SOURCE_HOLD references instead of rerunning identical inference. Fixed performance references are C_CONT and ANCHOR from R30 on precisely the same queries. SS and CC must exactly reproduce R32 SAME64_UPDATE and CROSS64_UPDATE, including every pre- and post-update hard mask. Compare before scoring; any mismatch blocks the scientific readout.

Query tails start at within-domain image65, ORIGA586 and REFUGE_Valid736 images. Order0 CROSS histories are REFUGE→ORIGA and ORIGA→REFUGE_Valid; order1 histories are REFUGE_Valid→ORIGA and Drishti_GS→REFUGE_Valid. They are distinct ordered treatments. Never pool them into an unqualified cross-domain effect.

Matrix:3 seeds×2 orders×2 query domains×4 states =48 branches, in six worker jobs. Each job replays all1951 original C arrivals solely to obtain the exact query augmentation RNG schedule and verify R30 C masks, performs256 history updates, then5288 query updates. Formal totals:11706 replay arrivals,1536 warmup arrivals,31728 query arrivals,44970 Adam/backward calls,391488 forwards (one extra read-only pre-prediction per query). Diagonal pre/post comparisons cover15864 query arrivals each for pre and post; first-query optimizer-invariance checks total24. Approximately4.16GB of hard-mask output plus logs;64GiB engineering output cap.

## Controls and implementation checks
Reuse StateHost, its exact restore/clone machinery, R32 replay/warm/query/RNG isolation, and the existing process/receipt framework. State exchange cannot mutate donors. SS and CC must be identical to their original snapshots. SC shares SS parameters; CS shares CC parameters. The same-parameter pairs must have exactly identical first pre-update predictions. All future branches use the native replay's per-query RNG input and verify its output, so optimizer exchange cannot silently change augmentation draws.

Generated-input tests cover state ownership, clone isolation, diagonal identity, unchanged pre-predictions under Adam swaps, unchanged clocks, rejection of mismatched ages, and a complete synthetic CPU scoring run with known factorial contrasts. Existing pre-probe/native parity and history-disjointness tests are retained. A CPU supervisor must not perform GPU-binding policy checks: the worker does so after binding; parent admission checks may inspect UUID/free memory.

Each of three A100 profiles runs8 native replay checks, two64-update histories,8 calibration updates and four8-query branches. Expected profile176 Adam updates and1440 forwards. Profiles are engineering-only, do not score labels and do not use a short NATIVE state as a scientific comparison. SS/CC exact R32 full-tail checks occur in every formal job; profiles check state exchange and first pre-prediction identity.

The active storage must be NFS with64GiB free reserve and a successful small write/read probe. Check GPU0/1/2 availability with margin; sharing is permitted, modifying other jobs is not. Plain file writes avoid unsupported copystat. Use neutral process arguments and verify ps/nvidia. No cumulative GPU cap. Preserve this round's T0 and failed costs through any repair; retain R32 costs separately. Seven-day wall, six-day online and22h worker guards are engineering safeguards. Real profiles determine admission and ETA. Repair only engineering failures, never retry based on metric signs.

## Labels and estimands
Online sees images and opaque planned positions only. True domain identity constructs the offline interventions and stratifies offline scores, never drives native C. Online can read only registered source/image assets and historical reference bitmasks for replay checks. Do not read old or new per-content score files. All six successful workers must exit0 and retire before the CPU-only scorer releases labels. Context arrivals are processed but unscored.

Score nine conditions per eligible content: SS,SC,CS,CC,SAME64_HOLD,CROSS64_HOLD,SOURCE_HOLD,C_CONT,ANCHOR. Expected7206 eligible seed/order/query-content instances×9=64854 observations. This count is not independent patients. SEARCH is primary; legacy SEALED_REVIEW is already development-exposed and descriptive only.

Primary paired contrasts, separately by query domain and order:
- SC−SS: Adam C versus S at fixed S parameters.
- CC−CS: Adam C versus S at fixed C parameters.
- SS−CS: parameter S versus C at fixed S Adam.
- SC−CC: parameter S versus C at fixed C Adam.
- Interaction=(SC−SS)−(CC−CS), and diagonal SS−CC.

Report each state's update effect against its parameter-matched frozen HOLD, and every state against both C_CONT and ANCHOR plus source HOLD. Preserve ALL-tail, fixed first64-query and four positional quartile bins, OD and OC, individual seeds/orders and every negative cell. Auxiliary pre/post effects do not replace accumulated effects. A conditional Adam effect does not isolate m from v; off-diagonal results do not prove a universally beneficial donor. No winner selection, adaptive matrix expansion or post-hoc gate.

Query-tail findings are not full-stream efficacy. Same-content reuse across seeds/orders, duplicated frozen states, unknown patient/ROI provenance and exposed development roles prohibit independent confirmation or clinical claims. Publish protocol/code, all aggregate tables/negative cells, cost/operation receipts and an interpretation acknowledging parameter/optimizer coupling. Exclude images, labels, masks, weights, states, identities, private per-content scores and paths. Final delivery requires proxy-only GitHub push and verified remote commit plus anonymous report access.
