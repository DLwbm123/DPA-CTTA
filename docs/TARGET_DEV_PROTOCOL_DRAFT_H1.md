# Next-round target-dev protocol draft — NOT EXECUTED

This is a draft for a separately authorized task. No target files are enumerated or read for this document. Target domains, dataset/split registrations, image counts, evaluation authority and resource budget remain unspecified and must be fixed before any target-dev run. They must not be selected to favor the current source result.

Candidate arms are N (standard source BN), original A (native VPTTA), original B (native A + 0.1 × original K=4 source proxy region), and at most one diagnostic-supported H. The final H1 report names at most one candidate or explicitly states insufficient evidence. No H implementation or search is part of H1. Original A/B retain their names and settings; any changed normalization or learning rate requires a separately named H.

Use the same authorized source checkpoints, preprocessing, evaluation thresholds and declared source-proxy provenance for all relevant arms. Rebuild each arm from the same source state and seed and use identical registered stream order. Freeze the target-dev split and sequential/reset protocol before scoring; do not adapt using target labels. Labels belong only to a separate final-prediction evaluator. Keep an untouched final evaluation split for later independently authorized work; development scores are not final benchmark claims.

If H fixes source normalization statistics, explicitly define its adaptation objective: do not silently disable BN matching by making its loss constant or gradient-free. A small-learning-rate H, if supported instead, changes only that single predeclared learning rate. Proxy/query mismatch evidence does not itself authorize style transfer, DD, reselected K or a search over H variants.

Report per-domain/channel Dice, Fundus macro, paired A−N/B−A/H−A/H−B, empty/full counts, jointly defined ASSD differences and denominators, adaptation behavior and actual compute. Preserve every negative result and distinguish content groups from independent patients. Do not require a method to exceed N on source before entering genuinely shifted development data. Freeze all success/failure/coverage rules and the finite run budget in the next task; no score-triggered extension or automatic recovery.

Current status: DRAFT_ONLY. Execution authority, target asset identities, exact budget and the optional H choice are pending a future task; this document authorizes none of them.
