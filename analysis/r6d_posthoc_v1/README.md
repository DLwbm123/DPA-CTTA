# R6-D scalar diagnostic

POST_HOC_EXPLORATORY. This tool does not launch a model or revise R6-A's NO_ADVANCE decision.

`core.py` implements the fixed scalar formulas, fixed C-anchor strata, registered timing and aggregate export. `run.py` snapshots the explicitly allowed historical JSON/JSONL, validates the snapshot, runs once in the foreground, and checks source bytes and pointers afterwards. `test_scalar.py` uses synthetic scalars only. The supplied package is archived without reformatting in `input/`; its manifest covers four files and is an artifact-integrity record, not a test result.

Set `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2`. The private `R6D_CONFIG` JSON supplies `code`, `source`, `work`, `analysis_sha` (and optional original control-directory metadata). `work` must be a new directory disjoint from source and code. Paths and private configuration are never published. Launch through a neutral Python stdin/runpy entry, with the module directory in `sys.path`; do not place private paths or method names in visible process arguments.

The `public/` directory is an intermediate analysis output, not a final scientific decision until REPORT, HYPOTHESIS_DECISION and DELIVERY have been written. `joined.private.jsonl`, input snapshots, original mappings and failure logs remain private. No automatic retries. A missing ledger requires aggregate-only recovery; a malformed available ledger is INCOMPLETE and must not be replaced with fabricated pairs.

## Audited validator call graph

No production module is imported. Only named function ASTs are compiled, with explicit scalar standard-library globals:

- `r1.plan`: registration digest, primary stream, binding/bound. `science()` is an adapter reading frozen configuration JSON; the original r1/r3 digests are explicitly checked.
- `r3.kernels.recurring_stream` and `r3.plan.stream/stream_summary`: metadata only; no tensor imports execute. Registered chunk schedule and contiguous segments are reconstructed unchanged.
- `p2_analysis.validate_metric` then `r5_update_acceptance.analyze.validate_metric`: exact original pixel-count, Dice, flags, ASSD-null and TP/FP/FN/TN checks.
- `r6_regional_consistency.plan`: stream/summary, fingerprint, matrix and allocation. These read code/config metadata only.
- `r6_regional_consistency.analyze`: read/lines, f32/near, audit_channel, replay, join, gate/gate_inputs, completed and validate_cross_GT. `completed` only reads snapshot JSON/JSONL, compares metadata and searches failure-marker names. Its `output_bytes` dependency is bound to the pre-snapshot metadata total of the original output directory, not a smaller subset snapshot. Original tolerances are unchanged.

`recompute`, `invalidate`, `publish` and all model/execution entrypoints are excluded. Model/native/GPU/process/network imports and source writes are blocked by a Python audit hook during the fixed run. This is defense in depth over inspected standard-library code, not an OS sandbox or a claim to instrument arbitrary native code. Frozen production fingerprint and science digest are verified before scalar replay.

## Table semantics

- CSV `mean`, median, quantiles and tails describe actual paired rows in that cell. They are not differences of marginal quantiles. Empty CSV numeric fields mean null.
- `signed_net`, `positive_mass`, `negative_mass` use the original four-domain weights. `macro_contribution` halves OD/OC contributions; `primary_contribution` additionally halves each primary stream. The `primary` combined rows already contain this order weighting. Recurrence is never added into the primary metric.
- `weighted_conditional_mean` normalizes only within the named cell and is never a replacement primary metric. Missing domains have zero contribution and are not silently reweighted.
- All partition and quartile cells are emitted, including empty/missing groups. Quartile boundaries are C-only, remaining-dev, stream/domain/channel specific and use linear quantiles with ties assigned downwards. Each variable is analyzed independently. Time halves use all registered arrivals before the scoring filter.
- Regional references in controls are hypothetical BAL weighting at that arm's own state. Actual C/SCALE foreground shares equal their plain shares; actual BAL shares use its region weights. Actual SHUFFLE foreground energy/BCE shares are unavailable because permuted partition-specific sums were not logged. Neither hard-error mass nor local logit cosine is a parameter-gradient or causal-forgetting measurement.
- Full table contributions are checked against domain, segment, chunk and stratum partitions. Identical content across streams is joined by its private content key, never by visit or sorted score.

Analysis is limited to one process, at most two configured threads, a 1800-second alarm, and 2 GiB new outputs; snapshot bytes are separately accounted. CPU/memory/I/O cost and rejected protective test attempts are distinct from the always-zero model-call budget.
