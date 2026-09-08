# M2 episode coverage comparison — implementation checkpoint

**NOT_RUN at this implementation commit.** No D2/O2 segmentation result is available yet. M1 remains a valid completed experiment with mixed results; improving index coverage does not prove improved adaptation.

## Fixed change and reused assets

| Component | M2 treatment |
| --- | --- |
| Joint episode triples and playback | New deterministic Appendix A quota construction; 600 unique triples per task |
| Query/state/transform marginals | Exact identity-wise M1 counts retained |
| Source checkpoints, original Real pixels and K4 masks/layout | Reused, no reselection or source training |
| Source/target query identities, exclusions and order | Exact M1 registration reused; 52 source and 224 target groups |
| 32-state Base history per task | Existing saved M1 files reused; zero reconstruction updates |
| Offline D/O, differentiable Adam, preprocessing, native host and proxy loss | Imported unchanged; file identities frozen in the M2 config |
| Training orchestration | Explicit seed reset for each method; same M1 per-episode math and save points, new D2/O2 names |
| Scoring and reconstruction | Only D2/O2 generate new records; old five arms retain M1 provenance |

The old M1 configuration, source files, branches and private artifacts are unchanged. The new branch starts at `2e6738e31708249902368333b882897c57280405`, with fixed dependency `dbff0d985c6c95345d9fb78f5b1daef57b392564`. Execution and later publication commits must be reported separately.

## Actual metadata coverage check

| Task | M1 unique triples | M2 unique triples | Query transform types, old → new | Query distinct states, old → new | Max triple repetition, old → new |
| --- | ---: | ---: | --- | --- | --- |
| Fundus | 160 | 600 | 1 → 4 | 4 → 12–15 | 4 → 1 |
| Polyp | 64 | 600 | 1 → 4 | 1 → 8–10 | 10 → 1 |

These observations are from the actual saved M1 episode registration. All query/state identities retain exactly their original counts: Fundus queries 15 each; Polyp queries 9/10 with the same identity assignment; state counts 18/19 with the same assignment; each transform 150. Every state now covers all four transforms with per-transform quotas differing by at most one. Full index lists and mapping remain private; public evidence contains distributions rather than source identifiers.

906 existing registered asset size/mtime checks passed without any change requiring a new content hash. Both original history files passed small-state format, moment, counter and memory checks. M1 did not save history-file digests: M2 anchors their current digests and format, without claiming a retrospective comparison to nonexistent original digests. Twenty old scoring JSONL files are available; their current identities were anchored for later paired reconstruction. Old final proxy saved-digest records and current sizes agree; those tensors are not used as M2 initialization. No asset selector was invoked, and the existing six ETIS exclusions remain.

## Finite execution contract

The entry `scripts/run_m2_episode_coverage.py` provides `smoke`, `run` and `recompute` stages with separate M2 receipt checks. Registration is a small overlay referencing M1; D2/O2 consume the same frozen task list. An example metadata preparation call is `m2_registration.register(old_m1_directory, new_private_directory)` with explicit existing paths, never a NAS search.

Each method starts from original Real RGB and empty image Adam, restores seed 20260907, and uses the original 32-state mapping. Training order is Fundus D2/O2, then Polyp D2/O2, 600 episodes each. All four final artifacts freeze before scoring. New evaluation is 104 source + 448 target = 552 records and online updates; expected new arms are D2/O2 only. N/A/R/D1/O1 are reused old evidence, not new GPU work. Fully available display coverage is 1,932 records, of which only 552 are new.

Single smoke budget: 8 online Adam, 4 image outer Adam and 2 differentiable inner computations. It uses original pixels, registered history index 16 and fixed procedural query/labels; temporary optimized proxies are discarded after matched M1/M2 host comparisons. Default total budget: 560 online Adam, 2,404 image outer Adam, 1,202 differentiable inner calculations. Six-hour GPU-stage wall limit, one authorized GPU with GPU7 preferred, 2 GiB new private outputs, owned 0700/0600. No background resource waiter, automatic resume or M3 is authorized.

## Verification at this checkpoint

The local selected test run passed 21 tests, exit 0, plus one additional seven-arm/interaction/missing-evidence test, exit 0. These cover sampler purity and exact quotas, frozen-list consumption, unchanged scientific callables, D2/O2-only record completeness, existing M1 image meta-gradients/Adam/label and pixel contracts, and scoring negatives. Deployment tests and GPU smoke are pending at this commit. The full historical 184-test M1 regression is reused evidence and was not rerun for M2.

## Interpretation boundary

The final report must show all seven columns and all seven domains, focusing on O2−D2, O2−O1, D2−D1 and the difference of method gaps, plus comparisons to N/A/R. Public aggregates will preserve paired medians, signs, tails and common ASSD cohorts. Missing old raw records will be explicitly NOT_AVAILABLE for affected pairing rather than reconstructed from rounded means.

Both crossed combinations and SGD playback order change. Their effects are not independently identified. Prior target exposure, UNKNOWN patient/video associations, single seed/order, fixed off-policy Base history and one-step optimization remain limitations. Negative or tiny gains do not trigger extra training or cancellation of the remaining fixed comparisons. Engineering completion is distinct from a useful method result.
