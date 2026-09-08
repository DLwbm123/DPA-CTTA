# M2 episode coverage comparison — stopped smoke report

Historical first attempt. The later user-authorized repair and full experiment are complete: [final experiment report](../results/m2_episode_coverage_v1/M2_EXPERIMENT_REPORT.md). The first-attempt evidence below is preserved; its original aggregates remain at commit c1af390.

**M2_PARTIAL.** Polyp D2 failed the fixed same-device logit comparison during the single GPU smoke. Formal training and scoring were not started. No D2/O2 segmentation result is available. M1 remains a valid completed experiment with mixed results; improving index coverage does not prove improved adaptation.

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

The local selected test run passed 21 tests, exit 0, plus one additional seven-arm/interaction/missing-evidence test, exit 0. These cover sampler purity and exact quotas, frozen-list consumption, unchanged scientific callables, D2/O2-only record completeness, existing M1 image meta-gradients/Adam/label and pixel contracts, and scoring negatives. The selected deployment run passed all 22 tests in 2.167 seconds (exit 0). The single GPU smoke exited 1 as detailed below. The full historical 184-test M1 regression is reused evidence and was not rerun for M2.

## Interpretation boundary

The final report must show all seven columns and all seven domains, focusing on O2−D2, O2−O1, D2−D1 and the difference of method gaps, plus comparisons to N/A/R. Public aggregates will preserve paired medians, signs, tails and common ASSD cohorts. Missing old raw records will be explicitly NOT_AVAILABLE for affected pairing rather than reconstructed from rounded means.

Both crossed combinations and SGD playback order change. Their effects are not independently identified. Prior target exposure, UNKNOWN patient/video associations, single seed/order, fixed off-policy Base history and one-step optimization remain limitations. Negative or tiny gains do not trigger extra training or cancellation of the remaining fixed comparisons. Engineering completion is distinct from a useful method result.

## Actual execution and stop

Execution commit: `8f038d8d3821145375adf4492d7fcc8842c3c808`. This report is in a later publication commit; its identity is available in Git history. The remote execution checkout remains at the execution commit. Environment comparison with M1 passed without differences (Python 3.10.6, Torch 2.2.1+cu121, CUDA 12.1, cuDNN 8902).

| Stage | Exit | Actual result |
| --- | --- | --- |
| Metadata registration and coverage | 0 | Both frozen lists passed; both histories reused |
| Selected deployment CPU tests | 0 | 22 passed; no failures, errors or skips |
| GPU smoke | 1 | Fundus D2/O2 passed; Polyp D2 logit comparison failed; Polyp O2 not run |
| Formal training | NOT_RUN | Fundus D2 0/600, O2 0/600; Polyp D2 0/600, O2 0/600 |
| Formal scoring | NOT_RUN | 0/104 source + 0/448 target; zero new records |
| Independent score recomputation | NOT_RUN | No new scoring JSONL exists to reconstruct |

The actual smoke prefix consumed **6 online Adam, 3 image outer Adam, 1 differentiable inner**. No history rebuild updates occurred. Fundus D2 and O2 each had exactly equal paired logits and passed prompt, Adam, counters, memory, RNG, source and gradient checks. Their image gradient norms were 74.89250946044922 and 0.3965001702308655. Polyp D2 completed its finite/nonzero image-gradient and finite changed-pixel checks and both online steps, but its outer scalar details were not persisted before the assertion. Do not treat the Polyp state/RNG/source checks after that assertion as passed.

Polyp D2: 2,299 of 123,904 logit elements (1.9%) failed `rtol=1e-4, atol=1e-5`. Greatest absolute difference: 0.0003592967987060547; greatest relative difference: 0.12090007960796356. Both online counters show 2 model forwards, 3 prompt forwards, 1 proxy forward of 4 images, 1 Adam update/backward/memory push, and zero retrievals. The failure is at the first `close(a,b)` in `m2_run.smoke`.

Read-only source tracing confirms that `m2_host` delegates directly to the unchanged M1 `make_host` with D2→D and O2→O, and the Polyp prompt initializes to ones. This does **not** establish the root cause of the observed numerical difference. No additional GPU diagnosis, automatic restart, seed change or tolerance relaxation was performed. The shared smoke prerequisite remains incomplete, so neither task entered formal training. M1 results remain valid; this failure is not evidence of M2 method ineffectiveness.

## Seven-arm display and unavailable comparisons

Dice means below are percent. N/A in the D2/O2 columns means unavailable, not a numerical zero. The old five columns reuse the already published M1 aggregate; they were not rescored or independently recomputed in this M2 closeout. The 1,380 old records are available and their 20 files were anchored, but no 1,932-record merged comparison exists. The accompanying public aggregate also retains separate source and OD/OC/channel means. Full historical distribution, paired and ASSD evidence remains in the unchanged [M1 aggregate](../results/m1_method_validation_v1/public_aggregate.json).

| Target task / domain | N | A | R | D1 | O1 | D2 | O2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus / REFUGE | 82.6467 | 84.7589 | 84.8029 | 84.9710 | 84.9565 | N/A | N/A |
| fundus / ORIGA | 56.7760 | 72.1003 | 72.0025 | 72.7252 | 72.1623 | N/A | N/A |
| fundus / REFUGE_Valid | 62.3375 | 74.7249 | 73.8143 | 72.1268 | 71.9181 | N/A | N/A |
| fundus / Drishti_GS | 74.6114 | 74.8411 | 75.6269 | 77.4187 | 77.6034 | N/A | N/A |
| polyp / CVC-ClinicDB | 78.4816 | 70.4921 | 70.2819 | 70.3216 | 70.3735 | N/A | N/A |
| polyp / ETIS-LaribPolypDB | 68.8296 | 81.5325 | 81.7458 | 81.5770 | 81.6458 | N/A | N/A |
| polyp / Kvasir-SEG | 88.0730 | 83.2893 | 83.2884 | 83.0521 | 83.4696 | N/A | N/A |

O2−D2, O2−O1, D2−D1, the interaction difference and all new-arm comparisons to N/A/R are **NOT_AVAILABLE in every domain and channel**. New paired medians, signs, tails and common ASSD cohorts cannot be estimated without new scores. Q1 (coverage improves D/O), Q2 (O2 adds clear benefit), and Q3 (both tasks or mixed) are all unanswered. The successful index coverage construction alone does not answer them.

## Resources and public delivery boundary

GPU 7 (RTX 3090) was used for the single foreground smoke; no M2 background formal job was launched. The persisted Fundus peak allocation is 7,806,945,792 bytes (7.27 GiB). Polyp peak and the failed-stage GPU timer were not saved. The environment-file to failure-file timestamp interval is 14.489 seconds, an approximate observed interval rather than a GPU wall-time measurement. Private job files total 807,077 bytes across 12 files, including the transferred source bundle; this excludes the execution checkout and all reused M1 data/models.

Published additions contain source, frozen configuration, tests, deidentified coverage, the failure prefix, resource/count audit and seven-column aggregate with explicit missing results. Private checkpoints, pixel/mask tensors, history contents, query identities, episode identity mappings, registration paths and raw scoring payloads are excluded. The execution code was not edited after the failed smoke. Stop status is **M2_PARTIAL**, with no automatic next round.
