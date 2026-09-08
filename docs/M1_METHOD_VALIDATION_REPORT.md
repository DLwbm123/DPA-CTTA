# M1 adaptation-oriented DD validation

Five-arm source and target results: **PENDING — M1_RUNNING**. The formal process started successfully; at the one-time startup check, Fundus Base history completed 32 visits and Fundus GM-DD recorded 13 episodes. No scoring or scientific conclusion is available. This is a timestamped startup delivery, not a completed experiment.

Clean execution commit: `d92ed88e603eb68aea97a57493c96a084e98d91c`. Publication commit is the Git commit containing this report. The execution checkout remains frozen at the former commit. Branch: `experiment/m1-adaptation-dd-validation-v1`. Physical GPU 7, UUID `GPU-885f8cbc-d6bd-3ba9-f65d-a34373a93c0c`. The detached finite pipeline continues without SSH and stops on failure or its budget; no scheduled monitoring or automatic resume is installed.

## Frozen protocol and implementation

The M1 task supersedes H2's missing-role gate. Target rows receive a new exploratory development role, retain original CSV splits and historical evaluation exposure, and use UNKNOWN for unavailable patient/video associations. No untouched-final, patient-independence, clinical, SOTA or privacy claim is made. H2 remains NOT_EXECUTED; no H normalization modification is used.

Source assets and original split selection are retained. Registration verified 906 selected file identities against recorded digests and container geometry. Source training queries: Fundus 40, Polyp 64. Each task uses 32 Base history visits, the original four proxy masks and original 20/32 source validation representatives. All seven target domains selected 32 content groups. Six ETIS rows duplicate image content assigned to an earlier domain, reducing its eligible pool from 196 to 190 before selection. Private registration contains exact identities, exclusions, original order and the shared 600-episode sequence.

N is source standard BN; A is native VPTTA; R maps to original B; D/O share the same online R code and fixed mask/layout, differing only in frozen proxy pixels. A/R source requires_grad and gradient behavior are unchanged. D is prompt-space supervised gradient cosine matching; O minimizes source query region loss after one differentiable native Adam update. Neither uses target supervision for training. Offline source model parameters are frozen; differentiable pixels bypass the online FixedProxy detach path. Native AdaBN statistics and detach semantics remain intact.

The Adam primitive retains the original forward, moments, bias correction and epsilon. At exactly zero second moment, sqrt has an explicitly defined zero derivative, avoiding zero-times-infinity while preserving all forward values. The host-gradient component can be detached after computing it because it has no dependence on synthetic images; the proxy-to-prompt-gradient-to-Adam-to-query path is retained. The objectives use the same query, transform, initialization and immutable Base history distribution. This is a truncated off-policy one-step objective, not differentiation through a complete Ours trajectory or memory retrieval.

## Execution and reproducibility

Configuration: `configs/m1_adaptation_dd_validation_v1.json`. Independent entry: `scripts/run_m1_method_validation.py`, stages `smoke`, `run`, `recompute`, each using a private receipt. The registration function `dpa_ctta.m1_data.register(old_pilot_directory, target_manifest_mapping)` accepts the existing exact manifests; it does not scan NAS or widen the source whitelist. Receipt binds clean execution commit, fixed dependency, config, registration and GPU UUID; run/recompute additionally bind smoke digest and coverage. Final artifacts are all frozen before any source or target scoring.

Private state is saved at 0/100/300/600, with only step 600 evaluated. The one-shot run has no resume behavior. It checks a six-hour GPU-stage wall limit and 2 GiB output cap, preserves scalar failure prefixes, and saves a small partial image/optimizer state on budget exhaustion. Private outputs use 0700/0600 outside the checkout. Synthetic medical tensors and masks are not public artifacts.

Nominal budgets are 1,242 real online prompt Adam calls, 2,404 image outer Adam calls and 1,202 differentiable inner computations, separately counted. Source/target scoring has 260/1,120 records. Histories, training and smoke records are not scoring records. Full-K4 statistics, resolutions and backbones are unchanged. One authorized GPU is used, with GPU7 preferred; no other process is changed.

## Verification status at this commit

The supported full CPU regression passed **184 tests, 0 failures/errors/skips, exit 0**, in 393.417 seconds using local Python 3.12.9 / PyTorch 2.6.0. The deployment environment (Python 3.10.6 / PyTorch 2.2.1+cu121) passed all **7 M1-specific tests**, exit 0, including the two later checks for equal-domain aggregation and task-specific history retrieval. Five overlap the full regression; these counts must not be summed as unique tests.

The single GPU smoke passed both tasks in 38.794 GPU-stage wall seconds: **74 real online Adam calls, 4 image outer calls, 2 differentiable inner calculations**, exit 0. Both Base streams agreed across 17 visits including retrieval, initial R/D/O predictions and states agreed, and D/O produced finite nonzero synthetic-image gradients on procedural inputs. Observed peak allocated memory was 4,957,076,992 bytes for Fundus and 2,910,537,728 bytes for Polyp; these are smoke peaks, not predictions of final training cost. Native Adam epsilon was verified as 1e-8 with unchanged options. Detailed procedural evidence is in `results/m1_method_validation_v1/execution_audit.json`.

At startup check, no failure file existed, the detached process was alive and training JSONL records were being appended. Final source/target metrics, common ASSD cohorts, tails, training costs and scientific signals remain pending. The runner is scheduled internally to freeze all four step-600 artifacts, evaluate all five arms and independently reconstruct results in CPU-only mode, with no performance-triggered gate.

After execution, an independent CPU process validates registry order, all scoring channels and lifecycle events, exact training sequences and counts, final artifact bindings, and completion/source-state checks. It produces public aggregate and execution audit JSON without sample IDs or private paths. Missing tasks/domains or an incomplete method cannot be described as a complete M1 comparison.
