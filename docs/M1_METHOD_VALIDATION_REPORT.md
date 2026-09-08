# M1 adaptation-oriented DD validation

Five-arm source and target results: **NOT_RUN at this implementation commit**. No performance conclusion is available. This report will be updated from actual execution evidence; implementation and registration do not establish method effectiveness.

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

Five new small CPU tests passed for image gradients/isolation, empty/existing Adam state and zero moments, double-precision optimizer reference, deterministic registration negatives, and stopped GM reference gradients. An additional aggregation test passed for unequal domain sizes and ASSD adverse-tail direction. The full supported regression run is recorded separately when complete; this document does not predeclare its result. GPU smoke and real training have not run at this commit.

After execution, an independent CPU process validates registry order, all scoring channels and lifecycle events, exact training sequences and counts, final artifact bindings, and completion/source-state checks. It produces public aggregate and execution audit JSON without sample IDs or private paths. Missing tasks/domains or an incomplete method cannot be described as a complete M1 comparison.
