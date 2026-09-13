# R2 implementation report

Stage I: five new arms implemented; external review and Stage II execution remain separate. The complete source/test/config implementation commit is recorded in DELIVERY.json. No GPU experiment, real checkpoint load, target RGB/mask read, source asset access, offline update or background approval waiter was performed.

## Implemented scope

R2_E, R2_D, R2_DE, R2_F and R2_DE_S use the frozen equations, rank/support/refresh rules, lambda=0.05 and half-life=128. C and the original absolute+cumulative C_PCA_REGION remain available only as compatibility test paths; the formal matrix contains exactly the five new arms times four orders. D imports the original cumulative class directly. E imports the old absolute reconstruction function directly. F retains exponential shadow PCA cost/readiness but excludes its direction from the loss. DE-S preserves R1 quotas and local RNG salts and retains same-position feature pairing.

The R2 entry uses the reviewed R1 byte-verification/evaluator, neutral process invocation, process-group ownership cleanup, caps and atomic publication. It supplies the R2 schedule/budgets and complete scalar validation without changing original R1 constants. R2 scalar replay checks W/Q/raw counts/images/versions/refresh and matches historical controls by full run/code/science/registration/job/device identity. Truncation, mismatched completion and contradictory weights invalidate current success. Algorithm changes, evaluator changes and extra hyperparameters were not introduced.

## Frozen references

- Reviewed runtime: `54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b`.
- Worktree base / historical public results: `0411fdfba8c05b08b6f7172e8d002ad3cc738e38`. This descendant adds R1 delivery evidence; it does not alter the reused R1 runtime. The R2 narrow patch excludes those already-published R1 results.
- CTTA dependency: `dbff0d985c6c95345d9fb78f5b1daef57b392564`.
- GraTa dependency: `33ae20d664f305af34739ec54a5bec7da53ffa0b`.
- Original R1 science: `e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2`.
- R2 science: `882323714fc29b439bccb540cfe7e685733da1e7f0012af8a202954ec2e12353` (exact supplied proposal bytes).
- Registration: `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`; unchanged body, no registered asset rewriting.

The disabled execution config has enabled=false, empty GPU IDs, zero workers, background_allowed=false and no approval references. The user's GPU 6/7 preference is recorded for future reviewed Stage II; no executable authorization was generated.

## Verification and boundaries

Development checks: 14 new CPU tests passed locally with Python 3.12.9 / Torch 2.6.0. They include a real ResUNet34 structure with random weights and the complete seven-host, two-step mechanical smoke on CPU: 112 forwards, 14 backwards and 14 Adam calls, including five ready-memory new arms. This CPU smoke is not a substitute for Stage II's per-device smoke. All other input tensors, masks used by evaluator fixtures and scalar records are programmatically generated.

The final existing-environment CPU run is recorded in CPU_TEST_LOG.txt and CPU_TEST_RESULT.json, including all 37 unchanged R1 regressions plus 14 R2 checks. Test success is engineering evidence only. It is not an external review result or an effect gate. The guarded entry drops CHECKPOINT and rejects CUDA lazy initialization. Its real_target_reads field states the suite's procedural-only scope rather than a system-wide IO monitor.

Coverage includes explicit weighted sample references, empty visits, half-life, rho=1 equivalence, rank-zero/eigh failure, detached snapshots and gradients, projection identities, actual-rank compensation, region weighting, full-feature reduction, RNG isolation, paired shuffle matching, lambda-zero C equivalence, original R1 REGION compatibility, 8-forward merge timing, failure cleanup, historical binding, scalar replay and invalidation of incomplete results. Unchanged R1 tests cover ownership cleanup on ENOSPC/EIO, signal races, timeout and unrelated-process protection.

The provided independent NumPy plan algebra was also run; its results are separate from model implementation tests. Local Apple NumPy emitted matmul RuntimeWarnings while all reference assertions passed with finite outputs; the original plan script is also checked in the existing Linux environment, with the actual log preserved. No numerical formula was changed to suppress warnings.

## Dry-run and not-run list

Exactly 20 jobs, each 1,951 images: 39,020 new scores/backwards/Adam updates and 312,160 forwards. Historical C and REGION reuse 15,608 scalar records. Future per-device smoke adds 112 forwards and 14 backwards/Adam, so GPUs 6/7 with two workers would total 312,384 forwards and 39,048 backwards/Adam. Allocation rotates `(arm_index+order_index)%workers`. The fixed resource ceilings remain 2 hours/job, 24 hours wall, 24 active worker hours, 2 GiB output and at most three workers/two CPU threads per worker. No GPU time or memory measurement is claimed.

Not run: GPU initialization or queries, target image/mask decoding, real checkpoint loading, GPU smoke, formal trajectories, pairing actual scalar metric contents, source-data operations, new model training, effect selection or follow-on experiments. Historical paths and receipts/completion metadata only were checked in Stage I. Full old/new scalar pairing and unchanged verified-byte asset loading belong to Stage II.

No unresolved scientific choice was filled by observing target scores. Runtime authorization and GPU 6/7 availability at actual launch remain pending external review. The CPU analyzer can verify scalar recurrences and metadata, but cannot reconstruct feature covariance, basis directions or ASSD geometry from scalars. Four orders reuse exposed development contents and are not independent samples. Drishti_GS has 37 remaining_dev contents but receives one-quarter of the primary domain weight.

Public scope: new source, science/defaults, tests, dry-run matrix, deidentified reports and true CPU evidence. Excluded: private registration/paths, credentials, raw per-content records, patient/subject information, source data, target data, model weights, hardware identifiers and supplied third-party PDFs. No exclusion blocks this source/review delivery.

Final delivery status: R2_IMPLEMENTATION_READY_FOR_REVIEW. Stop here for external review; do not self-issue a review pass or run authorization.
