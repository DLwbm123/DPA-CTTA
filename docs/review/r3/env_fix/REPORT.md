# R3-ENV-01 environment fix

**R3_ENV_FIX_READY_FOR_REVIEW** — external difference review pending; no execution authorization is granted here.

Implementation: `d601496a0827af3e1fe728612a4b9f17a613955d`. Reviewed code baseline: `6d8fc7317506400b039d3d2ed41bca18523beb27`. Branch: `experiment/r3-five-region-frameworks-v1`. The code commit is followed by a separate evidence publication commit.

## Change and entry points

R3 now forces the smoke workspace to `:4096:8` in a copied child environment before `Popen`, and removes the variable for formal children. Parent environment values and other launch fields are preserved. Unknown phases reject. After the existing exact-identity authorization, the worker checks this contract before checkpoint loading, backend/device queries or model work. Direct `RUN_MODE` entry cannot silently bypass it. CPU calls to `smoke(..., device='cpu')` require no workspace variable.

| Scope | Source at implementation SHA |
|---|---|
| Pure child environment and phase assertion | [src/dpa_ctta/r3/execution.py:33](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/r3/execution.py#L33), `:42` |
| Actual policy observation (no device query) | [src/dpa_ctta/r3/execution.py:48](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/r3/execution.py#L48) |
| Smoke comparison context evidence | [src/dpa_ctta/r3/execution.py:71](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/r3/execution.py#L71), `:94` |
| Authorized worker entry / formal evidence | [src/dpa_ctta/r3/execution.py:137](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/r3/execution.py#L137), `:158` |
| Environment fixed before spawn | [src/dpa_ctta/r3/execution.py:190](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/r3/execution.py#L190) |
| Six new environment tests | [tests/test_r3_execution.py:52](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/tests/test_r3_execution.py#L52) through `:100` |
| Existing 38-update real-model CPU test extended | [tests/test_r3.py:126](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/tests/test_r3.py#L126) |
| Test-only EIO/ENOSPC diagnostics | [tests/test_r1_fixes.py:229](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/tests/test_r1_fixes.py#L229), `:301` |

[ENV_FIX.patch](ENV_FIX.patch) is the complete code/test difference from the reviewed implementation, scoped to `src scripts tests configs`. Those trees were identical at the original implementation and its documentation publication commit `98e18ae4b8b8320a53943dce98950e131f9178ad`; excluding old publication documents makes the requested patch narrow. Four files change: one R3 runtime file and three tests. The shared supervisor, IO/NFS/evidence implementations and R1/R2 runtime remain unchanged.

## Before / after workspace matrix

“Absent” means the key is absent, including removal of an empty or invalid inherited value. Before values follow the reviewed `launch.start` source's unconditional environment copy; after values are asserted by the eight subcases of `test_phase_environment_matrix_preserves_parent_and_identity`.

| Parent value | Before smoke | After smoke | Before formal | After formal |
|---|---|---|---|---|
| absent | absent | `:4096:8` | absent | absent |
| `invalid` | `invalid` | `:4096:8` | `invalid` | absent |
| `:16:8` | `:16:8` | `:4096:8` | `:16:8` | absent |
| `:4096:8` | `:4096:8` | `:4096:8` | `:4096:8` | absent |

The test also verifies parent immutability and preservation of CUDA visibility, RUN_MODE, worker/job/packet identity, entry point, Python path and an unrelated variable. Direct-worker rejection additionally covers empty strings. No subprocess GPU launch or real device query was needed to test the pure builder.

## Actual backend evidence

`smoke.completion.json.backend` remains the pre-comparison `environment()` evidence, augmented with the actual `warn_only` and `cublas_workspace_config`. The separate `paired_comparison_backend` is sampled inside `deterministic_smoke_pair()`: strict algorithms enabled, warn-only disabled, and the actual environment value. It is never represented by the pre-context switches.

The real-model CPU test deliberately begins with algorithms disabled and warn-only enabled, with workspace absent. It observes strict/false/absent inside the comparison, then verifies both switches and the environment restore exactly. A failure-path test verifies restoration when an injected exception occurs before model work. A formal worker test uses a mocked device backend and observes the actual caller flags without changing them; its workspace evidence is null because the key is absent. Production launch does not add a deterministic algorithm policy to formal execution.

CPU evidence proves these branches and observations, not CUDA numerical acceptance. Smoke tolerances, ready paths, 38 Adam/loss backward calls, 316 forwards and the VJP ceiling of 24 are unchanged.

## Real CPU verification and first failure

| Run | Result | Duration | Evidence |
|---|---|---:|---|
| First targeted | 9 tests, 1 failure (test argument index) | 2.46 s | [log](logs/targeted-01.log), [JSON](logs/targeted-01.json) |
| Corrected targeted | 9/9 passed | 2.42 s | [log](logs/targeted-02.log), [JSON](logs/targeted-02.json) |
| Normal full CPU regression (one run) | **104/104 passed**, no failures/errors/skips | 253.28 s | [complete log](logs/cpu-full-01.log), [JSON](logs/cpu-full-01.json) |

The [actual full-model traces](FULL_MODEL_CPU_TRACES.json) and [extracted backend/counter evidence](CPU_BACKEND_EVIDENCE.json) record **316 forwards, 38 Adam calls, 38 loss backward calls and 18 VJPs**, with 5 actual parameter replacements. The existing checker reports CUDA uninitialized and zero real RGB/mask/registered-checkpoint/source reads. Existing IO regressions use only freshly generated procedural files. The main CPU process command was inspected and used the neutral environment-driven entry.

Six new distinct tests cover the parent matrix, unknown phase, direct-worker mismatch, disabled authorization priority, formal evidence/policy preservation, and smoke failure restoration. The existing full-model test gains backend/restoration assertions without another model call. Existing process test assertions remain intact. The historical 98 distinct tests plus these six are 104; targeted runs overlap the full suite and must not be added to that total.

The first targeted run's sole failure was a new test indexing `trajectory` mock argument 6 (checkpoint IO) instead of argument 5 (backend). The observed value was `{'bytes': 7}`. Only the assertion index was corrected; [targeted-01.log](logs/targeted-01.log) and its JSON preserve the failure. No production change followed that failure, and there is no automatic retry. The corrected targeted run and normal full run have their own files. [REPRODUCE.md](REPRODUCE.md) gives the CPU-only recipe and exact de-identification policy.

## EIO residual

**本次未重现，根因未知。** The normal full suite's persistent-EIO case returned all four owned children and found all four already reaped before fallback, with the unrelated control still alive. Both formal fixture children returned -15; both smoke fixture children returned 0. All cleanup calls completed without a recorded cleanup exception. The injected EIO still caused supervisor subprocess exit 1, as required; this is expected test evidence, not a unittest failure. Fallback subsequently waited for all owned children and its own control. ENOSPC ownership assertions also passed. No repeated EIO diagnostic run was added.

The original CPU05 observation remains in [historical development evidence](../DEVELOPMENT_LOG.md). Neither its nine subsequent passing diagnostics nor a later passing regression establishes a cause. The shared supervisor is not changed. The test-only observer forwards each existing `stop_owned` call, retaining cleanup exceptions and returncodes, pre-fallback owned PID/phase/key and `waitpid` results, control PID liveness, final fallback `wait` completion and the captured subprocess traceback. It adds no dispatch, retry, sleep, or relaxed assertion. These procedural process IDs identify only short-lived CPU fixtures.

## Frozen scope and evidence limits

- Science SHA256: `73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e`.
- Base registration digest: `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`.
- Secondary recurrence stream digest: `cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`.

[FROZEN_INPUTS.json](FROZEN_INPUTS.json) records an empty Git diff for the science, disabled execution config and plan against the reviewed code; published science/registration digests, 85-job JSON/CSV and secondary summary are unchanged against their original publication commit. The requested science SHA was recomputed. Private registration and real identity sequences were not read or recomputed this round; their unchanged digest declarations are inherited from the frozen public evidence.

All five frameworks, controls, thresholds, state lifecycles, methods, four primary streams and secondary construction are unchanged. The optional zero-edge iteration bookkeeping change was skipped to keep this repair to ENV-01. Its previously disclosed nonblocking accounting limitation therefore remains.

The plan stays 17 arms × 5 streams = 85 complete trajectories; formal budgets remain 165,835 scoring/Adam/loss-backward calls, 1,385,210 forwards and at most 234,120 VJPs. No real trajectory was run. This round used no GPU, registered checkpoint, true target RGB/mask, source data/proxy/prototype, source retraining, formal experiment, background task or automation. Public files contain code, procedural evidence and supplied review documents only.

## Review provenance and stop

The [supplied external report](input/R3_REVIEW_REPORT_6d8fc731.md) requests changes for ENV-01; the [supplied fix prompt](input/R3_CODEX_ENV_FIX_PROMPT.md) defines this repair. Their statements are external review inputs, not a new approval authored by this implementation task. The original [method provenance](../METHOD_PROVENANCE.md), [state lifecycle](../STATE_LIFECYCLE.md), [85-job dry-run](../DRY_RUN_MATRIX.json) and [stream summary](../STREAM_SUMMARY.json) remain the scientific references.

Stop at **R3_ENV_FIX_READY_FOR_REVIEW** after publication. External review and any later resource authorization are separate pending steps.
