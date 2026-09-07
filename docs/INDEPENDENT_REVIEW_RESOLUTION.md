# Independent review resolution — minimal host core v1

**IMPLEMENTED_FOR_SECOND_REVIEW — NO REAL-DATA TRAINING STARTED**

Baseline: `DLwbm123/DPA-CTTA@54911f9e1aff4cc0338dfac0264e67456024d503`.
Branch: `review/minimal-host-core-v1`. This report applies to its containing Git
commit; it does not certify the historical publication or authorize experiments.
The original publication worktree/main, the earlier original implementation, old
reports, and fixed CTTA source checkout were preserved. CI: **NOT_CONFIGURED**.

## Findings resolved or retained

| Finding | Disposition | Actual implementation/evidence |
|---|---|---|
| F1: rank-two sine projection | FIXED, schema changed | `FrozenDescriptor` uses a private seeded CPU Generator and QR, validates legal row dimension, finite full row rank and PP^T; global RNG unchanged. `descriptor_schema_version=2` is stored/required in descriptor/scaler states, including strict=False loads. Config deserialization requires explicit schema 2; new `configs/method_v1.json`. No old checkpoint/cache silently accepted. |
| F2: Polyp effective modulation rank | HISTORICAL LIMITATION RETAINED | Three one-channel `ra4_conv5`, `ra3_conv4`, `ra2_conv4` output hooks form score-map FiLM with a fixed-basis control rank at most six, not a demonstrated 16-D spatial manifold. Fundus `seg_head` also modulates output scores. Historical injection configs unchanged; no new search/M0. |
| F3: premature online state commit | FIXED | `DPAOnlineAdapter.step` predicts using a local candidate, checks finite logits, then commits z_prev. Injected final-forward failure/nonfinite output leaves previous state bitwise unchanged and raises explicitly. No label-based rollback, silent no-op or nan_to_num. |
| F4: proximal input/shape validation | FIXED | Exact nonempty shape/dtype/device agreement for previous/precision/rhs; finite positive scalar lambda/floor; finite nonnegative normalized weights with matching batch/dtype/device; finite input/result checks. NaN state/rhs/center, broadcast rhs, wrong dtype/device and invalid weights are rejected. Normal first-order/direct-solve regression remains. |
| F5: oracle basis gradient pollution | FIXED | `optimize_oracle_state` uses `autograd.grad(loss,state)` and the state-only optimizer. Two calls preserve existing non-state .grad values/None and source/FiLM weights exactly. Its return remains detached, intentionally not a DD inner update. |
| F6: K+2 Atlas forwards | DOCUMENTED, NOT OPTIMIZED | One current zero pass + K complete anchor zero passes + final adapted pass on every image. K=8 remains ten forwards, not a measured tenfold latency. No cache or offline/online cache split added to the non-mainline Atlas. |
| F7: pixel/model-input contract | IMPLEMENTED IN NEW HOST, EXPLICIT EXCEPTION | Target/proxy preprocessing is shared and declared: Fundus per-image min-max; Polyp ImageNet normalization. Already-resized pixel_rgb is separate from model_input. No resize or new style code. Native Polyp FFT uses normalized input; it is preserved for exact native equivalence rather than silently changed to raw pixels. Old Atlas still has no task loader. |
| F8: native VPTTA host missing | MINIMAL IMAGE-STEP IMPLEMENTED/VERIFIED | Imports real native model, AdaBN, Prompt and Memory and compiles the unchanged pinned VPTTA.run image block. A positive proxy term modifies one loss assignment and uses the original optimizer update. Four-image full-model zero-weight sequence checks and enabled lifecycle/gradient checks run for both tasks. No source_only model substitutes for native host verification. |
| F9: executing source not constrained by HEAD | FIXED FOR CRITICAL IMPORT SCOPE | The dependency bridge checks HEAD, compares actual tracked critical Python bytes using git archive (even when index flags hide worktree changes), rejects untracked source in those scopes and foreign module origins, and isolates native absolute-import aliases. It rejects preexisting conflicting user modules instead of clearing them or editing any checkout. See initialization scope below. |
| F10: trainer/audit gaps | AUDIT ENTRY FIXED; RESEARCH GAPS RETAINED | Historical pretraining_audit.py now exits with DEPRECATED_AUDIT_ENTRYPOINT and cannot emit a new PASS. Old reports unchanged. New runner derives test count/failure/error/skip/exit from results and records unrun/unknown scope explicitly. No DD trainer, real-data loop or source pilot configuration implemented. |

Concrete regression source: [test_review_regressions.py](../tests/test_review_regressions.py).
New mainline contracts: [HOST_CONTRACT.md](HOST_CONTRACT.md),
[MEDICAL_LOSS_CONTRACT.md](MEDICAL_LOSS_CONTRACT.md).

## Supported-environment replay of the original review

The supplied [blob-pinned probe](../audit/review_probes_54911f9.py) was replayed
**on the unchanged baseline worktree**, not on patched files. Its pin was not
bypassed. Output: [review_baseline_probe_replay.json](../audit/review_baseline_probe_replay.json)
and [log](../audit/review_baseline_probe_replay.log).
The [reviewer's original results](../audit/independent_review_probe_results_54911f9.json)
are retained separately. The supplied original review attachment was read and
left unchanged; the F1–F10 table above supplies its public resolution.

Actual replay environment is Python 3.12.9 / PyTorch 2.6.0, matching the publication
runtime. The original probe contains a **hard-coded** `same_environment_as_publication=false`;
that literal was preserved with the unmodified script and is not used as a fresh
runtime judgment. Read the actual `environment` object instead.

The defects reproduced: projection rank two; final-forward failure changing state
by L2 0.3432372808456421; invalid rhs/previous and broadcast accepted; NaN center
polluting state; K=8 causing ten source forwards; all six basis gradients doubling
on the second oracle call. Valid proximal solve discrepancy was only
2.220446049250313e-16, with the contraction check passing; this was not treated as
a defect or a new numerical gate. Descriptor history difference remained zero.

The initial direct Downloads invocation failed before probes because an unrelated
`code.py` shadowed the Python standard-library `code` module. Replaying with
Python's `-P` safe-path flag resolved it without editing that file or upgrading
Python. The failed attempt generated no probe certification; its traceback is
not published because it contained machine-local paths.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$PYTHON" -P \
  "$REVIEW_ROOT/audit/review_probes_54911f9.py" \
  --repo "$UNCHANGED_BASELINE_WORKTREE" \
  --output "$REVIEW_ROOT/audit/review_baseline_probe_replay.json"
```

## Current runner and actual checks

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
DPA_CTTA_BASE_ROOT="$REFERENCE" "$PYTHON" -P audit/run_review_fix_checks.py
```

Run from this branch's root. `$REFERENCE` is the clean CTTA pin above; `$PYTHON`
is the existing supported environment. Nothing was installed or upgraded.

Final observed run: **158 tests, 0 failures, 0 errors, 0 skips; exit 0**. Python 3.12.9, PyTorch 2.6.0; elapsed 38.845 seconds. CUDA initialized: false; blocked image/checkpoint/download/network/CUDA API attempts: zero. These are runner observations, not independent scientific signoff.

The observed result is in [review_fix_results.json](../audit/review_fix_results.json)
and [the aggregate CPU log](../audit/review_fix_cpu.log), generated directly by
[run_review_fix_checks.py](../audit/run_review_fix_checks.py).
The runner blocks image opens, Torch checkpoint/download APIs, CUDA initialization
and in-process socket connections. It also executes the earlier full-model focused
checks without rewriting their historical audit files. CLI subprocess tests use
only inspected dry-run/rejection paths and local synthetic Git fixtures.

No fixed test-count target or performance gate was used. Regression coverage
includes exact state failure handling, oracle gradient ownership, descriptor
schema/rank, dirty/foreign dependency rejection, per-image medical hand references,
boundary direction/sign rejection, empty/full-mask gradients, native zero-weight
sequence equivalence, proxy gradient sensitivity, and full-model state isolation.
The later safe-path runner invocation also checks that the documented `-P`
reproduction command can import its audit helpers.

## Initialization and scope limitations

Dependency validation occurs at model/host construction, not every source forward.
It covers the suite loader/data/metrics modules and the task's native network,
utils, control-flow and preprocessing Python files. Imported module.__file__ must
match the expected module path. The import bridge is for sequential initialization
in a dedicated experiment interpreter; it does not provide concurrent namespace
loading or detect arbitrary runtime monkey-patching. It does not reset/clean
worktrees, refresh permissions, or evict preexisting foreign modules.

The host reuses the original VPTTA **image block**, excluding its loaders/evaluator
and filesystem constructor. Thus checkpoint-backed or full CLI/stream behavior
is NOT_VERIFIED, not a mock PASS. All native checks here use randomly initialized
full ResUNet34/PraNet at task resolution; the native-neighbor test fixture uses 2
in both arms so four images exercise retrieval, while the default remains 16.

Auxiliary isolation currently costs one full frozen source clone. No measured
online latency/memory benchmark or efficient deployment claim is made. Real RGB
resize, source split/provenance ingestion, distance generation, coreset selection,
DD, target-style transfer and all actual experiment configurations are NOT_IMPLEMENTED.
The native Polyp FFT input conflict is explicit in the host contract.

Historical Atlas remains a control only: anchor anatomy uses trainable synthetic
masks while target anatomy uses source predictions; free charts cannot demonstrate
DD necessity. `distillation_loss` is still a four-term tensor combiner, and its
reference_mask=None fallback is not endorsed for unmatched source/synthetic layouts.
No full DD trainer exists. The old hard-max worst-class loss is separate from the
new per-image balanced region/boundary objective. No accuracy, anti-collapse,
privacy, clinical, source-proxy usefulness or SOTA claim follows from these tests.

## Delivery and next action

Only `review/minimal-host-core-v1` is eligible for push. No main merge, force push,
new GitHub workflow, token scope change, server login or background job.
Public payload: source/config changes, generated procedural tests, these contracts,
supplied review evidence, CPU logs/results. No image/mask/weight tensor files or
private paths/credentials are included.

The next step is independent review of the fixed commit. A real source pilot,
M0/M1/DD or target run still requires separate approval of a concrete commit and
configuration. This implementation round does not grant that approval.
