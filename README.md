# DPA-CTTA

**Current release branch: `experiment/source-pilot-run-v1`.** See the [release contract](docs/SOURCE_PILOT_RELEASE_CONTRACT.md) and [local regression report](docs/SOURCE_PILOT_RELEASE_LOCAL_REPORT.md). The earlier preparation status below is historical; actual registered execution is tracked separately.

**Current branch: `experiment/source-pilot-io-v1`**, based on `c92b64b2284eb5da0631c00d15cc44af773f4b2d`.
**SOURCE_PILOT_PREPARED — AWAITING EXACT COMMIT/CONFIG REVIEW**.

Read the [I/O contract](docs/SOURCE_PILOT_IO_CONTRACT.md),
[prepare report and changed-file table](docs/SOURCE_PILOT_PREPARE_REPORT.md),
[disabled draft config](configs/source_pilot_v0.json), [runner](src/dpa_ctta/source_pilot.py),
and [machine audit](audit/source_pilot_prepare_audit.json).
Only local CPU/procedural checks ran. Real source/checkpoint/device registration and pilot are NOT_RUN.
All real `run` entries reject `PILOT_COMMIT_CONFIG_APPROVAL_REQUIRED`. CI is NOT_CONFIGURED.

```bash
PYTHONPATH=src "$PYTHON" -P -m dpa_ctta.source_pilot prepare
```

## Historical minimal-core review (c92b64b)

The following section and old audit files describe the previous scope and are retained as history.

**Second-review branch: `review/minimal-host-core-v1`.**
Baseline: `54911f9e1aff4cc0338dfac0264e67456024d503`.
Status: **IMPLEMENTED_FOR_SECOND_REVIEW — NO REAL-DATA TRAINING STARTED**.

Start with [F1–F10 resolution](docs/INDEPENDENT_REVIEW_RESOLUTION.md),
[host contract](docs/HOST_CONTRACT.md), [medical loss contract](docs/MEDICAL_LOSS_CONTRACT.md),
and [observed CPU results](audit/review_fix_results.json).
The new host supports `base` and fixed-proxy `proxy_rehearsal` on synthetic CPU inputs.
It imports the pinned native VPTTA model/Prompt/AdaBN/Memory and image-step code;
no dataset, checkpoint, or DD training pipeline is included. CI: **NOT_CONFIGURED**.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
DPA_CTTA_BASE_ROOT="$REFERENCE" "$PYTHON" -P audit/run_review_fix_checks.py
```

Use the existing supported Python/PyTorch environment and the pinned CTTA checkout.
Descriptor/scaler state schema is now **2**, with `configs/method_v1.json`;
old descriptor artifacts/configs must be rebuilt, not silently loaded.
The old Atlas remains a research control, with documented score-map/rank/forward-cost limitations.
All real source pilots, M0/M1/DD, target evaluation and server jobs remain unrun.

## Historical first-publication overview (54911f9)

The text and audit links below describe the preserved first-publication snapshot;
the unchanged-source claims apply to that earlier commit, not this review-fix branch.

**Publication for independent review, 2026-09-07: CODE_REVIEW_PENDING.**
This snapshot preserves the original Atlas scientific code from
`617f67464df37d6e49972c4932a85f56d210ccab`. It does not implement the revised
research direction and does not authorize any experiment.

Start with the [review index](docs/CODE_REVIEW_INDEX.md),
[publication audit](docs/PUBLICATION_AUDIT.md),
[research decision](docs/RESEARCH_DECISION_20260907.md), and
[supplied revised plan](docs/REVISED_EXPERIMENT_PLAN.md).
The older pre-training audit is historical implementation evidence, not
independent signoff for this publication commit. CI: **NOT_CONFIGURED**;
`ci_templates/unit-tests.yml.disabled` is an inactive historical template.
For the CPU publication check, use the existing environment and the pinned
reference checkout with `python audit/run_publication_checks.py` (see audit).

Status: implementation-only research prototype.
No real-data training or target evaluation has been run.

Distilled Potential Atlas for Continual Test-Time Adaptation (DPA-CTTA) represents adaptation by a 16-dimensional state shared across three decoder/tail scales. A frozen descriptor associates the current sample with synthetic potential anchors, and a diagonal proximal objective updates the state in closed form. Online inference has no label input, optimizer, or backward pass.

## What is implemented

- Exact-zero, three-scale latent FiLM over a frozen segmentation model.
- Synthetic image/mask anchors with descriptor-derived association, local charts, and strictly positive diagonal precision.
- Opponent-color, Fourier, frozen-feature, and differentiable anatomy descriptors.
- Closed-form continual update with a reported contraction bound.
- Source-only oracle and distillation loss APIs, guarded behind the offline package.
- CPU synthetic tests and optional pinned ResUNet34/PraNet integration tests.

## Quick check

```bash
python -m pip install -e .
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
python scripts/dry_run.py
python scripts/m0_oracle.py
```

The final command is dry-run only. Real-data M0 execution is blocked until a separate GPT Pro review returns `CODE_REVIEW_PASS`.

## Pinned CTTA integration

The optional integration uses `ctta_suite.models.build_model` and `model_logits` from public repository `DLwbm123/CTTA` at commit `dbff0d985c6c95345d9fb78f5b1daef57b392564`. It neither copies upstream model code nor loads checkpoints.

```bash
export DPA_CTTA_BASE_ROOT=/path/to/pinned/DLwbm123-CTTA-checkout
CUDA_VISIBLE_DEVICES="" python scripts/discover_injection_points.py
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

If the checkout commit differs, external integration and every future real-data entry point refuse to run. Core synthetic tests remain available.

See [method](docs/METHOD.md), [implementation contract](docs/IMPLEMENTATION_CONTRACT.md), [injection contract](docs/INJECTION_CONTRACT.md), and [pre-training audit](docs/PRETRAINING_AUDIT.md).

## License

Apache-2.0. The external CTTA checkout retains its own licensing and is not vendored here.
