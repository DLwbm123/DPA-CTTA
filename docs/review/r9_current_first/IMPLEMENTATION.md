# R9 Current First — implementation review

## Status and authority

This is an implementation/review candidate based on the unchanged Screen24 report commit
`a9d3a43959da73ef552401342cc087d5ae8e6ba2`. It is **not an executed R9 experiment**.
The imported prompt explicitly separates implementation from real-data/checkpoint/GPU
execution. `LAUNCH.disabled.json` keeps `execution_authorized=false`. No R9 private
images, masks, checkpoints, or GPUs were opened; no monitor was created.

The separate `r9_current_first` namespace imports frozen R8 mathematical primitives,
data/split auditors, native baselines and atomic target journals. It does not modify
R8 files or reinterpret SELF as a fix to the original experiment.

## Implemented paths

- 43 source jobs; 610 core slots plus at most 161 distinct final-16k sensitivity
  slots. `TASK_GRAPH.json` expands dependencies without constructing a model.
- Original LEGACY fit kernel and complete pre-cal step-4000 continuation; receipt-chain,
  optimizer/RNG/state checks; explicitly recorded fresh-16k fallback only when the
  full continuation snapshot is unavailable. Identity corruption does not enable fallback.
- SELF current-query observation/prediction and same-query A clean reference;
  SELF_TASK removes fit auxiliary losses; fixed 16k schedule and original TBPTT/static rules.
- Independent 256-step calibration at four checkpoints (MLP has no calibratable
  temperature); pre/post calibration aligned and legacy validation; balanced hard-Dice
  checkpoint/recipe selection; confirmation seeds depend on frozen recipe selection.
- Same-weight RESET, PRECAL with recomputed B step size, PRED_ONLY, ISTA20 and
  output-only residual alpha. Separate learned STATIC weights remain separate.
- Six gradient arms with pinned six-view C0 teacher/strong augmentation, fresh Adam,
  scaled latent updates and BN-affine-only reset. Actual per-visit hooks verify
  8F/1B/1Adam or 10F/3B/3Adam on the synthetic network.
- Source-cal LR implementation supports the three proposed seed policies but defaults
  to **unresolved**. It cannot run until the user's choice is bound.
- Target image reader receives image-only metadata. Full source selection is locked
  before target execution/labels. Float32 probability journals are sealed before the
  separate CPU score process can read masks.
- Finite native R9 queue, per-attempt receipts, host-local locks, atomic phase snapshots,
  one evidenced infrastructure recovery per job, retained failed reservations, aggregate
  caps and soft per-job estimates. Numerical failures are retained and independent
  ready branches can finish; identity/isolation/resource failures stop the queue.
- Exact same-trajectory aliasing removes redundant final-16k evaluations. Historical
  baseline reuse requires the separate complete identity check; the default runner
  conservatively executes new baseline trajectories and does not infer reuse by score.
- Anonymous aggregate exporter includes hard/soft Dice, four-domain/two-channel means,
  source/checkpoint selection, per-seed/order groups, OC errors, common-valid ASSD,
  descriptive paired tails, and LONG10 first/last and per-cycle outputs. Public export
  is blocked until the finite queue ends. Missing scores stay missing.

## Approved storage/scoring amendment

The user explicitly approved: **internal per-trajectory scoring; release results only
at the end of the full round**. This resolves the conflict between exact soft Dice,
all-output retention and the 64 GiB limit. Maximum raw float32 predictions would occupy
about 4.28 TiB; one LONG10 trajectory occupies about 38.11 GiB.

The pipeline seals one complete trajectory, runs CPU scoring, validates the scalar
receipt, then retires only that R9 trajectory's temporary probabilities. It retains
prediction hashes, scalar/trace journals, snapshots, recovery evidence and receipts.
Source choices never consume target scores. The runtime selects no strategy based on
internally sealed target metrics. This amendment does not authorize a real launch.

The queue is deliberately serial: at most one GPU job or CPU score job is active.
This stays within the stated maximum of three GPU workers and bounds temporary
storage. It does not claim three-GPU throughput or a wall-time forecast. Parallel
execution would need concurrent storage/CPU resource profiling before implementation.

## Explicit implementation decisions for review

LR source-cal uses four representative visits (positions 0,8,16,24) of the unchanged
32-visit source curriculum, with 16 episodes per each of the four modes. Each visit
observes/predicts the same current styled query; labels enter the detached scoring
path. Cal split roles/anchors are reused, simulator keys are separately named
`R9_GRAD_LR_QUERY`, and the original augmentation-seed helper is retained. This gives
an explicit deterministic interpretation of the plan's 64 four-visit balanced episodes.
It does not silently reuse R8's three-pattern, cross-query, soft-Dice LR selector.
The exact schedule is reviewable in `gradient_calibration.py` before any execution.

## Validation and limits

Run from the repository root with the project Python containing PyTorch, NumPy,
SciPy and Pillow:

```sh
PYTHONPATH=src python -m unittest discover -s tests/r9 -v
PYTHONPATH=src python scripts/r9/manifest.py
python docs/review/r9_current_first/input/verify_plan.py
```

See `VALIDATION.json` for the observed synthetic results. These checks use generated
inputs, a small real-autograd network and temporary output directories. A synthetic
4000 boundary validates state restoration, not a claim to have trained 4000 real
steps. They do not establish CUDA/kernel determinism, medical-data utility, actual
ResUNet34 peak memory, complete private receipt availability, or a resource PASS.

Uncompleted admission work is intentionally visible:

1. User decision on LR source-seed aggregation (first two / first only / per-seed).
2. New private metadata/asset/snapshot bindings and freshly chosen GPU UUIDs/output root.
3. Authorized exact-code real source/kernel/scoring/IO profile, including fresh-fit
   fallback, validation/calibration, temporary disk and recovery costs.
4. Verified runtime inventory and complete node budgets; an independently authorized
   real launch. No blank profile or synthetic timing is accepted as measured proof.

`profile.units()` enumerates measurement units; `profile.projection()` conservatively
includes all 43 fresh-16k fits and one full failed reservation plus retry per unit.
It may exceed the proposed caps. It does not assert that the package fits, and does
not auto-increase caps, reduce work, or substitute synthetic timing. A tighter bounded
recovery-reservation scheme would require explicit implementation/review, not editing
an admission JSON to PASS.

## Runtime use after the missing bindings and authorization

`run.py` and `worker.py` are templates to copy to neutral entry filenames. Pass the
private launch file and neutral interpreter/worker paths in environment variables;
never put private identities/project names in process arguments. The queue verifies
exact runtime inventory and disabled/authorization/profile checks before private I/O.
A worker checks physical GPU UUID and measured VRAM headroom. The authorized operator
must perform the prescribed mount/free-space/write-read probe before a large remote
launch, then start the queue with the existing reliable background wrapper and check
full process command lines/GPU display. This review has not performed those actions.

No new hourly monitor is part of this implementation. The original R8 source files,
artifacts, costs and reports remain unchanged.
