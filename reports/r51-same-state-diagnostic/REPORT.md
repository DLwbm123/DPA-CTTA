# R51: fixed late-state counterfactual diagnostic — stopped before qualification

Status: **NOT_EVALUABLE_IMPORT_FAILURE**. Formal coverage: **0/96 pairs**.

The single GPU-assigned worker exited during Python imports, before source-model construction, CUDA policy initialization, anchor restoration, or any real-model forward/backward pass. The frozen 96-pair matrix was never admitted. This is a deployment failure, not a scientific negative result about joint sensitivity.

## Cause and scope

The deployed source artifact included Python files and the new protocol, but omitted a resource read by an existing transitive dependency at import time: `docs/review/r9_current_first/input/R9_SPEC_AND_MATRIX.json`. Importing the reused runtime therefore raised `FileNotFoundError`. Generated CPU tests exercised the diagnostic math and synthetic host, but did not exercise this deployment dependency chain. The execution source was `01c9f07f54dd8fc6e713b614bba7412c78ae68d4`. No post-failure source modification or second launch was made.

The planned diagnosis was restricted to two R50 CW_LSO visit1900 states, seed17011, orders0/1, and a shared 48-content historically exposed SEARCH panel. It did not authorize replay or early/middle-state claims. No inference, labels, or counterfactual effects were obtained in this attempt.

## Qualification and preparation evidence

- Four generated CPU checks passed in the final preparation run: native-step/restore parity and separated gradients; geometry matching and degeneracy; invalid-input rejection; and clustered-interval constant-effect behavior.
- One earlier generated test failed because a degenerate null took an unnecessary floating-point metric roundtrip. The pre-GPU implementation was corrected to preserve the original J vector exactly; matching tolerances were not changed. That failure and its cost remain recorded.
- Pre-GPU code review changed the no-update prediction to use the same evaluation RNG/buffers as every other branch. The resulting final CPU checks passed. No real-model measurements motivated the change.
- Earlier metadata/transport preparation failures are retained in `PREPARATION_FAILURES.json`.
- Real-model restoration qualification: **NOT REACHED**. Real-model gradient matching: **NOT REACHED**. Formal scoring: **NOT STARTED**.

## Resources and completion boundary

Charged GPU-worker wall time: **2.554734 seconds** (0.000709648 hours), including failed-worker imports, against the original **1800-second / 0.5-hour** cap. CUDA initialization was not reached; charged GPU-worker wall time is not GPU kernel execution time. Remaining budget was not used for a retry.

Recorded generated CPU-check time, including the failed check: **23.543173 seconds**. Other metadata/transport preparation CPU time was not measured. There were no formal masks, gradient-vector outputs, label reads or replay steps. The controller and worker exited, and a terminal process/GPU-PID check found no remaining owned processes.

No Dice effects, confidence intervals, or continuation decision can be estimated. The outcome does not add positive or negative scientific evidence to layer-one screening. All existing R49/R50 artifacts remain unchanged.

Any later attempt would require repairing the deployment resource bundle and adding an import-only preflight, followed by explicit authorization for a new attempt. This run does not auto-retry, relax a gate, tune, or expand samples.

Only own code/protocol and aggregate qualification, failure and cost evidence are public. Image/label data, content identities, private paths, raw logs, checkpoints and per-content artifacts are excluded.
