# Implementation contract

## Online boundary

- `DPAOnlineAdapter.step(image)` accepts no label, mask, ground truth, metric, or future sample.
- It runs the frozen source model at `z=0` to obtain descriptor feature/logits.
- It performs soft atlas association and the closed-form update under `torch.no_grad()`.
- It changes only the `z_prev` buffer. It creates no optimizer and invokes no differentiation API.
- The source base, task head, and batch-normalization parameters/buffers remain frozen.
- `online.py` does not import the offline package.

## Offline boundary

- `optimize_oracle_state` requires an explicit `source_label` argument and optimizes only a cloned latent state.
- Offline losses are pure tensor functions.
- `scripts/m0_oracle.py` is dry-run by default, rejects a real-data path without explicit authorization, and still stops at `BLOCKED_PENDING_CODE_REVIEW_PASS` in this version.

## Data and device boundary for this release

- No real source/target image is read.
- No source checkpoint is loaded.
- No model, image, feature, prediction, or logits artifact is saved.
- Discovery and tests use synthetic CPU tensors with `CUDA_VISIBLE_DEVICES=""`.
- The external checkout must match the pinned commit for real-model integration.

## Serialization

Standard PyTorch `state_dict` includes the FiLM basis, synthetic anchors, scaler buffers, and online `z_prev`. Tests cover round-trip replay. External source weights are frozen but remain part of the wrapper state dict for ordinary PyTorch compatibility.
