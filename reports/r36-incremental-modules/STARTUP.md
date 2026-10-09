# R36 incremental modules: prepared, not yet launched

The user authorized prioritized incremental modules, an own LSO/LSM implementation,
hourly monitoring and sustained background execution on 2026-10-09.

Seven fixed conditions: C, retained W06_LR15 (W), W_TP, W_LSO, W_LSM, W_LS, W_BAL.
Three original seeds and both original orders, 42 full trajectories of 1,951 arrivals.
No RL, source retraining, extra models, online labels or outcome-driven retuning.
TP is a simplified cross-threshold component score, not a PH/OT TopoOT reproduction;
LSO/LSM use independent nested sigmoid structure tensors; BAL is bounded BCE
contribution balancing inspired by DSBR, not the original entropy loss.

Five CPU mechanical tests passed in 6.431 seconds including disabled-module exact
host parity, finite/nonzero structure gradients, nested-channel symmetry, empty/stable
component behavior and EMA snapshot recovery. Real GPU profiles and matrix admission
are pending. Preparation failures retained: NFS copystat errno5 (copyfile/readback
passed), incomplete-deployment test import failure (1.995 CPU seconds), and system
Python torch._C import failure (reused the existing verified runtime). No GPU work was
started by those failures; unmeasured preparation time is NA, not zero.

Budget: 24h wall, online22h, 48 GPU-worker hours, 32GiB output. Profiles count from
original launch T0. Entire matrix must be admitted before formal workers; no pruning,
scientific retries, tuning or budget resets. SEARCH and legacy REVIEW are historically
exposed; neither supports independent clinical generalization.

[Protocol](../../docs/protocols/R36_INCREMENTAL_MODULES.md)

Status is preparation only. Runtime T0, launch receipt and active workers will be
recorded after verified startup. Execution completion and GitHub result delivery are
separate. Publish only own code/protocol and anonymous aggregates; private data,
predictions, weights, logs, identifiers and third-party PDFs stay private.
