# R36 incremental modules: background startup verified

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
component behavior and EMA snapshot recovery. At the first healthy startup check, C, W and W_TP real GPU profiles passed; LSO/LSM/LS profiles were active. The remainder of profiling and full-matrix admission continue in the detached queue. Preparation failures retained: NFS copystat errno5 (copyfile/readback
passed), incomplete-deployment test import failure (1.995 CPU seconds), and system
Python torch._C import failure (reused the existing verified runtime). No GPU work was started by those preparation failures; unmeasured preparation time is NA, not zero. Initial watch import then failed because inherited R9/R10 planning JSON dependencies were omitted. These were restored, full runtime import passed, and the failed watch entry alone was recovered before any GPU profile. Original T0 and budget were retained; no healthy process was restarted. PREPARATION_NOTES.json retains all five failures.

Budget: 24h wall, online22h, 48 GPU-worker hours, 32GiB output. Profiles count from
original launch T0. Entire matrix must be admitted before formal workers; no pruning,
scientific retries, tuning or budget resets. SEARCH and legacy REVIEW are historically
exposed; neither supports independent clinical generalization.

[Protocol](../../docs/protocols/R36_INCREMENTAL_MODULES.md)

Original T0: 2026-10-09T22:18:48.317183+08:00. Scientific code: `0eda9a1b235216b9648435e6458c88fda8ca07d6`. Runtime status at the first healthy check: PROFILING, 3/7 profiles COMPLETE, three active GPU profile workers, zero failed GPU receipts. All visible watch/supervisor/worker Python commands were neutral; ps and nvidia-smi were checked. Three CPU geometry/state checks plus two host checks passed, and complete runtime import passed. An hourly heartbeat was created and verified ACTIVE in the app. Profiles passing and queued formal work do not mean all 42 trajectories have begun or finished. Execution completion and GitHub result delivery are
separate. Publish only own code/protocol and anonymous aggregates; private data,
predictions, weights, logs, identifiers and third-party PDFs stay private.

Latest startup verification: **RUNNING_FIXED_MATRIX**, all 7/7 real GPU profiles COMPLETE, full 42-job matrix admitted. Three formal workers advanced to visits [65, 61, 64]; 0 formal trajectories completed and no GPU failure receipts at this check. Projected formal GPU-worker cost 36.786/48h, three concurrent workers. Estimated execution plus scoring roughly 12–14h, subject to I/O and actual load; this is not a completion result.
