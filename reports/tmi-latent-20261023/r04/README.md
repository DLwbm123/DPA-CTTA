# Frozen latent channel coupling r04

Read PROTOCOL_ZH.md. Requires the read-only r01 implementation and both final source checkpoints, configured via private parent_root. Runs on GPU1 with WORK_ROOT through neutral supervisor/worker entries. Two matched source-fit metric arms, 500 episodes each per seed; final four-arm readout. No target data, comparator mechanisms, model-weight update, or cross-image prior update. Reports must use segmentation gates, not J descent. src/analyze.py aggregates existing private JSON without model calls.
