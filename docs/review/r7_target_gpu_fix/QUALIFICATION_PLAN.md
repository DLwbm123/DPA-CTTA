# Finite qualification, distinct from TARGET_SCREEN

User request: resolve the supplied review issues and start experiments immediately. This authorizes necessary bounded GPU qualification and read-only deployment loading here; it does not manufacture an external TARGET PASS. No real target or source pixels, source retraining, protocol changes or monitoring restart.

One worker on physical GPU 6, RTX3090. CPU preprocessing/method/FP64 state; FP32 CUDA backbone, no AMP/TF32, deterministic cuBLAS. Random original full ResUNet, eight arms, three procedural 512x512 visits each: 63 GPU forwards, 3 backward, 3 Adam. Three fresh STATIC comparisons add 6 forwards; repeated C0 adds 2. One fresh C_BASE GPU comparison and one CPU comparison add 16 forwards, 2 backward/Adam: total 87 forwards (79 GPU + 8 CPU), 5 backward/Adam. No VJP, AdamW or source training. Six procedural package serialization/reloads. Fixed CPU/GPU comparison tolerance rtol=.003/atol=.0003; same-backend repeat exact. Upper bound 100 forwards, 5 backward/Adam, wall 600 s, output 64 MiB. Failure stops qualification; fixes use a new candidate SHA, preserved failure log.

Separate read-only real roundtrip: six fresh original checkpoints/segmenters and six load_artifact calls, independent inventory from the supplied review, original contexts unmodified, backend identical to source. Zero image reads, forwards, backward, VJP or optimizer calls; six checkpoint and six artifact deserializations; wall 300 s, output 16 MiB. Verify group/mode/effective tensor/context/method identities using existing trusted loader. Publish aggregate hashes and counts only, no tensors or private per-content ledger.

Affected CPU acceptance: target shell including failure-cost regressions plus existing R7 math/context and source boundary checks on merged code. No historical 166 suite and no SOURCE_PREP rerun.

TARGET remains default-disabled; exact final code/source/backend/registration/resource/output binding and external review required before real 24-job dispatch. One-worker serial proposal preserves frozen job list; three-worker assignment remains optional and unapproved.
