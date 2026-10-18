# Pre-inference packaging repair

The first worker exited1 during import, before entering its body, model loading, CUDA initialization, target image/label decoding or any forward/update. The source bundle omitted an existing `docs/review/r9_current_first/input/R9_SPEC_AND_MATRIX.json` dependency imported transitively by the reused runtime. The original directory, logs and process/cost receipts are retained.

Repair: include the existing tracked `docs` dependencies in the code bundle and extend the CPU self-check to import connected worker/scorer dependencies and the pinned augmentation module. Use a distinct repair directory. Charge the failed worker's entire process wall time against the original1200-second budget; do not grant a fresh GPU budget. Preserve preparation costs, including the earlier wrapper-only environment-merge error, separately.

This is a pre-inference engineering repair under the user's execution/fix authorization, not a scientific-negative retry. The original protocol, six states, feedback identities, H50, three branches and thresholds are unchanged. Its no-automatic-retry boundary continues to prohibit efficacy reruns or budget increases; this note records the packaging exception before any target computation.
