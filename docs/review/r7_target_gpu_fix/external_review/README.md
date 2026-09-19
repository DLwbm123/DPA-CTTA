# R7 component-scoped external review

See output/EXTERNAL_REVIEW.md and output/R7_EXTERNAL_REVIEW_RECORD.json first.

The archive contains an actual external review record, not a target launch approval.
GPU source code and submitted completion evidence are accepted within the stated scope.
TARGET_SCREEN requires the failure-cost fix and GPU-backend integration.
Real six-package serialized-loader evidence remains pending. No model/optimizer/GPU execution was performed by the reviewer.

Reproduce the three standard-library control-flow cases:

    python review_probes.py

These assert the current buggy behavior and success control; passing the probe script is not a production acceptance PASS.

Sources are pinned in output/SOURCES.json. The target excerpt is manually materialized from the connector's exact function text; the complete target file was not rehashed. Source arithmetic checks use a previously archived exact expected_counts function (unchanged in the GPU delta); observed output values were transcribed from the submitted report. No private model bytes are distributed.
