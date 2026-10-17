# R66 directional boundary residual: design, not a registered experiment

2026-10-17. Incorporates the user's request to read the shared Pro analysis. This document is a bounded source-code/evidence review. No R66 model inference, label access, training, candidate scoring or checkpoint replay has occurred. R65 is already complete and delivered; no RCA continuation is launched.

## Crucial prior work omitted from the proposed framing

R16 already implements image-conditioned boundary repair: `r16_evidence_correction/methods.py::seeds/prototypes/Engine.prepare` uses frozen up3 features, confident background/disc-rim/cup seeds, class prototypes, and a six-pixel editable boundary band. Its image-conditioned residual head consumes logits, flip difference, entropy, signed distances, up3 features, RGB and gradients. D_CONTEXT and D_LOGIT each used2000 source training steps and9026 parameters. This was historically authorized R16 work, not current permission to repeat source training.

The [completed R16 report](results/r16_evidence_correction/REPORT.md) records P-minus-DS +0.104620pp but source feature shuffling did not support spatial correspondence as its benefit source. D_CONTEXT-minus-D_LOGIT was only +0.012961pp; all new directions failed their frozen +0.5pp versus C0 gate. These are that round's static-source/DS comparisons, not increments over today's CW_LSO. Their numerical results must not be spliced into R52.

R21 uses historical class prototypes and connected-disagreement-region soft corrections. R22 uses mutual nearest neighbors in frozen up3 features and pseudo-label votes from earlier images. A new global prototype, source-feature selector, or generic residual head would overlap substantially. The earlier unimplemented source-reference nearest-neighbor idea is not the active proposal.

## What remains distinct, and unproven

The plausible distinction is a class-local **signed normal displacement**, estimated from the current image's local boundary neighborhoods. It must generate both inward and outward corrections rather than choose an entire historical candidate or protect current structure. Current predictions define seed regions but do not certify them as correct. Locality alone is not novelty or an efficacy argument.

A minimal candidate specification for mechanical development (not a data-run registration): reuse frozen source up3 at its native128x128 resolution; represent the two candidate boundaries on that grid; estimate outward normals from signed distance, whose sign is positive outside. At each boundary location sample two local sides at normal offsets[-4,-3,-2] and[2,3,4], each with tangential offsets[-1,0,1]. Admit only pixels whose candidate class matches the appropriate side; for OC restrict the outer side to disc-minus-cup, for OD use background. Require at least4 valid samples on each side. Normalize feature vectors; compute local side means and compare cosine affinities along normal offsets[-1,0,1]. Accept a displacement only for a unique inside-to-outside affinity crossing with nonzero normal and nondegenerate side means; otherwise abstain. Interpolate that crossing within one native feature pixel (four original pixels). This fixed local support is tied to representation resolution; no radius/layer/margin grid. Collision handling, interpolation and mask reconstruction still need an explicit deterministic implementation and synthetic tests before registration.

Mandatory generated fixtures: known inward/outward boundary errors with informative image features, identical/constant features, missing rim/side seeds, empty/full masks, corners, multiple crossings, duplicate boundary edits and disconnected components. Correct behavior includes abstention when there is no directional evidence. A synthetic pass establishes mechanics only.

## Decision sequence before a real run

1. Inventory existing checkpoint metadata and registration only: which early/middle/late states actually exist and are restorable? Current R61 qualification covers two visit1900 anchors only. Do not claim temporal coverage or silently replay a whole stream to manufacture anchors. Freeze any required replay separately with full costs.
2. Freeze one offline residual-direction diagnostic with image-only generation, retirement then target-label scoring. Keep the current candidate, a matched spatially shuffled-feature control, and a matched-magnitude randomized-sign control. Preserve per-class inward/outward counts, abstention denominators and all adverse domains; compare the information in the direction, not just edit size. Labels may evaluate direction but may not choose local radii, thresholds, states or signs. Fix numeric gates and finite budget before observing new outputs.
3. Only if the direction is informative, evaluate a separately registered parameter-update diagnostic. In the quadratic objective centered on native Adam, the term -J_b^T W_b r_b supplies signed correction; with r_b=0 it collapses toward the old preservation premise. The formula itself is not a novelty claim. Compare the same residual under ordinary optimization and under the joint formulation; JS must earn its extra computation.
4. A fixed short closed-loop test is distinct from a one-step test. Late-state single-step near-zero gains are not an upper bound on multi-step or earlier-state effects. Do not start multiple horizon searches. A short-stream length and temporal coverage must be frozen prospectively.

No target-ground-truth gradient enters an online update. Supervised gradients, if included later, belong solely to isolated offline attribution and cannot substitute for finite-step Dice or closed-loop evidence. Source-trained correction is a separate historical family (R16); the currently authorized eight references do not authorize a new source-training program.

## Current disposition

DESIGN_ONLY. No experiment number is marked running. Remaining work: deterministic geometric reconstruction and synthetic checks, bounded state-availability audit, then a complete registration if those prerequisites justify it. Do not start another RCA selector or another JS strength/ridge scan. Continuous30-minute research remains active; report preparation as preparation, without invented scientific evidence.

## Mechanical preparation update

Array-only geometry and generated fixtures are now implemented; see [preparation report](../reports/r66-boundary-preparation/REPORT.md). Still no real-data registration or efficacy result. The previous unresolved reconstruction item is addressed with nearest-boundary ownership, original-resolution signed-distance edits and nesting veto. Bounded state inventory found only1900/1951 slots in inspected R50 trajectories; use archived predictions for an initial output-only diagnostic rather than claim unverified early-state update coverage.
