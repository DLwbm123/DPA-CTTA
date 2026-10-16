# R61 evidence design: no experiment registered

2026-10-17. This is bounded evidence synthesis, not an efficacy result. No GPU job or new dataset/model was launched. Continuous research remains active.

## Existing evidence constrains the next action

- [R23](../reports/r23-interval-quick/REPORT.md): disagreement masking and interval supervision changed macro Dice by -0.0835 and -0.1473 pp versus CW; both orders were negative. Uniformly suppressing disagreement is not supported.
- [R28](../reports/r28-common-state-diagnostic/REPORT.md): the four-candidate oracle gained 0.080122 pp, below the 0.3 pp allocation gate; actual nesting selection gained 0.000840 pp. A better selector alone is insufficient for this candidate pool on these states.
- [R29](../reports/r29-update-strength-diagnostic/REPORT.md): the FULL/ZERO/HALF oracle gained 0.010225 pp, also below its gate. Do not expand the strength grid.

These are local, development-exposed diagnostics. They do not bound all multi-step effects or all possible candidate pools. See the broader [closed-route ledger](RESEARCH_STATE_R60.md).

## Primary-source checks

[TEGDA, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2263_paper.pdf), sections 2.1–2.3 and implementation details: ADIC combines dropout agreement and confidence. Historical features passing a past-score percentile enter a FIFO bank; similarity-weighted feature fusion produces refined pseudo-labels. The paper uses ten dropout inferences at rate 0.5, bank length ten, and percentile ninety. Its experiments concern cardiac and brain MRI. This suggests historical image features as an additional information source, but does not establish calibration or efficacy in our Fundus stream. Our successful R57 control explicitly asserts absence of dropout modules. Inserting dropout into this checkpoint would be a new perturbation design, not faithful reuse of a trained dropout model. ADIC estimates absolute quality; usefulness for choosing a correction requires a separate test of paired improvement, not only correlation with Dice.

[BA-TTA-SAM, arXiv 2512.04520v1](https://arxiv.org/html/2512.04520v1), sections 3.2–3.3 and 4.1.3: Gaussian prompt injection and shallow-feature Sobel boundary alignment operate in SAM. Inference uses a bounding-box prompt covering the object. The text reviewed does not establish an automatic label-free source for that box. Thus its reported gains do not establish a compatible prompt-free mechanism here. Raw RGB/green-channel edge selection is also not the mechanism tested by that paper. Do not infer that its results justify a new erosion/dilation selector.

## One remaining candidate and its unresolved requirements

Candidate: use strictly earlier target-image features to refine a current pseudo-target. The additional information is other observed images' representations, rather than another scalar weight on the current prediction. This is a literature-derived hypothesis, not a novelty claim or a selected implementation.

It is not yet launch-ready. Resolve these questions before registration:

1. Locate the exact deployed encoder/decoder interface and compare existing feature/prompt memory implementations. Determine whether spatial feature fusion is meaningful without correspondence; do not average anatomically unaligned maps merely because shapes agree.
2. Establish a fixed, label-free reliability rule compatible with the existing checkpoint. Do not replace ADIC by entropy and call it a reproduction. Do not add or tune dropout locations/rates against exposed target outcomes.
3. Specify an actual correction candidate and why it can move both inward and outward boundaries. High confidence and historical similarity alone do not imply correct error direction.
4. Before any full trajectory, freeze one matched-state diagnostic against the unchanged pseudo-target, with past-only bank construction, one fixed configuration, a finite budget, and labels withheld until prediction/choice completion and worker retirement. Assess paired candidate headroom and actual selection, all domain/channel cells, and image-weighted change. Register numeric allocation and harm thresholds before collecting scores. Do not import R28's result as a bound for this different candidate pool.

Next scheduled step is a bounded compatibility/code audit of this single candidate. If the interface or reliability premise fails, record that failure and choose a distinct evidence source; do not launch a proxy controller or another morphology/learning-rate sweep. Source retraining, a new foundation model, and additional datasets are outside this step. No performance or independent-generalization claim follows from this design note.
