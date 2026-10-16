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

The compatibility audit below supersedes the proposed next step. Source retraining, a new foundation model, and additional datasets are outside this step. No performance or independent-generalization claim follows from this design note.

## Compatibility audit outcome

Completed 2026-10-17 using deployed dependency source, the host call chain and existing reports. No inference, training or additional target scoring was performed.

**Interface is accessible; the reliability premise remains unsupported and overlaps earlier work.** The runtime builder uses `networks.ResUnet_TTA.ResUnet`, not the neighboring `ResUnet.py` class. Its forward returns `(seg_output, sfs, head_input)`. `res` supplies the encoder bottleneck and four skip maps; `up1` through `up4` concatenate upsampled features with the current image's skips. `up5`, `bnout` and `seg_head` produce the two-channel output. A bottleneck-only replacement therefore retains current-image skips; copying complete historical maps would mix anatomical positions without established correspondence. Matching tensor shapes does not establish alignment.

Existing mechanisms already address parts of this issue:

- [R20 host](../src/dpa_ctta/r20_model_only_search/method.py) and [R21 correction](../src/dpa_ctta/r21_regional_quick/method.py) use a frozen source feature coordinate system and class prototypes, avoiding direct spatial-map averaging. R21 stores up to eight image prototypes per class, expires them after 128 arrivals and writes current entries after prediction. Source/flip confidence is a fixed gate, not a guarantee of correctness.
- [R22](../src/dpa_ctta/r22_cut_quick/method.py) already uses frozen `up3` features, spatially pooled tokens and mutual nearest-neighbor correspondence to the preceding eight images. It compares source and student candidates without changing the baseline updates. Thus token retrieval, past-only storage and fixed coordinates are not new contributions here.
- [VPTTA host](../src/dpa_ctta/hosts/vptta.py) retrieves prompts using low-frequency image keys and writes adapted prompts after the current step. That is prompt initialization, not an anatomical feature bank or an independently reliable pseudo-label teacher.

[R21's fixed pilot](../reports/r21-regional-quick/REPORT.md) found CWR-CW -0.0277 pp, with 299,874 corrected versus 302,932 damaged pseudo-label pixels. [R22's diagnostic](../reports/r22-cut-diagnostic-quick/REPORT.md) found cut AUROC 0.5333 versus confidence 0.5600, and cut selection changed candidate macro Dice by -5.2452 pp. Those are pseudo-target selection results, not a trained-method score.

An arithmetic reanalysis of the existing [R22 aggregate](../reports/r22-cut-diagnostic-quick/RESULTS.json), subtracting student from judge in each of its 16 order/domain/channel cells, gives:

| Judge | Positive cells / 16 | Minimum change (pp) | Maximum change (pp) |
|---|---:|---:|---:|
| Cut | 0 | -13.026971 | -0.192176 |
| Confidence | 2 | -12.378328 | +3.791837 |
| Entropy | 1 | -12.448060 | +0.736838 |

No target files or new predictions were used for this table. The cut failure is not confined to one adverse cell in these aggregates. This does not prove that all feature-based methods fail, but it removes the justification for launching the presently proposed bank merely under a new name. Changes to capacity, neighbor count, threshold, dropout or feature layer alone are not a supported next experiment.

**Decision:** do not register or run the historical-feature-bank candidate. The next bounded evidence task must identify feedback outside these same-checkpoint predictions and their historical feature votes. Inspect primary-source evidence for label-free, image-derived anatomical boundary information under a prompt-free Fundus protocol, and check prior image/feature-boundary mechanisms before proposing an action. Explicitly address vessel edges, illumination and cup/disc ambiguity; a Sobel filter alone is not an anatomical cue. If no usable evidence is found, retain a documented design gap rather than run an arbitrary edge or morphology sweep. Any eventual diagnostic still requires a fixed candidate, matched comparator, numeric falsification gate and finite budget before target scoring. This is continuation of research design, not an experiment failure or a claim that the research goal has been achieved.
