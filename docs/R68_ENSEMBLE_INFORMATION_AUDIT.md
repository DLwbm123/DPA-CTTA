# R68 preparation: ensemble-information overlap audit

Decision: **no new experiment registered or launched**. Generic ensemble-disagreement weighting is already implemented. The inspected evidence does not justify another confidence gate, variance temperature, view mixture or update-strength sweep. This is a code/protocol audit, not a new measurement or literature novelty claim.

## Evidence inspected

| Existing mechanism | Direct evidence | What it does and does not establish |
|---|---|---|
| Six-view variance reliability | `src/dpa_ctta/r20_model_only_search/method.py`, `weights_for` and `_criterion`; `docs/protocols/R20_EXECUTION_REGISTRATION.md` | W already computes variance about the six-view probability mean, applies a positive exponential reliability factor and boundary boost, then normalizes spatial mean weights. A variance-disabled boundary-only ablation is specified. Calling this ensemble disagreement does not introduce a new information source. |
| Confidence-gated horizontal fusion | [R14 report](results/r14_uncertainty_flip/REPORT.md), `src/dpa_ctta/r14_uncertainty_flip/view.py` | Fixed original probability [0.4,0.6] gates quarter flip mixing. Target hard Dice improved 0.111434pp over C0 but was 0.048329pp below constant quarter. This is frozen output fusion, not the current CW state or a learned direction signal. |
| Source/ensemble disagreement | [R23 report](../reports/r23-interval-quick/REPORT.md), `src/dpa_ctta/r23_interval_quick/method.py` | MASK suppresses source-versus-mean hard conflicts; INTERVAL projects detached strong predictions into their probability interval. INTERVAL-CW -0.1473pp and MASK-CW -0.0835pp. These are conservative supervision tests, not a direct test of every within-ensemble statistic. |
| Candidate nesting and scalar step strength | [R28](../reports/r28-common-state-diagnostic/REPORT.md), [R29](../reports/r29-update-strength-diagnostic/REPORT.md), both `method.py` files | Four residual actions selected by nesting and FULL/ZERO/HALF updates had insufficient immediate oracle headroom (0.080122pp and 0.010225pp respectively). These were ANCHOR states, not current late CW states; they do not universally bound future closed-loop changes. |
| Original versus ensemble target | [R62 protocol](protocols/R62_GEOMETRIC_TARGET_DIAGNOSTIC.md), [report](../reports/r62-geometric-target/REPORT.md), `src/dpa_ctta/r62_geometric_target/run.py` | Replaces only weighted BCE target, retaining W and LSO. Update gain +0.001350pp crosses zero; original-view target quality -0.124533pp. Both gates fail. No view-mixture scan follows. |

## Why variance is not a correction direction

For aligned probabilities p_v and mean q, current W uses v = mean_v (p_v-q)^2. Reflecting deviations about q leaves v unchanged, and permuting view identities also leaves it unchanged. This statistic measures spread, not which view should be trusted. With a fixed detached target q and positive detached weight w, the pointwise weighted-BCE logit gradient is w*(sigmoid(z)-q): changing w cannot reverse that pixel's gradient sign. The aggregate parameter gradient can change direction because spatial weights change; this audit does not claim otherwise.

Identical prediction tuples can also occur with different unknown labels. Thus disagreement alone is not a universally identifying error oracle. That mathematical limitation is not proof that disagreement has no statistical predictive value on a population. Ordered view responses, higher moments or image-conditioned relations are not all excluded by the inspected experiments. However, no specified label-free rule here establishes which such statistic provides the correct sign, and fitting one to these exposed target labels would be a new supervised development procedure, not a free unsupervised correction.

R23 concerns source-versus-ensemble conflict, whereas R14 uses original confidence and R62 tests a target replacement. They must not be presented as exhaustive tests of all six-view information. R28/R29 must not be misrepresented as a full-trajectory impossibility bound.

## Scope and next bounded action

Close the vague proposal to add augmentation disagreement beyond the mean: variance is already consumed, conservative disagreement variants failed, and a distinct signed feedback construction remains unspecified. No new source training, view/radius/temperature search, extra target-label fitting, or GPU replay is authorized by this audit. Existing continuous authorization is unchanged; this decision is not a pause of the research task.

Next perform a bounded primary-source/repository audit of test-time objectives using an observable image reconstruction or acquisition-consistency residual, asking whether any concrete residual is feasible with the existing Fundus inputs and frozen ResUNet without new source training. Compare against the existing R61 image-energy rejection and R66 feature-boundary failure first. An image prior, entropy or self-generated target alone is insufficient. If a candidate needs an unavailable forward acquisition operator, reconstruction decoder or trained judge, record that requirement and reject it within this conversation's current scope rather than fabricate a residual. At most one supported construction may proceed to a fixed protocol; do not promise a run from this preparation.

No new inference, label scoring, process launch or scientific performance result occurred. Previously reported numbers above retain their original sample/state/metric limitations. The next question is an audit target, not a claim that such a residual exists or works.
