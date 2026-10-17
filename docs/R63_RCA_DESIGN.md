# R63 candidate-specific reverse-classification feedback

Status: CONCRETE DESIGN, NOT REGISTERED OR RUN. A source-access scope decision is pending. Existing R20 protocol section 2 forbids source images/labels and extra pretrained models; a new RCA run cannot silently override this boundary. No source data have been opened for this design.

## New information and overlap

[Valindria et al., TMI 2017](https://arxiv.org/abs/1702.03407) constructs a separate reverse classifier from each image and its predicted segmentation, and evaluates that classifier on annotated reference images. The reference database supplies correctness information unavailable in a target-only confidence score. The paper's atlas-forest variant uses 50 trees, maximum depth 30 and box features; no registration is required for that variant. Its evidence is multi-organ imaging, not a Fundus guarantee. This would be a task-specific RCA diagnostic, not a faithful recreation of all paper features or a novel-method claim.

For candidates A and B on the same image x, train f(x,A) and f(x,B) separately. Their source-reference segmentation scores can differ. This resolves R63's action-independent D(x) problem. Unlike R22, the score is not a graph cut or neighbor agreement; unlike R11–14/R53, it is not output averaging or a fixed set operator; unlike R61/R62, it does not rewrite the native update target. A bounded exact-term search of project protocols/implementation found no previous RCA/atlas-forest implementation. This is a repository overlap finding, not a universal novelty claim.

[In-Context RCA](https://arxiv.org/abs/2503.04522) is a related newer route, but would add an external in-context segmentation model as well as references. It is not selected here. The classical forest route avoids that dependency and GPU inference.

## Proposed fixed pilot if source-reference access is authorized

- Reference pool: eight existing source validation images from distinct available source groups, selected by the established frozen split order before opening their pixels or labels. No additional annotation, source backbone training or external model. If eight eligible groups or archive bindings are unavailable, report infeasibility; do not substitute target labels or another source fold silently.
- Target panel: the existing 48 balanced SEARCH contents (12/domain), seed17011 and both stream orders. Reuse archived CW_LSO and VPTTA masks from R52, provided their original identities and coverage match. Two candidates per state-image pair, 96 pairs. The models are separately adapted; this is output selection, not a same-state update intervention. Do not reuse R62 masks as substitutes.
- Proxy representation: 128x128 area-resized RGB, with raw RGB and channelwise local means/standard deviations at fixed 3/9/27-pixel windows (21 dimensions). Reflect padding. Candidate/reference masks resized nearest-neighbor for proxy fitting/scoring only; selected output and target evaluation remain the original 512x512 masks. No learned scaler, position features or feature/scale sweep.
- Train one 50-tree, max-depth30 random forest per candidate/channel on all 16384 current pixels; fixed seed20261017, sqrt feature sampling, standard bootstrap, no class weighting. Empty/single-class labels produce the corresponding constant predictor. Keep OD/OC independent, with no added nesting repair.
- Candidate score: for each reference, mean OD/OC hard Dice from the reverse prediction; take the maximum joint score across the eight references. Choose the higher-scoring candidate as a whole two-channel mask; exact ties choose CW_LSO. Do not combine separate per-channel maxima from different references or choose different candidates by domain.
- Baselines: unchanged CW_LSO, unchanged VPTTA, fixed equal random-choice expectation, and offline label-informed candidate oracle. The latter is diagnostic only and cannot influence selection. Primary selector minus CW_LSO, equal domain/channel/order weighting. Require >=0.1pp, both orders positive, content-cluster bootstrap95% lower bound>0, worst domain/channel/order >=-2pp, and REFUGE_Valid OC >=1pp. Report all failures, oracle headroom, pairwise ranking accuracy and selection coverage. No post hoc gate replacement.
- Separate stages: source-reference data are explicitly permitted only under the requested new scope; target labels remain denied until every candidate score and selected mask is committed and selector workers have retired. Source reference overlap with original checkpoint training is disclosed; source-reference quality is not independent validation.
- Resource cap: CPU only, two threads, 7200 aggregate CPU-worker seconds including profiling, <=3h after launch, <=2GiB new artifacts. First real pair qualifies determinism, output coverage and projected full cost, without reading target labels. Failure closes the matrix with receipts; no automatic reference/feature/configuration search.

## Before registration

Confirm the requested scope exception; check reference manifest/group availability and R52 archive bindings; confirm the installed forest dependency; implement the fixed score and test candidate dependence, channel/size handling, deterministic ties and target-label denial on generated data. Freeze source/code/configuration and a new run identity before the first real pilot prediction. No full adaptation trajectory is implied even by a positive selector result.

## Limits

A source-easy reverse prediction can be a bad quality proxy under domain shift. Candidate-label fitting can reward shared mistakes or source appearance similarity. Downsampling can erase thin boundary differences; current ROI provenance and patient linkage remain uncertain. This is an exposed-development pilot. Access to source references changes the deployment contract and must be reported explicitly; it cannot be called model-only or source-free CTTA.

## Preparation completed while source access remains pending

The [array-only scoring core](../src/dpa_ctta/r63_rca/score.py) now implements the fixed feature extraction, per-channel forests, constant-mask handling, maximum joint-reference score and CW_LSO tie fallback. For exact reproducibility, 512-to-128 RGB area resizing uses non-overlapping 4x4 averages; nearest mask sampling uses indices 2,6,...,510 (upper-index tie at each bin centre).

[Generated-data checks](../reports/r63-rca-preparation/SYNTHETIC_CHECK.json) passed in the existing CPU environment: candidate dependence, exact repeatability, tie handling, empty masks, joint-reference versus separate channel maxima, and rejection of malformed/nonfinite inputs. The constructed correct/reversed color segmentation scores are 1.0 and approximately 6.1e-11; these are fixture outcomes, not medical performance. CPU fixture elapsed time was 1.65 seconds and does not estimate full real-image runtime.

There is no data loader or launcher in this core. Source-reference authorization, actual archive/manifest bindings, the process-level target-label denial guard and real-pair resource qualification remain outstanding. The array signature alone is not a security boundary. No source or target images/labels were read and no formal experiment was registered or started.
