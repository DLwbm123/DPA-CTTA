# Independent code review index

Status: **CODE_REVIEW_PENDING**. Review the commit containing this document.
Original implementation: `617f67464df37d6e49972c4932a85f56d210ccab`.
CPU checks and provenance: [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md).
Passing checks do not grant experiment authorization.

## 1. Source loader and dependency pin

- [integrations/ctta_suite.py](../src/dpa_ctta/integrations/ctta_suite.py):
  `checkout_root`, `build_reference_model`, `_dependency_view`, `load_reference_interfaces`.
- Required CTTA commit: `dbff0d985c6c95345d9fb78f5b1daef57b392564`.
  The bridge imports `ctta_suite.models.build_model/model_logits`. It builds
  randomly initialized CPU models with `source_only=True`; it does not load weights.
  A temporary symlink view exposes the same checkout's VPTTA model definitions
  because the public pin omits `ctta-repro-suite/third_party`.
- [integration tests](../tests/test_real_integration.py): `ReferenceCheckoutTests`.
  `checkout_root` verifies HEAD, not a dirty tracked source tree. The local reference
  was clean in this run; production provenance remains a review concern.
- No real dataset loader is implemented by DPA. The external data/metrics modules
  are import-tested without opening images. No third-party source was vendored.

## 2. Manifold and FiLM initialization

- [manifold.py](../src/dpa_ctta/manifold.py): `LatentFiLM.__init__/forward`,
  `MultiScaleLatentFiLM.__init__/forward/train/close`.
  Three adapters share a 16-D state; gamma/beta bases initialize with normal std 1e-3.
  Source weights are frozen and the source model stays in eval mode.
- Zero state bypasses arithmetic only when the state does not require gradients.
  A differentiable zero state uses the formula, preserving a nonzero latent Jacobian.
  The unconditional bypass wording in the preserved METHOD is incomplete.
- [manifold tests](../tests/test_manifold.py): `LatentFiLMTests.test_zero_state_with_gradient_is_differentiable`,
  identity, nonzero-effect, frozen-source and cleanup checks.
  [publication checks](../audit/run_publication_checks.py): `focused_checks` also
  verifies input and zero-state latent gradients on both full model definitions.

## 3. Synthetic image/mask loss paths

- [anchors.py](../src/dpa_ctta/anchors.py): `image`, `soft_mask`, `descriptor`.
- [atlas.py](../src/dpa_ctta/atlas.py): `anchor_descriptors`, `forward`.
  Images enter the frozen source feature pass and image-style statistics; masks
  enter anatomy statistics. Both affect association, local chart deltas, proximal
  state, and downstream query logits. They are not an online supervised replay loss.
- [losses.py](../src/dpa_ctta/losses.py): `distillation_loss` sums a caller-supplied
  task loss, state MSE, trajectory-increment MSE, and classwise dense-feature loss.
  Its synthetic mask weights dense feature moments. A caller must construct the
  feature/image graph and task loss: this function is not a distillation trainer.
- [online contract tests](../tests/test_online_contract.py):
  `test_full_synthetic_distillation_gradient_connectivity` checks all anchor fields
  and FiLM bases through a synthetic logits/state objective. It does not execute
  the four-term `distillation_loss` as an end-to-end training pipeline.
  [loss tests](../tests/test_losses_and_offline.py) check the composite arithmetic
  and feature/mask gradients separately. No DD contribution is established.

## 4. Anchor versus target descriptors

- [descriptors.py](../src/dpa_ctta/descriptors.py): `FrozenDescriptor.forward`,
  `_style`, `_anatomy`; [online.py](../src/dpa_ctta/online.py): `_zero_pass`, `step`.
- Both use the same descriptor math and zero-state source features. Their anatomy
  inputs differ: trainable synthetic masks for anchors, source sigmoid predictions
  for target queries. Thus they share a feature definition, not an identical mask
  generation mechanism. Review this distribution mismatch before interpreting distances.
- `focused_checks` changes `z_prev` numerically to 0, 3, and -2 for the same image
  and compares complete descriptors. This supplements the original signature-only
  state-independence test. No detach-only argument is used as numerical evidence.

## 5. Chart and free state parameters

- `DistilledPotentialAnchor` holds trainable `image_logits`, `mask_logits`,
  `state_center` (m), `local_map` (B), and `precision_raw`.
- `DistilledPotentialAtlas.forward` uses scaled deltas for association and **raw**
  descriptor deltas for B. `test_local_chart_uses_unscaled_descriptor_delta` in
  [atlas tests](../tests/test_atlas_and_proximal.py) explicitly checks this.
  The preserved METHOD's scaled-q shorthand must not be read as B using scaled deltas.
- Free m/B/precision can encode useful updates without proving image distillation
  is necessary. Ablation and actual source/target performance remain unmeasured.
  `DescriptorScaler` defaults to unfitted identity; no source statistics were fitted.

## 6. Precision and proximal formula

- `DistilledPotentialAnchor.precision`: softplus(raw) clamped below at machine epsilon,
  then precision_floor added. Precision is diagonal.
- [proximal.py](../src/dpa_ctta/proximal.py): `closed_form_update` computes
  `(rhs + lambda * previous) / (precision + lambda)`.
- [ProximalTests](../tests/test_atlas_and_proximal.py): first-order residual,
  independent diagonal solve, floor validation, and contraction inequality.
  The contraction statement concerns identical current atlas inputs and differing
  previous states; it is not a segmentation accuracy or anti-collapse guarantee.

## 7. Online state, optimizer, and offline oracle boundary

- `DPAOnlineAdapter.step(image)` accepts one current image, has no label argument,
  and runs under `torch.no_grad()`. Only `z_prev` is intended to persist between
  calls; `reset()` zeros it. No online optimizer, replay memory, or native VPTTA
  warmup/counter lifecycle is implemented. This is the Atlas method, not VPTTA replay.
- Source and atlas tensors remain in `state_dict`; that serialization includes
  external source weights if a future caller saves it. No such artifact is published.
- [offline/oracle.py](../src/dpa_ctta/offline/oracle.py): `optimize_oracle_state`
  runs local SGD on a cloned state and returns it detached. It is source-supervised,
  not differentiable adaptation-oriented DD. Only the state is in the optimizer;
  trainable FiLM bases can still receive/accumulate gradients during backward.
  Review this before using it in repeated real offline episodes.
- Online mutation and round-trip semantics are covered in `OnlineContractTests`.
  `online.py` does not import offline code. A source_label argument name is a
  software boundary, not automatic proof of future caller data provenance.

## 8. Real-model integration and its limits

- [test_real_integration.py](../tests/test_real_integration.py):
  `TestFundusResUNet34`, `TestPolypPraNet`, full-size random CPU forward,
  identity/effect, module-output logits JVP, and hook cleanup.
- [configs](../configs/fundus_integration.json) and
  [Polyp config](../configs/polyp_integration.json) retain the historical discovery.
  This run reuses their injection paths without rewriting them.
- `focused_checks` verifies source parameters and registered buffers are unchanged
  through a differentiable full-model forward. The upstream source-only loader
  replaces custom BN with standard BN. This does **not** verify native custom
  warmup counters, unregistered attributes, or extra replay forwards in VPTTA.
- No pretrained checkpoint, clinical image, target stream, or accuracy evaluation.

## 9. False-positive, no-op, and disconnected-gradient checks

- Zero identity versus nonzero output changes: manifold and real integration tests.
- All anchor fields receive nonzero finite gradients: `AtlasTests.test_all_anchor_fields_receive_gradient`
  and the synthetic end-to-end contract check listed in section 3.
- Frozen weights still permit real-model input and latent gradients: publication check.
- Descriptor state independence: numerical publication check, not just API inspection.
- These checks test connectivity and behavior; no counterfactual DD-benefit experiment
  or revised host-disable equivalence test exists in this snapshot.

## 10. Label isolation

- `DPAOnlineAdapter.step` takes only image; losses/oracle use explicit source labels
  offline. [scripts/m0_oracle.py](../scripts/m0_oracle.py) rejects all real-data execution.
- `M0CommandTests` exercises dry-run and rejection paths using an intentionally
  nonexistent sentinel. Those subprocesses open no dataset and launch no experiment.
- The publication runner blocks image loading, torch checkpoint loading, and
  standard Torch download entry points during its in-process tests.
  No patient/target labels or synthetic medical tensors are included in this release.

## 11. Incomplete entry points and historical audit caveat

- `scripts/m0_oracle.py`: deliberate real-data blocker, not a usable M0 runner.
- [offline/distill.py](../src/dpa_ctta/offline/distill.py): only a loss re-export.
- [offline/trajectories.py](../src/dpa_ctta/offline/trajectories.py): tensor differences only.
- No M1/D0 trainer, source split registration, trained atlas, fitted scaler,
  real dataset loop, or target evaluation entry point is provided.
- [scripts/pretraining_audit.py](../scripts/pretraining_audit.py) is a historical
  report generator: it accepts test count and writes zero failure/error and access
  flags rather than parsing runner outcomes. It was not used to certify this release.
  Old audit JSON/manifest are retained as historical files, not new-commit evidence.
  Current test counts come directly from unittest in the publication runner.

## 12. Proposed paths without code

The [revised plan](REVISED_EXPERIMENT_PLAN.md) proposes base/medical_no_dd,
proxy_rehearsal/condensed_rehearsal, medical region/boundary losses, style conditioning,
and an optimizer-consistent adaptation-oriented DD update. These modes and training
pipelines are **NOT_IMPLEMENTED** here. No guessed filenames are review targets.
First obtain independent review of this exact commit, then specify and authorize
a minimal implementation patch and source-only pilot separately.
