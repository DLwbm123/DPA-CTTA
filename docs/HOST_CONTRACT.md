# Native VPTTA host contract — minimal core v1

Status: implemented for second review; synthetic CPU validation only. No independent
signoff, real-data pilot, source checkpoint, DD training, or target evaluation.

## Reference and exact reused code

Dependency: `DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`.
All paths below are inside that fixed checkout, read-only.

| Contract | Fundus | Polyp |
|---|---|---|
| Native control flow | VPTTA/OPTIC/vptta.py, VPTTA.run | VPTTA/POLYP/vptta.py, VPTTA.run |
| Random full model | OPTIC/networks/ResUnet_TTA.py, ResUnet | POLYP/networks/PraNet_Res2Net_TTA.py, PraNet |
| Native prompt | OPTIC/utils/prompt.py, Prompt | POLYP/utils/prompt.py, Prompt |
| Native adaptive BN | OPTIC/utils/convert.py, AdaBN | POLYP/utils/convert.py, AdaBN |
| Persistent memory | OPTIC/utils/memory.py, Memory | POLYP/utils/memory.py, Memory |
| Source loader preprocessing inspected | OPTIC/dataloaders/OPTIC_dataloader.py and normalize.py | POLYP/dataloaders/POLYP_dataloader.py |
| Input/model resolution | 512 x 512 | 352 x 352 |
| Native Adam | lr=.05, betas=(.9,.99), weight_decay=0 | lr=.01, betas=(.9,.99), weight_decay=0 |
| Shared fixed settings | prompt_alpha=.01, warm_n=5, iters=1, memory_size=40, neighbor=16 | same |

The bridge is [hosts/vptta.py](../src/dpa_ctta/hosts/vptta.py).
`native_step_from_source` selects the original `VPTTA.run` per-image AST, starting
at `self.model.eval()` and ending with `self.memory_bank.push`. It compiles that
unchanged image block and adds only a return of `pred_logit`. It verifies the
expected loss assignment and refuses label/data references. This avoids vendoring
or relicensing upstream source, and avoids executing its constructor, dataset,
checkpoint, metric, logging, and file-save paths. Upstream code retains
[VPTTA/LICENSE](https://github.com/DLwbm123/CTTA/blob/dbff0d985c6c95345d9fb78f5b1daef57b392564/VPTTA/LICENSE).

This is an executable native **image-step** bridge, not a new reproduction claim
for the full original loader/evaluator. The random model retains custom AdaBN and
Fundus feature hooks. It is separate from the old Atlas `source_only=True` bridge,
which intentionally converts custom BN to standard BN.

## Public modes and online boundary

- `VPTTAHost(task, mode="base")`: direct native step.
- `VPTTAHost(task, mode="proxy_rehearsal", extra_weight=0, proxy_factory=...)`:
  the same direct native step. The factory is not called; no proxy is built,
  no extra model/forward is created, and no extra random draws occur.
- With positive finite extra_weight, the source AST changes exactly one assignment:
  native BN loss plus `extra_weight * self._proxy_term()`. It uses the same prompt,
  the same one backward call, and the same persistent native Adam optimizer.

`step(pixel_rgb)` accepts one image only. It has no target/evaluator label, mask,
future image, metric, or candidate-selection argument. Fixed source/fixture proxy
supervision is prepared separately at construction. `FixedProxy` requires a
`ProxyProvenance` enum and owns detached copies; a caller's later edits to its
original tensors cannot change the prepared data. There is no target provenance
variant. This is a provenance contract, not an ability to detect a caller falsely
labelling target data as source: verified real-data ingestion is NOT_IMPLEMENTED.

## Native lifecycle retained

1. Set model eval, prompt train, native `new_sample=True`.
2. Initialize prompt from native memory when neighbors exist, otherwise ones.
3. Native prompt/model forward computes BN matching loss and advances native sample counts.
4. Optional fixed-proxy loss joins that loss; native Adam takes exactly one step.
5. Set native `new_sample=False`, run final native inference, then push memory once.

The optimizer contains **only prompt parameters**. Source weights and running
buffers do not update. Native source parameters retain their original requires_grad
and `.grad` behavior; base backward can accumulate source gradients even though
no optimizer owns those weights. The four-image check observed 134 Fundus and 259
Polyp source tensors with gradients. This is preserved and reported, not hidden
by a global zero_grad or a change to the native mathematics.

The upstream CPU memory stores NumPy views and has its original capacity/neighbor
semantics; the bridge does not silently repair those semantics. No native stream
resume/serialization API is claimed by this minimal host.

## Proxy isolation and its resource ceiling

A positive auxiliary weight creates one random-source model clone **before any
forward**. The clone is in eval mode with frozen parameters and owns separate
buffers, native BN attributes, and feature-hook objects. It shares the actual
native prompt, not a second prompt or optimizer. Each auxiliary call uses the
current native sample count but `new_sample=False` on its own AdaBN objects.
Native source running statistics remain immutable in this eval contract.

This clone provides a simple read-only source path without touching live native
BN loss/hooks or restoring graph-saved tensors in-place. Input gradients pass
through the frozen clone to the shared prompt. The clone is not updated, loaded
from a checkpoint, or saved. Its AdaBN moments still depend on the proxy forward
at the current warmup count, using the unchanged source running statistics.

Cost: one extra full model allocation (22,554,570 Fundus or 32,547,319 Polyp
parameters) and one extra full-model forward for the prepared proxy batch per
update. This has not been optimized or benchmarked. Replace it with a reviewed
functional forward only if memory cost justifies that extra implementation work.
No Atlas cache, style transfer, router, second optimizer, or implicit extra update.

## Pixel and model-input spaces

`model_input_from_pixels` requires CPU float32, already-resized `[B,3,H,W]` pixels
in [0,1]. It performs no image I/O or resizing. Target and proxy call the same
function exactly once; the caller must not supply normalized model inputs.

- Fundus: per-image min/max normalization across channels and spatial dimensions,
  matching the native `normalize_image_to_0_1`. Constant images are rejected because
  the original denominator is zero; no arbitrary replacement is introduced.
- Polyp: subtract [.485,.456,.406], divide by [.229,.224,.225], exactly once.
- Proxy masks are fixed independent binary OD/OC channels or one Polyp channel,
  at the same model resolution. The bridge does not resize masks or distance maps.
  A future source preprocessor must explicitly choose RGB resizing and nearest
  mask resizing, then recompute distances on that final grid. It must not inherit
  the legacy Polyp evaluator's interpolated GT rule silently.

**Known instruction conflict:** native Polyp Prompt performs its FFT on normalized
model input. Moving that FFT to raw pixels would alter the native base computation
and break the requested exact zero-weight equivalence. This bridge preserves the
native FFT input. No new color/FFT descriptor or style-transfer branch is implemented;
any future such statistics must use pixel_rgb and explicitly resolve this exception.
This is not an undocumented claim that all native FFTs use raw pixels.

## Actual evidence and limits

[tests/test_vptta_host.py](../tests/test_vptta_host.py) compares direct execution of
the unchanged pinned image block with the bridge using **four full-size procedural
CPU images per task**. Logits, prompt, persistent Adam moments/step, native counters,
memory contents, buffers, source gradients and Torch RNG match exactly. Neighbor=2
is used in both test arms to exercise retrieval within four images; production
constructor default remains the pinned value 16. No scientific tuning is implied.

Enabled checks invert known half-plane masks and signed distances, observe changed
nonzero prompt gradients, verify native BN/counters/memory/hooks/gradients are
unchanged by the auxiliary forward, and run the actual combined backward/Adam
update. No mocked or toy network replaces ResUNet34 or PraNet here.

These tests do not establish checkpoint behavior, medical accuracy, real-source
proxy utility, DD benefit, full-stream equivalence, or deployment readiness. There
is no checkpoint/data loader, real proxy selection, or source-split registration.
