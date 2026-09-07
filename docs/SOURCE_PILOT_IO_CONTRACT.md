# Source pilot I/O contract (draft, not execution approval)

Base: `c92b64b2284eb5da0631c00d15cc44af773f4b2d`. External source is pinned to `DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`. This patch prepares source-only N/A/B/C assembly. The supplied second review approved the previous minimal CPU core only. It did not approve this runner, any checkpoint/data registration, GPU use, or real pilot.

## Checkpoint, clone, device

`VPTTAHost(source_state=..., device=...)` constructs native prompt then full native model, validates the entire tensor state (exact keys, shapes, dtypes, finite values), loads with `strict=True`, then deep-copies the loaded live model before any forward. The clone has equal parameters AND running buffers, independent storage, frozen parameters, and isolated native hooks. Native custom AdaBN is preserved in A/B/C. No `strict=False`, downloaded weights, synthetic checkpoint files, or new source training is used.

All live/prompt/clone tensors and prepared proxy tensors move to the requested final device before native Adam is constructed. A first-live-forward guard also covers callers invoking the extracted native block directly. It checks initial counts/memory/Adam, optimizer parameter identity, and full live/clone state equality/disjoint storage. A later load-only-live attempt before that forward fails `SOURCE_CLONE_STATE_MISMATCH`. The initial hook removes itself; there is no per-step full-model backup or rollback. Loading/replacing weights after the stream starts is outside this contract.

The default remains random CPU construction for previous synthetic users. Base parameters retain native `requires_grad` and accumulated `.grad`; only prompt parameters enter Adam. The frozen clone still allows input/prompt gradients. Extra proxy forwards synchronize clone sample counters with live but never advance live counts, alter its hooks, step Adam, or push memory.

The future real runner deserializes only after its unconditional approval guard, with `torch.load(..., weights_only=True, map_location='cpu')`, then uses the same strict loader in every arm. Real registered checkpoint identity/path and training membership are **not verified in this phase**; the private checkpoint environment binding must be checked against the existing approved record before any future execution. Shape/key validation alone does not certify checkpoint provenance.

N uses the pinned `build_reference_model` source-only path: standard `BatchNorm2d`, evaluation mode, frozen weights, no prompt/adaptation. Native AdaBN with counter zero is not No Adapt. The synthetic state test changes convolution weights and running mean/variance and compares N to the correct full source model plus an independent standard-BN arithmetic reference.

## RGB, labels and distance

Readers use PIL RGB bytes, resize once, return CPU float32 `[B,3,H,W]` pixels in `[0,1]`. Fundus is 512 square with BICUBIC RGB, then per-image min-max. Polyp is 352 square with BILINEAR RGB, then ImageNet mean/std exactly once. The host separately moves normalized model input to its final device. It never calls the already-normalized old reader. Native Polyp FFT/memory still receive native normalized model input.

Verified upstream references: `ctta-repro-suite/src/ctta_suite/data.py::{read_image,Evaluator.evaluate}`, `methods/crisp_ctta/basis_training.py::_label`, and `VPTTA/{OPTIC,POLYP}/dataloaders`. All new masks resize with NEAREST on the final grid. Fundus channels are **OD = gray < 255**, **OC = gray == 0**; they are independent nested masks, not exclusive classes. Polyp is **foreground = gray > 127**. Legacy Polyp bilinear/equal-255 GT is not used for the main metric. Registered original image dimensions are checked independently on RGB and label reads.

Known fixed proxy masks produce `d = (EDT(1-y)-EDT(y))/hypot(H,W)` using installed SciPy. Distance is between opposite-class pixel centers (not a half-pixel interface offset), strictly negative inside, positive outside. Empty/full pairs return zeros and `defined=False`; their region loss remains active. Distances are generated after mask resizing and never interpolated from another grid. Proxy model input, mask and distances share float32 dtype and device. The explicit CPU pixel reader is not a GPU reader test.

Independent brute-nearest-distance tests cover small targets, non-square grids, nested OD/OC, empty/full. Existing medical-loss tests retain region/boundary gradient and graph-connected zero-boundary coverage.

## Frozen source registration

Only Fundus `RIM_ONE_r3` and Polyp `BKAI` are allowed. The public configuration contains environment variable names, not private roots. The adapter reads explicitly registered source CSV specs and the existing source manifest and `crisp-source-content-split-v1` artifact. It does **not** invoke upstream dataset audit, scan directories, read target metadata, create a split, or calculate new image/mask hashes.

The existing split metadata payload digest is verified; its group key is `SHA256(existing_image_digest + existing_mask_digest)`. Every record is joined to explicit CSV `image,mask` rows, source role, registered content digests, dimensions, and group membership. Exact coverage and cross-role sample/group/image-content separation are required. Duplicate pair groups use the lexically smallest existing sample ID as representative. Invalid domain/path traversal/symlink escape, mismatched CSV metadata, duplicate IDs or inconsistent frozen groups are rejected. Existing absolute manifest paths remain provenance; CSV-relative paths explicitly rebind the private root and must retain their registered suffix. Reading metadata does not revalidate actual image content against old digests; real source registration/freshness remains a future check.

Proxy: four distinct `basis_train` groups ordered by `SHA256(str(seed)+":"+group_id)`, tie-break group ID, fixed clean full batch K=4. `source_proxy` rejects query roles. Query: existing `critic_validation` group order, all 20 Fundus groups (a changed count rejects), first 32 Polyp groups. Insufficient groups stop; no repetition, random split, score selection or substitutions. Temporary test registries describe nonexistent procedural files and are never promoted to real registration.

The runner reads each query RGB, completes `host.step` (including native adaptation, inference and memory push), then calls the evaluator which first reads that query label. Host and proxy factory receive no query label. Replay tests change fake query labels and compare order, predictions, prompt and Adam; only metrics change.

## Lifecycle, metrics and failures

Arms are N source-only, A native VPTTA, B extra region weight .1, C extra region weight .1 and boundary beta .1. Both adaptive branches retain native lr .05/.01, Adam betas (.9,.99), wd=0, iters=1, warm_n=5, neighbor=16, memory_size=40 and prompt_alpha=.01. Every arm resets to the same source/RNG seed. Four sequential segments are clean, gamma .7, gamma 1.5, Gaussian 5x5 sigma1 with reflection padding; transforms precede normalization, labels do not change. State persists between segments.

The original memory evicts **before insertion only when its size is already greater than 40**, so 41 entries is expected. Default-neighbor tests instrument actual retrieval and eviction events and compare direct-native/wrapper output and state after every image. CPU NumPy views can share storage with the prompt; this original behavior is preserved. CPU equivalence does not establish CUDA memory semantics or CPU/GPU bitwise equality.

Per visit/channel records contain standard Dice at sigmoid >= .5 (0–1 scale, both empty = 1), region/boundary loss, defined boundary flags, GT/pred empty/full, ASSD in final-grid pixels, prompt update norm, native counts, Adam step, memory size, proxy losses and elapsed time. ASSD uses 4-connected foreground surfaces including image borders; either empty mask means undefined. It reports conditional means, valid/undefined counts and explicit valid cohorts. B−A, C−A, C−B comparisons retain each channel, segment and paired ASSD cohort. CUDA peak memory is `NOT_RUN` on CPU; the future CUDA branch is untested.

Nonfinite outputs/updates/losses, wrong weight state or label mismatch fail the loop; the real enclosing runner returns `INCOMPLETE` and never resumes the polluted host. It does not use GT to choose rollback or tune settings. Real-entry approval failure returns `NOT_RUN`, not an experiment failure or success.

The four transforms are repeated visits to the same development groups, not four independent patients. These groups were previously used; checkpoint training membership is UNKNOWN. This is a privileged raw-source replay mechanism control, not DD, strict source-free deployment, unseen generalization, multi-seed significance, a SOTA result, or clinical validation. All negative differences are retained.

## Entry and review boundary

`python -m dpa_ctta.source_pilot` defaults to `prepare`; `dry-run` also parses the disabled draft and reports its exact file SHA256 and `NOT_RUN`. `run` exits 2 with `PILOT_COMMIT_CONFIG_APPROVAL_REQUIRED` before even reading the config path. Direct `run_registered` raises the same guard before environment/metadata/image/checkpoint access. No flag, config value or environment override unlocks it. Current CPU core PASS is not used as authorization. A future task must explicitly approve the actual new commit/config, validate private registrations and authorized device, and review any corresponding gate change.

CI: `NOT_CONFIGURED`. Stop: **SOURCE_PILOT_PREPARED — AWAITING EXACT COMMIT/CONFIG REVIEW**.
