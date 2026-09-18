# Inference context and provenance

The authoritative implementation is `src/dpa_ctta/r7_shared/context.py`. JSON uses sorted keys, compact separators, UTF-8 and disallows NaN. The context is:

```text
schema: R7_INFERENCE_CONTEXT_V1
sha256: SHA256(canonical JSON of payload)
payload:
  environment:
    effective_state_sha256: canonical named frozen parameters and all non-None buffers
    observer_projection_sha256: actual projection tensor, canonical name "projection"
    model_layout: named module classes, eval flag, extra_repr; BN eps/momentum/affine/
                  num_features/track_running_stats and absent buffers
    policy: schema R7_SEGMENTER_POLICY_V1; pinned preprocessing, current-stat BN,
            FiLM sites/interface, observer and pinned CTTA source identifiers
  method:
    schema: R7_METHOD_DIGEST_V1
    group: A | B | C | C0
    mode: FULL | STATIC | ZERO
    weights_sha256: unchanged Method.digest (basis/scaler/weights/calibration), null for C0
  ablation: null | A_ISO_OBS | B_PRED_ONLY | C_CONST_R
  source:
    schema: R7_SOURCE_PROVENANCE_V1
    design_manifest_sha256: original design MANIFEST file digest
    science_sha256: mapping of four original spec file names to SHA256
    source_binding_status: PENDING | PROCEDURAL_CPU | BOUND
    checkpoint_file_sha256: null or independently supplied file SHA256
    source_manifest_sha256: null or independently supplied file SHA256
    source_split_sha256: null or independently supplied file SHA256
    training_asset_file_sha256: null or independently supplied external file SHA256
```

`R7_PREPARED_TENSORS_V2` contains `weights`, full `binding` context and separate legacy `method_digest`; `prepare_tensors` additionally retains projection/validation/budget metadata. `R7_ONLINE_STATE_V2` contains `context`, `binding` (the context SHA256), `ablation`, state, visit count and first-failure fields. Neither schema grants an execution authorization. Failed states remain non-resumable. Missing/old versions are rejected.

Named tensor canonicalization: sort unique names; hash domain separator `R7_NAMED_TENSORS_LE_V1\0`. For each tensor append an unsigned 8-byte little-endian length and JSON header (`name`, exact torch `dtype`, `shape`, `nbytes`), followed by an unsigned 8-byte little-endian raw-byte length and detached, contiguous logical C-order little-endian bytes. CPU staging does not change dtype or values. Names/shapes/dtypes/bytes matter; device, pickle timestamps and object addresses do not. Real and boolean strided tensors used by this backend are supported; sparse, complex and quantized representations are rejected. Same CPU/GPU tensor contents therefore have a defined same semantic identity, but this repair only tests CPU and does not qualify GPU execution.

| Identity | Meaning and trusted origin |
|---|---|
| Original science SHA256 | Exact original file bytes; independently fixed by supplied review record; no JSON reserialization |
| Implementation SHA | Git commit of executable repair and tests; recorded in delivery and each final log, not self-embedded into its own hash |
| Source checkpoint file SHA256 | Future externally bound checkpoint bytes; null/PENDING now; never inferred from random model weights |
| Effective-state SHA256 | Actual initialized frozen backbone tensors after the registered BN policy; not a serialization/file hash |
| Projection SHA256 | Actual observer projection content, not merely its seed |
| Method SHA256 | Existing method digest, including basis/scaler/learned/calibration buffers and group/static |
| Training-asset file SHA256 | Optional external asset provenance; null until a real external file is bound. It must not claim the hash of its own enclosing serialized context |
| Context SHA256 | Composite payload identity including environment, method, ablation and source/design provenance |
| Publication SHA | Later evidence-only commit; not falsely labeled as the tested implementation |

The expected context is persisted by trusted source preparation, retained outside candidate reconstruction, and passed unchanged into the loader. For an authorized deployment ablation, derive it explicitly from that trusted metadata. The loader calculates only the actual candidate identity. Original science digests are revalidated from archived bytes, and candidate source provenance must carry those same fixed science/design identities. Real source values are never replaced with target data. A source status BOUND requires non-null checkpoint/manifest/split fields but does not authenticate them or enable execution; the currently disabled IO layer must supply verified provenance in a later authorized stage.

Hash telemetry counts named-tensor calls/bytes and process/wall nanoseconds. Context-build timers include nested tensor hashing and method digest work; these timers overlap and must not be added. The per-visit check computes only lightweight metadata/version comparisons. No context-content hashing runs per image.
