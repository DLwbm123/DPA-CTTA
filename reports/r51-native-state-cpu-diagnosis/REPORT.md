# R51 CPU diagnosis: roundtrip risk proven, original failure not uniquely attributable

**The original GPU parameter-mismatch root cause cannot be uniquely identified from the saved evidence.** A float32 subtract/add restoration defect is demonstrated, but it is not established as the cause of that failure. No GPU retry, image/model replay, model forward, label read or production code modification was performed. Exact native-state equality remains mandatory.

## Evidence and measurements

The preserved first qualification case contains the visit1900 anchor, Adam state, diagnostic full gradients, metric, native displacement and branch displacements (82 parameter tensors, 19,136 float32 coordinates). It does not contain the original native-reference post-step parameters/gradients or the instrumented raw post-Adam parameters before shadow writes. The original failure recorded only `parameters`, not a coordinate count or magnitude. The second anchor was never qualified.

The CPU script imports the frozen production `write_delta`; it runs arithmetic on saved tensors and one CPU Adam evaluation with the already saved diagnostic gradient. It never instantiates the segmentation model or reads images. CPU-only execution is enforced and CUDA remains uninitialized.

| Quantity | Measured result | Meaning |
|---|---:|---|
| Saved native displacement vs saved native-branch displacement | 0 / 19,136 different; max absolute difference 0 | Displacements survived subtraction again; this does **not** establish parameter identity |
| CPU production write-back displacement vs saved native-branch displacement | 0 / 19,136 different | CPU reproduces the recorded displacement arithmetic |
| Repeating production write-back from the same anchor | Exact | No accumulating drift across repeated writes from the same anchor |
| Coordinates where multiple post-step float32 values encode the same saved displacement | 1 / 19,136 | Original raw post-step parameter cannot be uniquely recovered at that coordinate |
| Pure subtract/add restoration error bound for the saved diagnostic displacement | At most 1 coordinate; max absolute error <= 9.094947017729282e-13 | Conservative arithmetic bound, **not an observed original failure magnitude** |
| CPU Adam raw parameters vs their subtract/add restoration | 0 / 19,136 different; max absolute difference 0 | The saved-gradient CPU optimizer case did **not** reproduce the restoration error |
| CPU Adam displacement vs saved GPU diagnostic displacement | 2 / 19,136 different; max absolute difference 1.4901161193847656e-8 | Cross-backend optimizer arithmetic is not an exact GPU reference |
| CPU Adam metric vs saved GPU diagnostic metric | 6,298 / 19,136 different; max absolute difference 1.862645149230957e-9 | Further evidence that CPU optimizer reconstruction cannot replace missing GPU state |
| Saved full-gradient decomposition relative error | 8.369292561831491e-7 | Below the frozen 1e-5 gate; this compares diagnostic loss components, not diagnostic vs native-reference gradients |

### Why identical saved displacements are insufficient

The code computes `delta = float32(post - before)` after Adam, then restores parameters with `float32(before + delta)` for both the native prediction branch and the final native state. Rounded subtraction is not always invertible.

A generated scalar witness using the actual production writer has `before = float32(1e-4)` and raw post-step `float32(-1e-12)`. Subtract/add restoration gives **0**, differing from the raw post-step value by **9.999999960041972e-13**. Nevertheless, subtracting `before` again produces exactly the same saved displacement. Direct `copy_` of the preserved raw post-step value gives exact equality.

For the real saved tensors, the script examines the float32 neighbors of the reconstructed post-step parameter. At one coordinate both neighboring parameter values subtract to the same saved displacement. Elsewhere the subtraction preimage is unique, so pure displacement reconstruction cannot change the raw parameter there. The error bound uses exact rational midpoint intervals between adjacent float32 displacement values, with closed endpoints for conservative treatment of ties. It bounds only **diagnostic raw post-Adam vs reconstructed diagnostic parameters**, not **original native-reference vs diagnostic final parameters**. If a future retained failure comparison exceeds this count or magnitude, pure subtract/add restoration alone cannot explain it.

## Causal boundary

Define three states:

- **A:** original native-reference state after its step.
- **B:** instrumented diagnostic state immediately after Adam, before any shadow parameter write.
- **C:** diagnostic state after restoring the native branch at the end.

`A == B` with `B != C` establishes restoration corruption. `A != B` establishes a difference in the instrumented update before restoration; copying B exactly cannot repair that difference. Both mechanisms can occur together. The current saved displacement identifies neither A nor B, so the original failing `A != C` cannot be decomposed retrospectively. Matching binary masks and passing the internal CW+LSO gradient decomposition do not resolve this missing comparison.

Extra VJPs/forwards and gradient decomposition exist in the instrumented path. Whether they changed the native-reference gradient or update numerically is **unknown**, not a diagnosed cause. No nondeterminism, tolerance problem or altered objective is asserted without evidence.

## Minimal proposed repair — not applied

1. Immediately after `n.base.step()`, clone the exact post-Adam BN parameters. Continue computing the displacement for J/control geometry as before.
2. For the `native` prediction branch and final restoration, use `p.copy_(saved_native_parameter)` instead of subtract/add reconstruction. Leave J, radial/null proposals, matching thresholds, RNG handling, optimizer state and all scientific settings unchanged. Test exact copy on the scalar counterexample; CPU assertions already show zero copy error and unchanged optimizer state. This eliminates the proven restoration risk but is **not a certified fix for the observed failure**.
3. Before the existing assertion can abort, retain private A/B/C parameter snapshots and native-reference/diagnostic gradients and Adam states; write aggregate per-comparison differing counts, maximum absolute differences and L2 errors. Evaluate A–B and B–C separately, and report each remaining state-component comparison rather than discarding later evidence at the first failure. Keep `torch.equal` and the current hard stop; never replace them with `allclose`, a larger tolerance, or a silent correction of B to A.

Affected production sites are limited to `one_pair`'s post-Adam capture/native writes, and qualification's evidence capture before the unchanged state-equality assertion. A small regression for a non-invertible scalar write and exact restoration/unchanged Adam state suffices for the restoration fix. Any eventual GPU verification remains separately unauthorized here.

## Provenance, scope and delivery

Frozen inspected implementation: `886491b6673caa920eb71080c75c0e6779c1dc65`; prior failure report: [retry2](../r51-same-state-diagnostic-retry2/REPORT.md). Source locations: `diagnostic.py` lines 91–92 (rounded displacement), 108 and 121 (native reconstruction), 140–144 (exact comparison), and `run.py` lines 122–135 (reference only in memory, saved vectors, exception before completed qualification row).

`reproduce.py` and `RESULTS.json` retain the CPU reproduction and aggregate measurements. Two CPU invocations were made: initial measurements, then the added rational bound. Their timed analysis sections were 1.585972799s and 1.460882295s; interpreter/import and inspection time is not included. Added GPU-worker cost is **0**; the campaign remains at **30.534639769 / 1800 seconds**, formal coverage **0/96**, with all three prior attempts preserved.

No private weights, full gradients, parameter coordinates/values, image identities or paths are published. The scalar witness is synthetic. Production code and experimental gates remain unchanged; this report delivers a scoped repair proposal and an explicit unresolved original cause.
