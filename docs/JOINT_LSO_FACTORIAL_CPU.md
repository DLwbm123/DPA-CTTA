# C-background LSO x shared sensitivity: CPU wiring qualification

Scope: implement the agreed five conditions and validate them on generated CPU
fixtures. No GPU profile, formal experiment registration/queue, data access,
monitor, new experimental T0, or scientific result is created. CW_LSO remains the
unchanged strong control. Historical R46/R48 results and ledgers are untouched.

## Fixed conditions

Import `Host` and `candidates` from `dpa_ctta.joint_sensitivity.factorial`.
This module intentionally has no CLI/run/dispatch entry point.

| ID | Base consistency | LSO | Shared sensitivity J | Calls per image |
|---|---|---|---|---|
| C | original C, unweighted | off | off | 8F/1B/1Adam |
| C_LSO | original C, unweighted | 0.1 | off | 8F/1B/1Adam |
| C_JS_SHARED | original C, unweighted | off | R46 | 10F/5B/1Adam |
| C_LSO_JS_SHARED | original C, unweighted | 0.1 | R46 | 10F/5B/1Adam |
| CW_LSO | original C+W | original0.1 | off | 8F/1B/1Adam |

All five use the original BN-affine native C host and its optimizer, six weak
views, one strong view, pseudo-target and final prediction. J adds two fixed
imaging probes and four auxiliary VJPs;5B means four VJPs plus one base backward.
It does not add Adam steps, output fusion, labels, a new model or permanent state.

The first four candidates have no W component in the consistency loss. For the
two LSO arms, retain the original LSO branch's detached reliability calculation
at temperature0.1/boundary boost2. This is the same branch used in CW_LSO, not a
uniform-weight substitute for LSO. Reliability weights remain confined to LSO
in C_LSO/C_LSO_JS_SHARED:

    L_C = mean BCEWithLogits(z_strong, detached q_six_weak)
    L_C_LSO = L_C + 0.1 * original_LSO(z_strong, q, detached reliability)
    L_CW_LSO = original weighted_consistency + 0.1 * original_LSO

One combined loss.backward() differentiates the consistency and LSO terms.
The LSO reliability is not differentiated. Computing this branch uses no new
network pass. CW_LSO is delegated directly to the original loss implementation;
its candidate dictionary is exactly the existing strong-control dictionary.

J is precisely the R46 shared filter: the same Sensor/evidence, two probes,
four VJPs, shared basis rank<=4, strength1, ridge0.01, and Gram cutoff. Its Adam
metric is computed from the actual base-loss Adam state for that arm. The update
is the existing `filter_update(..., shared_span=True)` followed by its original
parameter write. Existing metric-contraction checks run after the step. There is
no R48 descent constraint or native-Adam commit repair in this factorial entry.
Thus this tests the shared mechanism, not the complete R48 package.

## Isolation and implementation boundary

Only the new factorial module admits the five configurations. J strengths are
fixed at1 for enabled candidates;0 is admitted solely to check the disabled
limit. Other IDs, loss strengths, learning-rate changes, component injections,
basis changes or descent flags are rejected before model/RNG initialization.

The historical JointHost CW_LSO-only admission rule is unchanged. The new entry
initializes the existing base host directly, defines C-background LSO in its own
criterion, and inherits the original shared step. It does not temporarily alter
candidate identity/configuration or relax any historical entry point. Existing
factories, exports, run scripts and frozen sources are not edited.

## CPU checks and evidence

Ten new test methods cover:

1. Exact five-candidate matrix, unchanged strong-control dictionary and fresh
   factory copies; reject unregistered changes without consuming RNG.
2. Two-arrival bitwise C and CW_LSO parity against the old base host: output,
   diagnostics and full snapshots, including optimizer, gradients, buffers and RNG.
3. Two-arrival J=0 parity against corresponding C/C_LSO, including complete
   snapshots, diagnostics, calls and unchanged frozen model state.
4. Independent BCE+original-LSO loss and logit-gradient reference for C_LSO;
   unweighted consistency is explicitly distinguished from weighted BCE.
5. All five physical forward/reverse hooks, four auxiliary VJPs only for J,
   exactly one Adam step, and matching final global/native RNG sequences.
6. Both enabled J arms use the original shared filter and commit its returned
   displacement exactly. At the matched first update, base gradients and Adam
   state equal the corresponding non-J arm; auxiliary VJPs do not contaminate them.
7. Exact next-arrival snapshot/restore replay for all five arms, including
   outputs, diagnostics, full state and RNG. Enabled trajectories may naturally
   diverge from non-J trajectories after their first filtered update.
8. Zero structure evidence retains the native update on both backgrounds,
   while still charging the declared auxiliary calls.
9. Historical enabled-J admission still rejects a C-only configuration.

The existing joint/shared-span tests also check the independent dense rank4
formula, contraction, frozen R45 behavior, controls, RNG, zero evidence and
snapshot replay. Original CW_LSO and profile-boundary CPU tests are included.
These are synthetic Tiny-model checks with 512x512 inputs, not real-model GPU
qualification or segmentation efficacy.

Final suite: **30 tests PASS**, of which10 are new and20 retained regressions.
Runner reports12.902s test body; subprocess elapsed14.212061166763306s.
An initial10-test pass had9 successes and1 test assertion failure: the assertion
compared a tuple RNG representation to a serialized dictionary. The assertion
was corrected to compare matching representations and to verify serialized
global/native equality separately. No method, tolerance or parameter changed
for that correction. Initial elapsed8.215692043304443s is retained separately;
total CPU subprocess elapsed22.42775321006775s, GPU use0.

The pre-existing base-host tensor-to-scalar warning was observed and left
unchanged. The final test runner exits0. Anonymous test evidence is in
`reports/joint-lso-factorial-cpu/CPU_CHECKS.json` and `CPU_TESTS.txt`.

## Planned scientific interpretation, not execution authority

The agreed design is C-background 2x2 LSO x J plus the strong CW_LSO control.
Primary contrast is(C+J)-C; interaction is

    I = [(C+LSO+J)-(C+LSO)] - [(C+J)-C].

The proposed three original seeds/two orders/1,951 images imply30 fresh
trajectories and58,530 scorer rows. This task creates no such trajectories.
SEARCH/REVIEW are historically exposed. A negative interaction can indicate
diminishing returns or interference; it does not prove an identical internal
function. Positive results on C alone do not establish superiority to CW_LSO,
and this matrix does not identify the interaction in a W-enabled background.

The original planned primary gate is >=0.1pp, paired CI lower>0, both orders
positive,>=5/6 positive trajectories,imageweighted>=0,worst seed-mean cell>=-2pp.
The proposed practical-zero interval is[-0.05,+0.05]pp. These remain a design to
freeze before any later formal run, not grounds for selecting parameters using
the present CPU fixtures. GPU latency/VRAM, whole-matrix admission and real-model
behavior remain unmeasured for this new C-background wiring.
