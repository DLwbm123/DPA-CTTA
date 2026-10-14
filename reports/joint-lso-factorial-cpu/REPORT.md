# Five-condition LSO x shared sensitivity wiring: CPU result

**PASS_CPU_ONLY:30/30 tests**, including10 new factorial checks and20 existing
joint/shared-span, strong-control and profile-boundary regressions. No GPU
profile or formal scientific experiment was started. GPU cost0.

| Condition | CPU execution | Physical calls per arrival |
|---|---|---|
| C | PASS, bitwise old C | 8F/1B/1Adam |
| C_LSO | PASS, unweighted BCE+0.1 original LSO reference | 8F/1B/1Adam |
| C_JS_SHARED | PASS, J=0 bitwise C | 10F/5B/1Adam enabled |
| C_LSO_JS_SHARED | PASS, J=0 bitwise C_LSO | 10F/5B/1Adam enabled |
| CW_LSO | PASS, bitwise original strong control | 8F/1B/1Adam |

Disabled parity checks cover two arrivals, outputs, diagnostics and complete
snapshots: parameters, buffers, gradients, optimizer, native bookkeeping and
global/native random state. All five replay their next arrival exactly after
snapshot/restore. Both enabled J branches retain their base gradients and Adam
state on the matched first update; each performs exactly four auxiliary VJPs
and a single Adam step. Actual forward/reverse/optimizer hooks independently
check the counters. Private Sensor RNG and acquisition probes preserve the
global/native augmentation sequence across the five arms. Zero evidence keeps
the exact native update while charging the declared auxiliary computation.

J is fixed R46 SHARED (rank<=4,strength1,ridge0.01), not R48 DESCENT. C-background
LSO retains its original detached reliability weighting; its base BCE is
unweighted. CW_LSO and all historical entry points/guards remain untouched.
Only `joint_sensitivity.factorial` admits the new configurations; it exports
Host/candidates and has no experiment runner, dispatch, monitor or launch entry.

Final body time12.902s, subprocess14.212061166763306s. First qualification had
one test-only RNG-format comparison failure, corrected without changing the
method; its8.215692043304443s and receipt remain recorded. Total CPU subprocess
time22.42775321006775s. The existing base-host tensor-to-scalar warning remains.

See [the wiring specification](../../docs/JOINT_LSO_FACTORIAL_CPU.md),
[machine-readable checks](CPU_CHECKS.json) and [test list](CPU_TESTS.txt).
These generated Tiny-model checks establish wiring properties, not real-model
GPU latency/VRAM, Dice benefit, LSO overlap, anatomical protection or independent
generalization. Those measurements require separately authorized later work.
