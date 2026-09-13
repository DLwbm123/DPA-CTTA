# Actual development failures and follow-up

Logs are retained, including failed attempts. Public copies replace workspace/home/interpreter/temp path prefixes with neutral labels; CSV line endings and one trailing log whitespace are normalized; test names, stack frames, failure facts, counters and timings remain. No real-data log is included.

| Attempt | Observed result | Diagnosis / action |
|---|---|---|
| Package reference 01 | 24 tests passed | Untouched supplied CPU kernels/tests, before host implementation |
| CPU 01 | 36 tests; two errors | U_RAND called R1's positive-index seed helper with 0; fixed its new namespace to constant index 1. Parity test used nonexistent `parameters` instead of existing `affine`; corrected the test field |
| CPU 02 | 36 tests; one error | The ready-state fixture attempted visit 1 merges into a bank already used by the cold step. Reset only the synthetic fixture's bank state before loading its fabricated prior; no formal initialization changed |
| CPU 03 | 36 tests passed | Included real architecture with random weights, all-family cold/ready execution and 38-update physical accounting |
| CPU 04 | 94 tests; three failures | The CPU-only asset guard compared a resolved `/private/var` fixture path with an unresolved `/var` root. Resolve both sides; retained denial outside the fresh procedural fixture tree. No registered asset read occurred |
| Execution 01 | 3 tests passed | All 85 synthetic job bindings, separate primary/secondary closeout, missing evidence/truncation and disabled-launch rejection |
| CPU 05 | 97 tests; one failure | Existing persistent-EIO supervisor test reported `all_owned_returned=false` and `all_owned_reaped=false` once. Its test fallback cleaned owned children; outsider remained alive. Other tests passed |
| EIO diagnostic 01/02 | 3 + 6 independent probes | Every probe observed intended EIO, all four owned children returned/reaped, outsider alive. Instrumented cleanup reported no secondary cleanup exception. The earlier intermittent failure's cause remains unresolved; no speculative supervisor change was made |

Other code-review fixes before freezing: preserve active model construction RNG as in old C while isolating only the extra measurement clone; count VJP/loss/Adam invocations before their calls so interrupted calls are not silently omitted; avoid computing the S-only full descriptor in T; record density snapshot versions separately. No scientific seed, threshold, rank, learning rate, half-life, stream length or arm selection was changed.

The first metadata CLI attempt omitted two required environment variables and exited with `KeyError: REG_FILE` before reading anything; the corrected metadata invocation succeeded. R3 metadata retrieval used `python3` and succeeded.

The final CPU suite and any final targeted checks are listed in `IMPLEMENTATION_REPORT.md`; do not replace this history with only the last green log. The single intermittent inherited EIO ownership-test observation remains a review item even if the final suite passes. This handoff does not assert that the anomaly has been explained or repaired.
