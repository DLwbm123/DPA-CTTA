# R7 Stage I review index

**R7_IMPLEMENTATION_READY_FOR_REVIEW** — A, B, C and the shared implementation passed the fixed-SHA CPU acceptance. This is not external review PASS and does not authorize execution.

- [Delivery/status and all boundaries](DELIVERY.json)
- [Exact implementation SHA](IMPLEMENTATION_SHA) · [implementation binding/fingerprint](IMPLEMENTATION_BINDING.json)
- [Immutable science file hashes](SCIENCE_BINDING.json) · [24-entry manifest verification](MANIFEST_VERIFICATION.json)
- [A PSF review](A/REVIEW_INDEX.md) · [B RCA review](B/REVIEW_INDEX.md) · [C RBE review](C/REVIEW_INDEX.md)
- [Shared implementation/interface contract](IMPLEMENTATION.md) · [state and label isolation](STATE_AND_LABEL_ISOLATION.md)
- [Test coverage and limitations](TEST_COVERAGE.md) · [actual CPU results and retained failures](CPU_RESULTS.json)
- [Final local log](logs/final-local.log) · [final server log](logs/final-server.log) · [independent local reference log](logs/final-math-local.log)
- [Measured numerical differences](REFERENCE_ERRORS.json) · [latent matrix/module cost](LATENT_COST.json)
- [24 / 9 / ≤12 metadata matrices](DRY_RUN.json) · [source budget including extra forwards](SOURCE_BUDGET.json)
- [Narrow implementation patch against baseline](implementation.patch)
- [Original master prompt](input/prompts/00_MASTER_CODEX_PROMPT.md) · [source attribution and limits](input/references/PROVENANCE.md)

The47 implementation tests passed on local macOS/PyTorch2.6.0 and server Linux/PyTorch2.2.1, zero final failures/errors/skips. The22 independent NumPy checks passed separately in both environments. Each final implementation suite made464 backbone forwards,36 backward calls,2 VJPs,24 Adam and6 AdamW calls on procedural CPU inputs. These are measured synthetic computations, not real source/target execution.

All three groups include complete source tensor-fit/calibration/validation and image-only online interfaces. FULL and STATIC are separate same-budget models. The true source manifest/split/checkpoint/trained-weight bindings remain PENDING; this blocks future source preparation until separately bound and authorized. It does not authorize target substitution.

The fixed implementation is `82aca2080bb80c019b0d39b4c95d21a6ba2cff3d`. Evidence-only publication may have a later SHA; no tests are claimed to have run on that later documentation commit. Raw test originals remain private; public logs replace only account/filesystem paths. Old C, historical science, old gates and R1–R6 results are unchanged. No GPU was queried, no real image/mask/checkpoint was read, and no real execution receipt or review PASS was generated.

Stopped for external review. SOURCE_PREP, TARGET_SCREEN, TARGET_MECHANISM and TARGET_EXTENSION are all NOT_RUN.
