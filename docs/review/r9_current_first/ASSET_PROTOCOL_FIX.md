# Historical asset protocol / R9 execution separation

Parent candidate: `224ccb2eee70dcab049f3d37ffbd1d52fb459a57`.
User authorized the minimal repair and publication after the metadata binding blocker.

## Change

`assets.validate_metadata()` now uses the pinned historical Screen24 artifact protocol
`f065dd12da5b8de0a91f251c92d6e70bbbd5df816d589a227296564198dffa6a`.
It no longer takes that identity from the process-dependent R8 runtime protocol.
The source index and each prepared receipt's embedded identity are checked as well as
the existing receipt digests, source config, code, checkpoint and metadata references.
The constant is part of the tracked runtime inventory; a regression independently derives
it from the original Screen24 protocol/spec bytes.

R9 execution stays in FULL scope with 16000 steps. The existing guard against inheriting
the 4000-step Screen24 trainer remains. No R8 file, asset, scientific matrix, training
schedule, LR policy, scoring policy or recovery limit changed. Runtime deployment contexts
continue to describe the R9 execution definition; historical input identity is separate.

## Evidence

- 25 R9 CPU synthetic tests passed, including the new metadata → real SourceTrainer
  construction/fit-step regression with MAX_STEPS=16000.
- Wrong oracle binding, wrong index protocol and wrong prepared receipt identity reject.
- Actual historical JSON receipt bytes were retrieved read-only and checked against their
  original digests. The production validator accepted them using a local transport mirror
  under FULL scope. No identity predicate was patched. This verifies metadata only.
- No medical image, mask or checkpoint binary was opened; optimizer/RNG payload and
  gradient-scale extraction remain pending authorized binary validation.

`execution_authorized=false`. No GPU profile, full round or hourly monitor was started.
Any profile package referring to 224ccb2 must be rebound to this new commit/inventory
before separate explicit REAL_DATA_PROFILE_ONLY authorization. Full-round authorization
remains separate and requires real resource admission and the final bindings.
