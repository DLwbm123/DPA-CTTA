# R9 review repair status

- `execution_authorized=false`; no real data, GPU profile, remote run or monitor.
- LR policy fixed to `first_two_mean`: source seeds 20260924/20260925, one global LR per gradient arm.
- Scientific matrix remains 43 sources, 610 core plus at most 161 sensitivity slots; training and internal-score/end-round-release definitions unchanged.
- Four review repairs implemented; finite recovery allowance is enforced by the queue and persistent ledger.
- 25 R9 CPU synthetic regressions and 6 inherited metadata/isolation checks passed; plan arithmetic passed.
- Actual asset/GPU UUID/output-root bindings and real resource admission remain NOT ESTABLISHED.
- Real profile requires separate explicit bound authorization. A full round subsequently requires explicit authorization for the new exact SHA, assets, GPU UUIDs, output root and measured profile.

See `REVIEW_FIXES.md`, `VALIDATION.json`, `RECOVERY_PROTOCOL.json` and `LAUNCH.disabled.json`.
This is a code review release, not a runtime or scientific PASS.

Historical Screen24 asset protocol is now independent of the R9 FULL/16k runtime.
Actual metadata receipts passed read-only validation; binary binding/profile remain pending.
See `ASSET_PROTOCOL_FIX.md`.
