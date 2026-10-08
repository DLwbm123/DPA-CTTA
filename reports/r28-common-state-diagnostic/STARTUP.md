# R28 A100 diagnostic startup

Status: RUNNING, no phase-B scientific result yet. Code:
`efb2bf758925d725f4550f63edfaef3fef7ed152`.

Phase A exact-selector audit is complete and published alongside this record.
Phase B freezes six ANCHOR reference trajectories, three seeds and two orders,
1,951 arrivals each, with96 prespecified SEARCH states in total. At each state,
one ANCHOR and four candidate predictions share the pre-arrival state/native RNG.
Five512px hard masks are scored offline only after all online workers retire.
Only the ANCHOR update is retained; labels never guide online state or sampling.
The local oracle is not a full-trajectory upper bound or independent confirmation.

Mechanical checks passed: exact tie replay, independent candidate noise, zero-
action parity, exact full-state rollback/replay and no probe effect on ANCHOR.
All three actual A100 smoke profiles passed after the infrastructure repair below.
Each smoke processed two arrivals/one probe,48 forwards and6 backwards. The first
three formal jobs are running on the authorized A100 physical GPUs0,1,2; three
more are queued. Process start identities and neutral command lines were verified.
Early throughput suggests roughly45–90 minutes for the replay matrix, then CPU
scoring; this is a load-dependent estimate, not a completion guarantee.

Initial profile attempts failed before any forward/backward because the copied
reference Git object database still used an alternate-object path on the original
host. The pinned reference was made self-contained using a local Git bundle; the
real model and upstream optimizer loaded successfully before the explicit retry.
All three failed receipts and16.403 GPU-worker seconds remain charged. The three
successful profiles charged53.001 GPU-worker seconds. The original start time was
preserved; no formal attempt was discarded or restarted, and no scientific setting
changed. The retry receipt plumbing is recorded in the code history.

The A100 uses PyTorch2.6.0+cu124 and batchgenerators0.25.2 in an isolated environment;
it is not a claim of byte-identical replay of the historical3090 environment.
No cumulative GPU cap; task/supervisory deadlines and storage guards remain.
Hourly monitoring is active. Stage C is conditional on the frozen phase-B headroom
gate and requires a separately frozen branch protocol before execution.
