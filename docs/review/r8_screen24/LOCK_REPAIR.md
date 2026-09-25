# Launch lock repair — 2026-09-25

Initial launch at code 57e3764 stopped before any oracle forward/backward. All three workers and queue blocked in nlmclnt_wait; a separate three-process NAS flock probe reproduced the timeout. The queue and owned workers were terminated, preserving every output and reservation.

Repair: use a user-owned, local lock inode for the ledger, require the recorded host on every operation, and keep all ledger data/checkpoints on NAS. This changes synchronization only. All workers are restricted to this host's physical GPUs 5/6/7. Do not mix hosts.

The initial shard-0 zero-step snapshot is preserved and validated before identity migration to the repaired code. Its missing atomic sidecar is reconstructed only in the recovery copy from the complete original tensor file. Shards 1/2 never entered execution or created scientific outputs. The replacement queue carries prior attempt counts for all three, disables any further infrastructure retry for those jobs, and retains the initial 10.36 GPU-hour reservations in aggregate cost. Source protocol, graph, RNG in the available snapshot, latent initialization, optimizer, seeds and ordering are unchanged.
