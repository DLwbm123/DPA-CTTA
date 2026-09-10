# B2 interval consistency — launch report

B2_RUNNING. This is a launch audit, not a completed experiment or scientific result.

Execution commit: `35d9d26c5554a23d81713f610d6ae53d93f54774`. Snapshot UTC: 2026-09-10T13:33:46Z.

- Existing remote Python 3.10.6 / Torch 2.2.1+cu121: all 9 CPU tests passed, no skips.
- Single GPU smoke passed: 20 Adam, 160 forwards, 20 backwards. C versus zero radius matched logits, BN affine, Adam and exact RNG; U/S/I were finite and had active procedural pixels.
- GPU 7; one detached launcher, no automatic retries. 63 formal records had completed at this snapshot. No immediate failure.
- Scheduled U/S/I × 2 orders × 1,951 groups = 11,706 formal records/Adam/backwards, 93,648 forwards. Old C/G and P2 controls are reused, not rerun.
- Full run continues independently of the Codex/SSH session. The launcher runs independent CPU scalar reconstruction and report generation after all six trajectories. No ongoing monitor was created.

Code/config/contract/tests and this deidentified startup evidence are public. Asset paths, identities, per-image logs and receipts stay private on NAS. No images, masks, logits, probability maps, model or optimizer snapshots are saved by B2. Final aggregation and scientific interpretation are pending; no candidate success claim.
