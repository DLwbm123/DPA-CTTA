# R66 attempt1: engineering failure, no efficacy result

All96 image/order pairs (672 branch predictions) were generated, but the final image integrity recheck was denied after current-image permission had been cleared. The worker exited1; no scorer was launched, no target labels were released, and the complete support audit was not persisted. Outputs remain private and preserved; they are not accepted as a complete evaluation.

Cost:101.4996743202 GPU-worker wall seconds and13.9644403458 prior CPU wall seconds. The per-image permission fix preserves the existing integrity checks and restores denial in finally. A generated two-file regression passed, including scorer denial. The first regression process failed only at guarded temporary-directory cleanup; explicit unlink/rmdir fixed cleanup and the subsequent process exited0. Both regression process costs are carried forward.

Retry1 uses a new directory with identical images, historical masks, geometry, random seeds, controls and gates. It carries all failed costs, leaving1698.50032568 seconds from the original1800 GPU-worker second cap. No efficacy score motivated this recovery. Current disposition: retry launched; scientific results pending. No source training or historical stream replay.
