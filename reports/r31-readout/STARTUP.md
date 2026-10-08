# R31 startup: final-classifier adaptation

**RUNNING; no performance results available.** All three A100 profiles passed; the fixed 18-stream matrix is admitted and first formal workers have begun. Original code commit747ba862df6f9f425dea3a113406f0be172701eb. Online labels remain closed until every formal worker retires successfully.

R30 found persistent C better than episodic adaptation, while delayed strength and periodic reset interventions provided no supported gain. R31 tests a distinct hypothesis: does enabling class-specific channel recombination in the existing final classifier help persistent consistency learning? C_HEAD adds only66 existing trainable scalars at fixed LR1e-5; BN remains1e-4. Fresh C_CONT and ANCHOR references run across the same3 seeds and2 orders. This is a limited mechanism test, not a new validated method or an LR search.

Generated-input checks passed for nonzero head gradient/update, exact frozen encoder, Adam ownership/clocks, C parity, state restoration and replay. Synthetic18-stream scoring coverage and original48-stream scoring regression passed. Each GPU profiled8 arrivals per arm and verified8 exact C_CONT plus8 exact ANCHOR reference replays. Head drift and gradients were positive on each GPU. No cumulative GPU-hour cap; wall/storage protections remain.

Conservative measured projection: 7.703 formal GPU-worker hours; online finish with20% wall reserve around 2026-10-09T09:42:10.597266+08:00, then CPU scoring. Profiles include extra reference replay and initialization, so this is a planning bound, not a guaranteed completion time. Peak measured reserved memory: 1.033GiB.

Prelaunch copy metadata failed on NFS; independent ordinary copy and read/write/fsync probes passed. Deployment used plain content writes. A system-interpreter mismatch and an occupied neutral launcher filename were caught before any campaign launch; the registered environment and a unique neutral launcher fixed both. No experiment worker was retried or restarted. Historical runs remain intact.

Primary gate requires C_HEAD to improve both references by at least0.3pp with positive effects in both orders, at least5/6 positive trajectories, and worst seed-averaged domain/channel/order effect at least-2pp. SEARCH and legacy REVIEW are development-exposed; REVIEW remains descriptive. All negative cells and costs will be delivered when complete. No images, labels, masks, identities, source/adapted weights or private paths are published.
