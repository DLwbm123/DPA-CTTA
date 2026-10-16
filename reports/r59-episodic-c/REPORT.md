# R59 episodic C history ablation

Frozen gate: False

All six new trajectories and six reused C trajectories retained. Current-image update-before-predict remains; parameters and optimizer are reset before each new arrival while augmentation RNG continues. This jointly removes learned history and Adam history, not a separation of either cause. Development evidence only, no novel-method or independent-generalization claim.

EPISODIC_C macro Dice75.266493% versus C78.437246%: -3.170753pp (95% CI[-3.518103,-2.842312]),0/6 positive trajectories, imageweighted-2.479432pp and worst cell-14.164390pp. REFUGE_Valid OC improves+4.654881pp, but the full frozen gate fails. Current-image reset cannot replace continual C on these data; persistent state provides gains elsewhere. This does not isolate learned parameter history from Adam history.

All6 trajectories completed and retired exit0;23412 scored rows. Conservative GPU-process accounting6077.555248s includes failed initial attempts25.926024s and qualification8.864959s;2289.731615s wall from original T0. Original first-arrival parity failure was caused by omitted historical numerical settings; corrected deterministic policy passed first-arrival C parity. Old failed directory retained, new full run identity used; no valid checkpoint resume claimed.

Next: matched numerical-policy audit of the three R57/R58 stateless controls, preserving old reports and fixed scientific settings. No further reset schedule or mixing-weight tuning before resolving that confound.
