# R2 method provenance

Authority: user-supplied `R2_EXPERIMENT_PLAN_AND_CODEX_PROMPT.md` and its byte-frozen `R2_SCIENCE_PROPOSAL.json`. This is an implementation proposal awaiting external review, not execution authorization.

| Component | Source and exact reuse | R2 change |
|---|---|---|
| C / model / augmentation / optimizer | Reviewed R1 runtime `54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b`; original `b1_host.Host` and R1 Host sequence | No changes; new Host overrides extra criterion and memory only |
| Feature grid, reliability, quotas and token IDs | R1 `region_memory.py`: 32x32 centers, seg_head input 32D, six weak hard agreements, probability <=0.1 or >=0.9, up to 32 reliable tokens per region | Direct inheritance; loss still uses all hard-partition tokens |
| Cumulative statistics | R1 `StreamingSubspace`, CPU float64 centered-scatter merge | R2_D uses this exact class, not weighted code with rho=1 |
| Absolute reconstruction | R1 `projection_residual` with detached centered-energy denominator | R2_E calls this exact function |
| Target structural alignment | Supplied SaTeen PDF, section 3.2, equations describing reliable feature collection, incremental PCA/SVD and subspace reconstruction | Mechanism inspiration only. This implementation uses the R1 covariance/eigh variant, not an exact reproduction of SaTeen IPCA |
| Paired complement objective | R2 task proposal | Unit(fs) - stopgrad(unit(f0)); complement of old U; d/(d-k_actual) scale; feature sum then token mean then active-region mean |
| Exponential forgetting | R2 task proposal | Every global visit multiplies W/M2 by rho, Q by rho squared. Stable weighted merge and covariance divisor W-Q/W; half-life 128 visits |
| F matched control | R2 task proposal | Full paired squared feature sum with scale 1; same exponential shadow memory and readiness rule. PCA direction affects diagnostics, not the loss |
| Shuffled control | R1 quota-preserving independent memory/loss shuffles | R2_DE_S keeps original R1 seeds/salts. A shuffled token maps to identical spatial u on both f0 and fs |
| Asset/evaluator/process cleanup | Reviewed R1 assets, `current`, ownership supervisor and atomic evidence publisher | Imported unchanged; R2 supplies its own matrix, counts, scalar replay and bound historical controls |

Five arms: E=(absolute, forgetting, region), D=(paired complement, cumulative, region), DE=(paired complement, forgetting, region), F=(paired full, forgetting shadow, region), DE-S=(paired complement, forgetting, shuffled).

The paired objective and exponentially weighted statistics are new proposed choices here. Neither the task package nor CPU tests prove improved Dice, theoretical guarantees, significance or superiority. Common lambda does not prove gradient fairness. Identical F/DE rules do not imply identical realized token selections after their models diverge.

No source RGB, source masks, source proxies/prototypes, extra pretrained components, offline updates or source-domain replay are used. Stage I uses programmatic random weights and synthetic pixels only. The supplied PDF was read for provenance and is not bundled or required at runtime.
