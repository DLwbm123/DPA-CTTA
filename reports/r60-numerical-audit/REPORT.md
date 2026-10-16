# R60 matched numerical-policy audit

Fixed stability gate: True

The three controls were replayed using historical C deterministic, TF32-disabled policy. This is numerical sensitivity validation, not another hyperparameter search or independent replication. Old predictions and results are retained. Interpret changes using NUMERICAL_SENSITIVITY and corrected per-stage reports; a small aggregate difference does not imply bitwise equality.

| Control | Matched macro Dice (%) | Shift (pp) | Maximum absolute cell shift (pp) | Changed images /1951 |
|---|---:|---:|---:|---:|
| SOURCE |68.200581|-0.002890|0.007833|1931|
| BATCH |75.079420|+0.000244|0.001386|1890|
| MIX |75.801772|+0.000613|0.002127|1909|

Both registered decisions remain [R57:true,R58:false]. The prespecified stability gate passes; matched-policy results support the same aggregate conclusions, resolving the identified numerical-policy sensitivity concern within these bounds. This does not assert identical predictions:16,240/10,107/10,884 channel-pixels changed for SOURCE/BATCH/MIX out of1,022,885,888 each. Use the matched-policy stage tables for subsequent comparisons to C. This removes this particular numerical concern, not development exposure or causal overinterpretation.

Both GPU stages retired exit0;5853 control-image predictions complete, stage scoring35118/46824 rows. GPU worker time including initialization429.464548s; total wall500.368618s. Old inputs/results preserved; no scientific retuning.

Historical coverage review found R30 already includes source/current-BN/episodic/reset controls, and R33 tests matched-age optimizer-state exchange. R57/R59 therefore add numerical/coverage confirmation, not a new mechanism discovery. No further reset, optimizer-donor or mixing-weight sweep follows from this audit. See the repository research-state ledger for closed routes and required novelty checks before the next experiment.
