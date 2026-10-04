# R17 background startup

Execution source `2f0ecba911d83a697863ed695cd28677c8f8e47a`; config `2d3cc305721a362fa24d120afdb3068415f27828fdecaca9f193692100f0a44a`.

Native qualification passed with 22 source accesses, zero target accesses: exact lambda-zero logits/BN/Adam/RNG parity and exact integrated next-image snapshot continuation. CPU objective/ablation tests: 3 passed. The initial CPU runner discovered zero tests and was corrected; that output was not treated as a pass. Two deployment-template/scope import errors were corrected before GPU execution; no scientific inputs or parameters changed.

Preflight actual cost: 41.410606384277344 GPU-worker seconds, 187 backbone forwards, 41 backward calls and 30 optimizer steps (includes the paired native controls and discarded mechanical head fits), 14 head forwards, no VJP. Original T0 unchanged.

Full six-trajectory matrix admitted at 1.3 safety factor: source projection 2770.041965 seconds, target projection 13255.066314 seconds, total projection including preflight 16066.518885 GPU-worker seconds. This is a cost forecast, not measured target performance or a wall-clock guarantee.

At startup verification the source phase was running, 32/512 captures, with no log errors. Neutral watchdog, supervisor and worker command lines and the actual assigned physical GPU were verified. Target processing has not begun. Native adaptation and source head fitting are real optimization; the head is frozen only during target processing.

See [frozen protocol](../../docs/protocols/R17_GRATA_INTEGRATED_V1.md). Three new arms: ordinary Adam + context correction, GraTa + context correction, GraTa + equal-capacity logits-only correction. Each has two complete orders; C0, DS, old static correction and intact GraTa are sealed historical references. All are exposed development data, not independent confirmation.

Hourly monitoring is active for this single round. After terminal scoring, all results and costs will be reviewed and published, then monitoring ends. There is no automatic successor.
