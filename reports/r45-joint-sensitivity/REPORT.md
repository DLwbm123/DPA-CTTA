# R45: joint acquisition/anatomical sensitivity on CW_LSO

Status: **COMPLETE**. Frozen primary: CW_LSO_JS.

Frozen jointly sensitive actual-Adam proposal filter, strength1/ridge0.01, two acquisition probes and four auxiliary VJPs. Keep original CW_LSO loss and Adam moments. Local structural proxy and metric contraction do not establish anatomical protection. No GC/GP/GDP/SCNP/SE/RL.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO | 79.0456 | 78.4130 |
| CW_LSO_JS | 79.0387 | 78.4065 |

Primary SEARCH gain against C/W each>=0.3pp and against CW_LSO>=0.1pp. All three comparisons require both orders positive,>=5/6 trajectories positive,imageweighted>=0,worst seed-mean cell>=-2pp. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 8.033/48h; CPU-worker 0.149h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, output fusion or scientific retries. Only the registered two acquisition probes are added. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Complete result and failed frozen gates

SEARCH JS-minus-C/W is +0.347168/+0.949080pp, both orders positive and6/6 trajectories positive for both comparisons. Imageweighted changes are +0.452026/+0.302575pp. These retain the foundation's gain; they do not identify a beneficial JS increment.

JS-minus-CW_LSO is **-0.006898pp**, coupled-content bootstrap95% interval **[-0.007949,-0.005819]pp**, with both orders negative (-0.007337/-0.006460pp),0/6 positive trajectories and imageweighted -0.006179pp. The difference is small in practical magnitude but consistently adverse in this exposed development matrix. It is neither an improvement nor independent evidence of general harm.

Worst seed-mean SEARCH cells versus W remain -5.138166pp (order0 REFUGE_Valid OC),-4.896469pp (same domain/order OD),-3.525033pp (order1 REFUGE OC) and-2.089187pp (order1 REFUGE_Valid OD). The OC worst cell improves only0.016969pp relative to the paired CW_LSO deficit -5.155135pp. Versus C the worst cell is -0.785277pp; versus CW_LSO the worst cell is -0.029658pp (order0 ORIGA OC). All adverse cells remain in ALL_NEGATIVE_CELLS. Primary passes the C comparison but fails W protection and CW_LSO increment gates; full success=false. No criteria or primary substitution changed.

Legacy REVIEW macro Dice78.4065 versus CW_LSO78.4130 is descriptive only. It is not an independent validation set and was not used to choose follow-up parameters.

## Mechanism and resource interpretation

On all6,102 SEARCH observations, the acquisition span has rank2 (no rank-zero/one fallback); across all11,706 formal arrivals rank2 remains present. Mean filter eigenvalues are0.906365/0.993027. Structural-evidence means OD/OC are0.009551/0.006947. The branch was active and its physical10F/5B/1Adam schedule was enforced by the host.

Mean per-arrival filtered/proposal Adam-metric norm ratio is0.998812 (mean reduction0.118756%); median0.999901,5th percentile0.994994,min0.883749. No ratio exceeds1.00001. JOINT_MECHANISM_SUMMARY reports all role/domain aggregates. The small whole-update norm attenuation is an observation, not proof that the identified span or evidence is wrong; norm shrinkage does not measure update rotation or anatomical protection. No counterfactual geometry/clinical causality claim is supported.

The isolated earlier real-model resource profile measured JS/base latency1.44-1.50x, allocated memory+38.91%,reserved+43.59%. That controlled short comparison supports additional compute cost; the full-round shared-GPU condition runtimes are not an exclusive-device speed comparison. Current JS has no accuracy gain to justify that extra cost.

## Verified completion and costs

All24 unique trajectories completed1,951 arrivals and all24 independent CPU scores completed1,951 rows:46,824 rows. All52 attempts (4 profiles,24 online,24 scoring) COMPLETE, no failures; every online attempt ended before the first scorer started. All receipt PIDs, watch and supervisor retired. GPU0/1 only. Online label/source-image reads remain0. Original T0 and budgets retained.

Round cost: GPU-worker 8.033344899h, scoring CPU-worker 536.594549s, preparation CPU10.775102s/GPU0. Execution wall 4.386453h. Ten closed-round GPU-worker cost 79.553766393h includes the unscorable R39 resource failure; adding the isolated JS resource profile gives 79.574425464h. Existing old ledgers are unchanged.

See PAPER_READINESS.md for the evidence-backed candidate paper direction and proposal-only minimal supplements. No new experiment, parameter scan, new data access or automatic next round was started.
