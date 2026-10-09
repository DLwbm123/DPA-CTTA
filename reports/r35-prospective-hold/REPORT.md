# R35 prospective update pause action value

Status: **COMPLETE**.

Primary H256 SEARCH: HOLD−UPDATE -0.765089pp; oracle−UPDATE +0.234880pp; oracle−best-fixed +0.234880pp. Best fixed action: UPDATE.

Next decision: **STOP_REGISTERED_PROSPECTIVE_HOLD**. This is a resource-allocation decision, not permission to start RL or claim a deployable selector.

HOLD256 preserves the current learned parameters and Adam state, performs current-image BN inference without updates, and advances arrivals and paired RNG. UPDATE reuses the freshly verified native trajectory. H256 is primary and H64 auxiliary. This tests future freezing, not retrospective rollback. Each branch starts independently; no full-stream controlled policy is evaluated.

The primary mean weights windows within each seed/order and then the six groups equally. Origin/future-domain strata, negative cells and all denominators are published separately. This is not a balanced four-domain or full-policy efficacy estimate. Windows overlap, contents repeat, SEARCH/legacy REVIEW are development-exposed and patient/ROI provenance is unknown. No independent/clinical claim or label-free observability/RL claim is supported by an oracle alone.

GPU-worker time including profiles/failures: 3.070191h. Public delivery is verified separately.

## Completion and registered gates

Completed on 2026-10-09 at 21:31:14 Beijing time. All six online workers and the CPU scorer retired with exit code 0; the label-release barrier passed. There were no profile, online or scoring failures. Preparation dependency and RNG assertion failures occurred before profiling/T0 and remain recorded in PREPARATION_NOTES.json; they are not scientific run failures.

| Endpoint | UPDATE Dice (%) | HOLD Dice (%) | HOLD−UPDATE (pp) | Oracle−best fixed (pp) |
|---|---:|---:|---:|---:|
| H256 SEARCH, primary | 76.393767 | 75.628679 | −0.765089 | +0.234880 |
| H64 SEARCH, auxiliary | 77.013099 | 76.818368 | −0.194732 | +0.064368 |
| H256 legacy REVIEW, descriptive | 77.145130 | 76.344318 | −0.800812 | +0.229304 |
| H64 legacy REVIEW, descriptive | 78.000272 | 77.797417 | −0.202855 | +0.074672 |

The exported role name SEALED_REVIEW is legacy terminology: these data are development-exposed, not an independent sealed evaluation. H64 and H256 have different eligible windows; their mean difference does not establish recovery speed.

For primary H256 SEARCH, HOLD−UPDATE is −0.987598pp in order 0 and −0.542580pp in order 1, with 0/6 positive seed/order groups. The fixed-HOLD gate fails the ≥0.3pp mean, both-positive-order and ≥5/6 positive-group requirements. Oracle−best-fixed is +0.172385pp and +0.297374pp within the respective orders: both exceed the per-order 0.15pp condition, but the overall +0.234880pp fails the required 0.3pp condition. The conditional gate therefore fails as well.

Some registered origin strata are positive: REFUGE_Valid/order 0 has H256 SEARCH HOLD−UPDATE +0.397578pp, while order 1 is −0.004829pp. These windows can cross domains and do not establish a deployable domain trigger or override the primary gate. All origin, future-domain and channel cells are retained, including 857 negative aggregate cells. No favorable stratum replaces the frozen endpoint.

Coverage is 90 paired decisions, 78 complete H256 windows and 90 H64 windows; 12 H256 windows are truncated to 223 or 95 arrivals and excluded from the primary endpoint. Scoring produced the registered 38,430 action/content observations. Formal execution used 11,706 backward/Adam calls, 21,876 HOLD arrivals and 115,524 model forwards; six additional profiles used 432 updates and 3,504 forwards. CPU scoring took 188.649941 worker seconds. Aggregate coverage and successful worker retirement were checked after completion; RESULT_VERIFICATION.json records the scope. No new rescore or experiment was run.

The registered decision closes only HOLD256 on native-C states at this fixed grid. It does not prove all pause durations, phase controls or RL ineffective. No horizon scan, selector training, new experiment or recurring monitor is authorized by this result.

Public artifacts contain code/protocol references, aggregate tables, costs and audits. Raw images, labels, prediction masks, weights, learning states, private per-content scores and operational paths remain private.
