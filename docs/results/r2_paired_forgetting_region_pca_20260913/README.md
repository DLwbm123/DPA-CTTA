# R2 paired forgetting / regional PCA: completed results

Completed **2026-09-13 20:14:15 China Standard Time**. All 20 selected trajectories (5 new arms × 4 fixed orders × 1,951 visits) completed: **39,020 formal new scoring records**. The frozen CPU result recomputation passed; the final launcher and all six final child processes exited 0.

**Decision: retain C.** No new arm reaches the frozen descriptive resource reference of at least +0.5 percentage points over C with improvement in at least 3/4 orders. R2_E is the best new arm (+0.1754 pp, 2/4 positive orders), but improves on C_PCA_REGION by only +0.0066 pp. R2_DE is below C by 0.0202 pp and does not support the matched necessity claim. These are development observations, not statistical significance or clinical claims. No further experiment is authorized or launched by this report.

## Primary endpoint

`remaining_dev`: 1,695 shared exposed contents. Average OD/OC Dice, then weight four domains equally and four orders equally. Dice values below are percentages; differences are percentage points. C and C_PCA_REGION reuse 15,608 historical scoring records without rerunning those controls.

| Arm | OD Dice % | OC Dice % | Macro Dice % | Delta C (pp) | Positive orders vs C |
|---|---:|---:|---:|---:|---:|
| C | 87.3482 | 69.8741 | 78.6112 | +0.0000 | historical control |
| C_PCA_REGION | 87.2942 | 70.2657 | 78.7799 | +0.1688 | historical control |
| R2_E | 87.3095 | 70.2635 | 78.7865 | +0.1754 | 2/4 |
| R2_D | 87.3402 | 69.8411 | 78.5906 | -0.0205 | 0/4 |
| R2_DE | 87.3396 | 69.8423 | 78.5910 | -0.0202 | 0/4 |
| R2_F | 87.3462 | 69.8256 | 78.5859 | -0.0253 | 1/4 |
| R2_DE_S | 87.3324 | 69.8154 | 78.5739 | -0.0372 | 0/4 |

R2_DE − R2_D = +0.0003 pp; R2_DE − R2_E = −0.1956 pp; R2_DE − R2_F = +0.0050 pp; R2_DE − R2_DE_S = +0.0170 pp. The descriptive factorial interaction is −0.0063 pp. Full order tables and all matched comparisons appear in [the runtime report](RUNTIME_REPORT.md).

R2_E's domain mean gains over C are REFUGE −0.2258 pp, ORIGA +0.4320 pp, REFUGE_Valid −0.5085 pp and Drishti_GS +1.0037 pp. Drishti_GS has only 37 remaining development contents but one quarter of the domain weight. Four orders reuse the same contents and are not independent patient samples. The aggregate's `risk_warning: false` only describes its specified numerical threshold; it does not remove the REFUGE_Valid OC decline or establish safety.

## Execution provenance and recovery cost

This is a user-authorized IO continuation, not an uninterrupted single-commit execution. The science and registration bindings stayed fixed; all abandoned artifacts remain preserved outside this public release.

| Attempt | Runtime commit | Selected completed trajectories | Interrupted scored prefixes |
|---|---|---:|---:|
| Original | `0d515328a6cc42d8e0c6a458265b41e41c454fa6` | 15 | 576 |
| Continue 1 | `66eea7e880e16d4d25efa4edabc9d8ad59ff175d` | 0 | 1,048 + 1,053 |
| Continue 2 | `1560f28d41dda5665404ececbefc4f0a7c69d192` | 1 | 1,408 |
| Continue 3 | `e0d3f6634bfa7349b214942fc01f64d57a65470f` | 4 | 0 |

Failures arose from live directory capacity scans encountering files removed or atomically replaced on NFS. The shared scanner now tolerates ENOENT enumeration/stat races while propagating other IO errors. The final continuation carried 16 completed trajectories with their original provenance and completed only the remaining four. See [repair history and validation](../../review/r2_io_continuation/LIVE_DIRECTORY_REPAIR.md).

Formal selected scoring used 312,160 forwards and 39,020 backward/Adam updates. The 4,085 interrupted scoring records are excluded from the formal table. Counting those prefixes and four smoke batches, actual physical cost is bounded at **345,736–345,752 forwards and 43,217–43,219 backward/Adam updates**. Two terminated workers may each have one unrecorded in-flight visit. Thus the actual execution exceeded the original compute budget; it is not presented as an exact-budget run.

Cumulative active worker time was 26,373.93 seconds; summed attempt wall time was 13,713.01 seconds (excluding between-attempt repair gaps). Final CPU closeout took 51.12 seconds. Per-trajectory latency, peak GPU allocation and source bindings are in the execution audit.

## Public evidence and reproduction boundary

- [Full aggregate](public_aggregate.json): all subsets, orders, domains, OD/OC, paired tails, ASSD and matched comparisons.
- [Mechanism and cost supplement](MECHANISM_AND_COST.json): per-bank W/Q/count/rank/version, readiness, energy/loss/update distributions and missing-region counts from existing scalar records.
- [Execution audit](EXECUTION_AUDIT.json): all 20 selected trajectory bindings, completion counts, timings, device models and continuation overhead.
- [Runtime report](RUNTIME_REPORT.md): frozen CPU-generated summary and focus-domain comparisons. Its smoke block describes the final attempt only; the continuation accounting covers earlier attempts.
- [Export check](EXPORT_CHECK.log): scalar-only export passed with 39,020 selected records and CUDA uninitialized.
- [Original implementation and protocol](../../review/r2/REVIEW_INDEX.md), [export script](../../../scripts/report_r2_results.py).

The export reads existing scalar records and validates selected bindings/counts/completion; it does not run a model or repeat metric evaluation. Use the final runtime commit's `src` on PYTHONPATH and set RESULT_DIR, CONTROL_DIR and a new REPORT_DIR to run the export script against authorized execution artifacts. Reproduction of the original experiment requires the separately authorized checkpoint, data, dependency environment and historical control artifacts described by the protocol; they are not bundled here.

Public files exclude sample/group identifiers, raw images/masks, per-content records, model weights, credentials, private server paths and device UUIDs. Scalar replay checks W/Q/counts/refresh, not feature covariance or ASSD geometry. F includes shadow PCA cost; matching its rule to DE does not guarantee identical selected tokens after adaptation. Offline updates remain zero, checkpoint-only, source-data prohibited, seed 20260907.
