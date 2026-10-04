# R18 quick full-RL + GraTa results

Status: COMPLETE. All six new full trajectories and independent CPU scoring are complete. Execution `e2da0680a325a38f0947e101cc84eaa00fd53960`. Frozen config `a5e9c7f4f1f847509c62c0a21332d62f2f8eff8fe5d79cb88595382d6e6a6d31`.

The unchanged full RL policy plus GraTa achieved 74.155734% domain/channel-macro Dice versus GraTa 77.229693% (-3.073959 percentage points). The predeclared priority signal requires at least +0.5 points and both orders positive; it was not met.

| Condition | Mean macro Dice (%) | Delta vs GraTa (pp) |
|---|---:|---:|
| C0 | 75.079420 | -2.150273 |
| G | 77.229693 | +0.000000 |
| RL_ORIGINAL | 72.145923 | -5.083770 |
| FIXED_GRATA | 75.923522 | -1.306171 |
| RL_GRATA | 74.155734 | -3.073959 |

Worst combination-minus-GraTa domain/channel: Drishti_GS/OC, order1, n=37, -11.153544 pp. Losses larger than2 pp meet the predefined risk flag.

Each new condition has two complete1951-visit trajectories, each1695 primary observations. In total11706 new visits and10170 primary scores; C0/G reuse four exactly matched sealed historical full trajectories. Main scores average the four domains and two channels equally, then average the two orders. `main.csv`, `domain-channel.csv` and `paired.csv` report all orders, OD/OC, ASSD valid/undefined denominators and image-weighted paired outcomes; macro and image-weighted effects should both be read.

RL_ORIGINAL and RL_GRATA use the SAME historical R10 POST GR_RET_EMA actor, frozen B64/global carrier, FiLM and full10-action use/write with cross-image m/q/h memory. GraTa completes its native current-image BN update before exact BN mirroring to the FiLM model; zero-FiLM logits equal native GraTa at every image. The policy then chooses FiLM use and memory write. GraTa evolution is independent of these actions. FIXED_GRATA fixes gain0.8, residual0, write0.5. Native GraTa seed20260907 and policy seed20260924 have distinct roles. Actor, carrier and non-BN weights are frozen; BN adapts by design. No target reward or online policy fitting.

No new source retraining was used. The original source-preparation plan was cancelled before formal policy training or any target access;17 completed source-cache episodes remain private. Earlier qualification included discarded cost-probe updates and is fully billed. Those source caches do not enter the quick screen. New source-fit/validation metrics and WARM controls are NA/not registered. Learned-versus-fixed is a deployment diagnostic, not isolation of RL training from the actor's earlier supervised fitting. A negative result concerns this fixed-policy coupling and does not rule out a separately refitted policy.

Quick GPU-worker cost: 6269.629868s. Prior qualification/cancelled source cost: 415.972638s. Combined campaign cost: 6685.602506s (1.8571 GPU hours). New online operations:93648 backbone forwards,15608 backwards,7804 Adam steps,0 VJP. Composite deployment costs11F/2BP/1Adam per image, original RL2F/0BP. Prior cancelled-stage operation counts are last-observed lower bounds, not exact totals.

Total wall time from original19:45 Beijing T0: 105.50 minutes, within the180-minute cap. All GPU workers completed without failure or recovery. Initial CPU scorer timed out after its1200s task allocation; five complete scalar streams and657 rows of the last stream were preserved. Only the remaining1294 rows were scored once in a CPU-only continuation, with unchanged metric code and absolute cutoff. Original timeout, CPU time and recovery evidence remain recorded. No scientific experiment was rerun.

All images belong to previously exposed development data. The orders contain the same content, patient dependence is unknown, and one fitted policy seed is not independent replication. No clinical or independent held-out claim. Target soft-probability metrics are NA because sealed target outputs are masks; private images, labels, content IDs, predictions, weights and credentials are not released. No successor is authorized; monitoring stops after verified publication.
