# Evidence for the narrow R33 question

R32 is complete and delivered at commit f63aeff2beb20180a4c9f0f538c5157a2a1fd951. Its source/HOLD uses current-image BN statistics, not the original source eval mode. Native C adapts only BN affine parameters; running means/variances are disabled. HOLD predictions do not consume Adam memory. Therefore R32 SAME64_HOLD versus CROSS64_HOLD already establishes parameter-mediated prediction differences for the specified histories (+4.149404pp ORIGA, +1.350557pp REFUGE_Valid joint SEARCH), not an unresolved choice between two equally supported explanations.

R30 full-domain scores also caution against blaming Adam: optimizer-only period32 reset minus C_CONT is −0.518269pp ORIGA and −2.081698pp REFUGE_Valid, while parameter-only reset is −11.252517pp ORIGA and +4.742383pp REFUGE_Valid. These periodic interventions alter trajectories and the optimizer reset changes age; they do not isolate matched-age Adam content. R33 is a limited conditional test of that remaining question.

R32's exact query-tail decomposition, in Dice percentage points, is:

| Domain | SAME64_HOLD − SOURCE_HOLD | NATIVE_HOLD − SAME64_HOLD | NATIVE_UPDATE − NATIVE_HOLD | Sum |
|---|---:|---:|---:|---:|
| ORIGA | +4.284991 | +4.698261 | +4.632449 | +13.615701 |
| REFUGE_Valid | +0.673797 | −3.314089 | −1.777520 | −4.417811 |

This is a telescoping identity, not a unique independent causal attribution. The NATIVE term retains prefix length/Adam-age confounding. The modest RV SAME64 gain is OD-driven (+1.451865pp); OC is −0.104271pp. Neither domain label nor the sign of these scores is an online observable decision signal.

For RV, accumulated UPDATE−HOLD further decomposes on the same query images into pre-update state difference plus the current update's immediate hard-Dice change:

| History | Pre-update state − HOLD | Immediate post − pre | Total |
|---|---:|---:|---:|
| SOURCE | −0.620319 | +0.007723 | −0.612596 |
| SAME64 | −1.644043 | +0.004201 | −1.639841 |
| CROSS64 | −0.733569 | +0.003805 | −0.729764 |
| NATIVE | −1.775864 | −0.001655 | −1.777520 |

Small immediate benefits do not validate persistent writes. These arithmetic contrasts are not a per-step causal credit assignment, and changing positional windows also changes image composition. R32's six seed/order trajectories reuse content; some HOLD conditions are identical across orders. Do not count them as six independent cohorts.

R33 asks how future updates depend on matched-age Adam memory at fixed parameters, and vice versa. A hybrid may be worse because components are incompatible. There is no precommitted claim that one memory is bad, that swapping it is a method, or that the result licenses RL/selection. Both domains, orders, channels, all fixed bins and negative effects remain mandatory.

Sources: [R32 report](../r32-history-origin/REPORT.md), [R32 query aggregates](../r32-history-origin/QUERY_CELLS.csv), [R30 domain/channel aggregates](../r30-state-history/DOMAIN_CHANNEL.csv), [R33 frozen protocol](../../docs/protocols/R33_STATE_EXCHANGE.md).
