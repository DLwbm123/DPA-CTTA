# R63 RCA diagnostic: completed, primary gate failed

All 96 candidate pairs (48 SEARCH contents, two orders) and 288 predictions were scored after selector retirement. Both processes exited zero; selection read eight authorized source images/masks and zero target labels. Total measured CPU-worker wall time including preparation was 332.816 seconds; GPU use was zero. Real-pair repeatability and access-guard qualification passed.

| Contrast | Mean Dice pp | Content-cluster 95% interval | Order 0 / 1 pp |
|---|---:|---|---|
| Selector minus CW | -1.0394 | [-2.0406, -0.1269] | -2.3702 / +0.2914 |
| Selector minus VPTTA | +3.3859 | [+0.6490, +6.5364] | +2.6417 / +4.1301 |
| Selector minus fair random expectation | +1.1733 | [-0.2607, +2.8598] | +0.1358 / +2.2107 |
| Whole-mask oracle minus CW | +2.5751 | [+1.6424, +3.5446] | +2.3223 / +2.8279 |
| Whole-mask oracle minus selector | +3.6145 | [+2.3488, +4.9529] | +4.6924 / +2.5365 |

RCA chose VPTTA 33/96 times and CW 63/96 times. All target-quality pairs were non-tied; ranking accuracy was 54/96 (56.25%). REFUGE_Valid OC improved +1.4214pp, passing that secondary component, but the primary mean, interval, both-orders and worst-cell conditions failed. The worst primary cell was order0 ORIGA OC (-6.6152pp). All signed cells and all negative cells are retained in the accompanying CSVs. A whole-mask oracle can worsen an individual channel while improving joint Dice; it is not a per-channel oracle.

This fixed reference-assisted RCA configuration is rejected for allocation. The positive oracle headroom shows candidate quality variation exists, but this feedback does not reliably exploit it. Beating VPTTA alone does not rescue the primary failure. No retry, reference expansion or parameter tuning was performed.

Source-reference-assisted, exposed development evidence only: references are content-grouped, not patient-independent; source checkpoint training overlap is unknown. This is neither source-free CTTA nor independent validation, and the proxy forest omits full deployment costs of producing both candidates. Private images, labels, per-content identities, masks and traces are excluded from publication.

Next: registered R64 attributes harmful switches and missed improvements using existing traces only. It will not fit a replacement selector or scan thresholds.
