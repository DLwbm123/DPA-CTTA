# R32: history content and subsequent adaptation

**Complete: six jobs, 96 query branches, 72,060 scored condition-content observations.** All six online workers retired with exit0 before CPU labels were released. Scientific runtime commit: `2a2015a91051de4fa4b6b6deeeef5aad48682903`. Recovery changes affected orchestration only.

## Finding

The tested updater has opposite domain-level effects. Continuing query updates improves ORIGA under every starting history, but lowers mean REFUGE_Valid Dice under every starting history. Matched same-domain history is better than the specified preceding-domain history in both query domains. REFUGE_Valid therefore exhibits both a history-content effect and a harmful subsequent-update effect; cross-domain history alone cannot explain the full degradation. These are finite counterfactual effects of the registered interventions, not a deployable selector or a universal biological domain claim.

All reported differences below are Dice percentage points, averaged equally over OD/OC and three seeds. Order0 and order1 are kept separate because their CROSS64 histories differ. SEARCH is primary development evidence; the legacy `SEALED_REVIEW` name in CSV files does not mean independent confirmation.

## Ordered history contrasts

| Query domain | Order0 CROSS history | Order1 CROSS history |
|---|---|---|
| ORIGA | REFUGE | REFUGE_Valid |
| REFUGE_Valid | ORIGA | Drishti_GS |

| Query domain | Contrast | SEARCH order0 / order1 | Legacy REVIEW order0 / order1 | Positive SEARCH trajectories |
|---|---|---:|---:|---:|
| ORIGA | SAME64_HOLD − CROSS64_HOLD | +3.3527 / +4.9461 | +3.5667 / +5.0751 | 6/6 |
| ORIGA | NATIVE_HOLD − SAME64_HOLD | +5.2900 / +4.1065 | +5.1785 / +3.9319 | 6/6 |
| ORIGA | SAME64_HOLD − SOURCE_HOLD | +4.2850 / +4.2850 | +4.6751 / +4.6751 | 6/6 |
| ORIGA | CROSS64_HOLD − SOURCE_HOLD | +0.9323 / -0.6611 | +1.1084 / -0.4000 | 3/6 |
| ORIGA | NATIVE_HOLD − SOURCE_HOLD | +9.5750 / +8.3915 | +9.8536 / +8.6070 | 6/6 |
| ORIGA | NATIVE_UPDATE − NATIVE_HOLD | +4.8382 / +4.4267 | +4.1167 / +3.6639 | 6/6 |
| ORIGA | SOURCE_UPDATE − SOURCE_HOLD | +9.1330 / +9.3947 | +9.0013 / +9.3338 | 6/6 |
| ORIGA | SAME64_UPDATE − SAME64_HOLD | +6.6017 / +6.7880 | +6.1406 / +6.3604 | 6/6 |
| ORIGA | CROSS64_UPDATE − CROSS64_HOLD | +9.6724 / +10.0062 | +9.3584 / +9.7559 | 6/6 |
| ORIGA | HISTORY_UPDATE_INTERACTION − ALGEBRA | -3.0706 / -3.2182 | -3.2178 / -3.3955 | 0/6 |
| REFUGE_Valid | SAME64_HOLD − CROSS64_HOLD | +1.4256 / +1.2756 | +1.3965 / +1.2527 | 6/6 |
| REFUGE_Valid | NATIVE_HOLD − SAME64_HOLD | -5.7528 / -0.8754 | -6.4250 / -0.9867 | 0/6 |
| REFUGE_Valid | SAME64_HOLD − SOURCE_HOLD | +0.6738 / +0.6738 | +0.4914 / +0.4914 | 6/6 |
| REFUGE_Valid | CROSS64_HOLD − SOURCE_HOLD | -0.7518 / -0.6018 | -0.9052 / -0.7614 | 0/6 |
| REFUGE_Valid | NATIVE_HOLD − SOURCE_HOLD | -5.0790 / -0.2016 | -5.9336 / -0.4953 | 1/6 |
| REFUGE_Valid | NATIVE_UPDATE − NATIVE_HOLD | -0.9121 / -2.6429 | -0.8595 / -2.4065 | 0/6 |
| REFUGE_Valid | SOURCE_UPDATE − SOURCE_HOLD | -0.5293 / -0.6959 | -0.6386 / -0.7781 | 0/6 |
| REFUGE_Valid | SAME64_UPDATE − SAME64_HOLD | -1.6104 / -1.6693 | -1.5711 / -1.6018 | 0/6 |
| REFUGE_Valid | CROSS64_UPDATE − CROSS64_HOLD | -0.6254 / -0.8341 | -0.5554 / -0.7465 | 0/6 |
| REFUGE_Valid | HISTORY_UPDATE_INTERACTION − ALGEBRA | -0.9850 / -0.8351 | -1.0157 / -0.8553 | 0/6 |

The interaction is `(SAME64_UPDATE − SAME64_HOLD) − (CROSS64_UPDATE − CROSS64_HOLD)`. Its SEARCH means are −3.1444pp in ORIGA and −0.9101pp in REFUGE_Valid. In ORIGA, this mainly reflects less subsequent improvement from the better same-domain starting state; it does not indicate that ORIGA updates are harmful. In REFUGE_Valid, further updates erode the advantage of SAME64.

NATIVE versus SAME64 retains the original prefix-length and Adam-age confounding. For REFUGE_Valid the prefix penalty is −5.7528pp in order0 and −0.8754pp in order1. These values must not be interpreted as a pure domain-origin effect. ORIGA CROSS64 versus source even changes sign between orders (+0.9323/−0.6611pp); pooling them would hide treatment heterogeneity.

## Fixed references on the same query tails

| Condition | ORIGA SEARCH Dice % | REFUGE_Valid SEARCH Dice % |
|---|---:|---:|
| SOURCE_HOLD | 66.0106 | 75.2392 |
| SOURCE_UPDATE | 75.2744 | 74.6266 |
| SAME64_HOLD | 70.2955 | 75.9130 |
| SAME64_UPDATE | 76.9904 | 74.2732 |
| CROSS64_HOLD | 66.1461 | 74.5625 |
| CROSS64_UPDATE | 75.9854 | 73.8327 |
| NATIVE_HOLD | 74.9938 | 72.5989 |
| NATIVE_UPDATE | 79.6263 | 70.8214 |
| ANCHOR | 80.0816 | 70.7409 |
| C_EPISODIC | 66.3994 | 75.3900 |

`NATIVE_UPDATE` exactly replays C_CONT. All eight diagnostic conditions remain below fixed ANCHOR on ORIGA; this experiment has not produced an overall better CTTA method. On REFUGE_Valid, SAME64_HOLD exceeds native C_CONT by5.0916pp and ANCHOR by5.1721pp, but that state uses oracle history construction and freezing chosen for diagnosis. No cross-domain policy is selected from these scores. C_EPISODIC is explanatory and never replaces the fixed C_CONT/ANCHOR references.

## Position and channel qualifications

| REFUGE_Valid UPDATE − HOLD | First64 query, order0 / order1 | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|---:|
| NATIVE | +0.1656 / -0.0980 | +0.1108 / -0.5756 | -0.2703 / -1.7271 | -1.1718 / -2.8570 | -1.9986 / -4.8148 |
| SOURCE | +0.7306 / +0.6485 | +0.7358 / +0.5742 | -0.3396 / -0.3909 | -0.2636 / -0.5015 | -1.9141 / -2.1107 |
| SAME64 | -0.1774 / -0.1202 | -0.3516 / -0.3832 | -0.9956 / -0.9117 | -1.5669 / -1.7091 | -3.1202 / -3.2374 |
| CROSS64 | +0.7477 / +0.7120 | +0.9686 / +0.7788 | -0.3255 / -0.4057 | -0.4382 / -0.6842 | -2.2923 / -2.5805 |

SOURCE and CROSS64 updates initially help on REFUGE_Valid but have negative full-tail effects. SAME64 updates are already negative in the fixed first64-query window. NATIVE has a small positive first64 effect in order0 and a negative one in order1. These paired contrasts compare the same images within each window; differences across windows also reflect changing image content, so the window sequence is not an isolated time-dose curve. All ORIGA joint UPDATE−HOLD window/order means are positive.

| REFUGE_Valid contrast | OD mean pp (positive trajectories) | OC mean pp (positive trajectories) |
|---|---:|---:|
| SAME64_HOLD − SOURCE_HOLD | +1.4519 (6/6) | -0.1043 (2/6) |
| SAME64_HOLD − CROSS64_HOLD | +2.2406 (6/6) | +0.4605 (6/6) |
| NATIVE_HOLD − SAME64_HOLD | -3.6878 (0/6) | -2.9404 (0/6) |
| NATIVE_UPDATE − NATIVE_HOLD | -2.0188 (1/6) | -1.5363 (0/6) |
| SOURCE_UPDATE − SOURCE_HOLD | -0.2109 (1/6) | -1.0143 (0/6) |
| SAME64_UPDATE − SAME64_HOLD | -2.1325 (0/6) | -1.1472 (0/6) |
| CROSS64_UPDATE − CROSS64_HOLD | -0.4395 (1/6) | -1.0200 (0/6) |

The modest SAME64_HOLD gain over source (+0.6738pp joint) is driven by OD (+1.4519pp); OC is slightly negative (−0.1043pp, positive in2/6 trajectories). Full-tail UPDATE−HOLD is negative in every joint REFUGE_Valid trajectory for all four histories, but a few OD-only trajectory cells are positive. All individual negative and positive cells remain in the tables.

The mean immediate pre/post hard-Dice change on REFUGE_Valid is +0.007723pp SOURCE, +0.004201pp SAME64, +0.003805pp CROSS64 and −0.001655pp NATIVE, whereas their accumulated UPDATE−HOLD effects are −0.6126, −1.6398, −0.7298 and −1.7775pp respectively. A small average one-step improvement therefore does not validate persistent state writes. These hard-mask diagnostics are not soft calibration measurements.

## Execution and audit

- Formal coverage:11,706 native replay arrivals;1,536 matched-history updates;63,456 query arrivals;44,970 Adam/backward steps and423,216 forwards. Each of six native replays matched all1,951 reference masks. Native UPDATE and SOURCE HOLD add15,864 exact query-reference checks across the matrix.
- Final online retirement:2026-10-09 15:39:26.4196 Beijing; CPU scoring started15:39:26.4528; labels released15:39:28.7541; scoring completed15:42:53.5085. All online label-read counts are zero.
- GPU-worker cost including profiles and the admission failure:9.275215h. CPU scorer:207.056s. Original-T0 wall duration:5.155747h. The failed admission cost3.149629s is retained. Recovery supervisor setup failures occurred before any new worker computation and remain documented in [RECOVERY.md](RECOVERY.md); they are not unreported scientific attempts.
- Three real profiles add528 Adam steps and4,416 forwards. The six successful formal jobs each record7,495 Adam steps and70,536 forwards. No budget or T0 reset, arm removal, outcome-driven seed choice, or change to the updater occurred.
- Public tables contain2,880 query aggregate rows,7,488 paired rows,3,603 negative paired rows and672 state-diagnostic rows. These overlapping bins and repeated conditions are not independent sample counts.72,060 is the number of scored condition-content observations, not unique patients.

## Interpretation and next diagnostic

The evidence supports separating the two carried state components next: learned BN affine parameters and Adam moments. SAME64/CROSS64 already match update count and optimizer age; exchanging their parameter and optimizer states in a separately frozen factorial would distinguish history encoded in parameters from history encoded in optimizer memory. Both ORIGA and REFUGE_Valid and both ordered history pairs must remain in that diagnostic. This is a justified next question, not a result obtained here or a claim that optimizer memory explains all degradation. In particular, the harmful SOURCE_UPDATE tail effect shows that pre-existing foreign optimizer memory is not necessary for the observed REFUGE_Valid loss.

No new experiment has been launched as part of this readout. A follow-up must first freeze its matrix, exact replay controls, shared query RNG, state-exchange validation, label barrier and cost accounting. Do not start RL, an outcome-selected stopping policy, an LR scan, or a domain-aware deployed controller from this diagnostic.

## Limits and artifacts

Queries start at within-domain image65 (ORIGA586 images; REFUGE_Valid736); they are not whole-stream endpoints. The first64-image history length and selected domains are post-hoc development choices. SAME64 and CROSS64 have distinct image contents but unknown patient linkage. SEARCH and legacy REVIEW are development-exposed; patient/ROI provenance remains UNKNOWN. The actual domain labels are used only for offline intervention construction and scoring, and content, prevalence, appearance and annotation practice may co-vary. These results provide neither independent confirmation nor clinical evidence.

- [Frozen protocol](../../docs/protocols/R32_HISTORY_ORIGIN.md), [query registration](QUERY_REGISTRATION.json), [prior position audit](PRIOR_AUDIT.md).
- [All query cells](QUERY_CELLS.csv), [all paired cells](PAIRED_CELLS.csv), [paired summaries](PAIRED_SUMMARY.csv), [all negative cells](ALL_NEGATIVE_CELLS.csv), [state diagnostics](STATE_DIAGNOSTICS.csv).
- [Completion audit](COMPLETION_AUDIT.json), [cost ledger](COST.csv), [recovery record](RECOVERY.md), [machine-readable readout](NEXT_DECISION.json).

Only source, protocols, aggregate metrics, operation/cost receipts and reports are public. Images, labels, masks, identities, per-content private scores, snapshots, weights and credentials are excluded.
