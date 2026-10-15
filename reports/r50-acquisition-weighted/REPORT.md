# R50 acquisition weighted structural filtering

Status: **COMPLETE**.

Primary: C_JS_ACQ_WEIGHTED minus C. C_JS_SHARED and unchanged CW_LSO are retained controls.
A gain over C is distinct from improving the original filter or the strong control. Each claim has its own frozen gate.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.685983 | 78.077388 |
| C_JS_SHARED | 78.670502 | 78.059690 |
| C_JS_ACQ_WEIGHTED | 78.682972 | 78.075381 |
| CW_LSO | 79.043595 | 78.412964 |

Frozen decisions: {"primary": "C_JS_ACQ_WEIGHTED", "standalone_signal": false, "improvement_over_shared": false, "improvement_over_strong_control": false, "review_descriptive_only": true, "independent_generalization": false, "automatic_followup": false}.
Each positive comparison requires >=0.1pp, paired 95% CI lower>0, both orders positive, >=5/6 positive trajectories, imageweighted>=0 and worst seed-mean domain/channel/order cell>=-2pp.
All 24 fresh trajectories and independent CPU scores are required; unscored or incomplete means NOT_EVALUABLE, not zero.
All positive and negative domain/channel/order/seed cells and bootstrap intervals are retained. SEARCH/REVIEW are historically exposed; REVIEW is descriptive. No independent generalization or clinical/anatomical protection claim.
The new matrix changes direction selection and effective strength. Derivatives describe a pseudo-objective; association cannot prove Dice preservation. No descent constraint, retuning, retry or next round.
GPU-worker 10.779942/48h; all profiles, initialization and failures billed. Original T0 retained.
Only own source/protocol and anonymous aggregate results/cost/audit are public. Raw images, masks, content identities/hashes, predictions, model state and private logs remain private.

## Full-result interpretation

All three registered improvement claims fail. The new filter is close to C with a small negative effect. A numerically higher macro mean than the original shared filter is not a passed improvement gate.

| SEARCH contrast | Delta pp | 95% CI pp | Positive trajectories | Order0 / Order1 pp | Imageweighted pp | Worst seed-mean cell pp |
|---|---:|---|---:|---|---:|---:|
| C_JS_ACQ_WEIGHTED minus C | -0.003011 | [-0.004056, -0.002027] | 2/6 | -0.007738 / +0.001716 | -0.001658 | -0.026574 |
| C_JS_ACQ_WEIGHTED minus C_JS_SHARED | +0.012469 | [-0.013511, +0.034938] | 3/6 | +0.075164 / -0.050226 | -0.104021 | -0.640832 |
| C_JS_ACQ_WEIGHTED minus CW_LSO | -0.360624 | [-0.410162, -0.309661] | 0/6 | -0.202435 / -0.518813 | -0.470826 | -2.130065 |
| C_JS_SHARED minus C | -0.015480 | [-0.038387, +0.010978] | 3/6 | -0.082902 / +0.051942 | +0.102363 | -0.847606 |
| CW_LSO minus C | +0.357612 | [+0.307464, +0.406962] | 6/6 | +0.194696 / +0.520528 | +0.469169 | -0.794731 |

New minus C fails the magnitude, CI, order, trajectory and imageweighted criteria. New minus old shared fails the same criteria, including a CI crossing zero and negative imageweighted effect. New minus CW_LSO fails every criterion, including the -2pp worst-cell limit: REFUGE_Valid OD, order1 is -2.130065pp. CW_LSO minus C remains a positive same-round control: +0.357612pp, CI [0.307464,0.406962], six positive trajectories, both orders positive and worst cell -0.794731pp.

REVIEW remains descriptive despite the legacy SEALED_REVIEW field name: new minus C is -0.002007pp, CI [-0.003297,-0.000764]; new minus old shared is +0.015690pp, CI [-0.016608,+0.041291]; new minus CW_LSO is -0.337583pp, CI [-0.387024,-0.284405]. None supports a new improvement claim.

## All domain/channel effects

Each entry below averages the paired effects across the original seeds/orders. Units are percentage points. Full domain/channel confidence intervals are in CONTENT_BOOTSTRAP_CI.csv; all order and seed cells, including positive cells, are in ALL_PAIRED_CELLS.csv. ALL_NEGATIVE_CELLS.csv is the complete negative subset.

| Domain | Channel | Old shared minus C | New minus C | New minus old shared | New minus CW_LSO |
|---|---|---:|---:|---:|---:|
| REFUGE | OD | +0.076339 | +0.006819 | -0.069519 | +0.047934 |
| REFUGE | OC | +0.195639 | +0.005379 | -0.190260 | +0.370212 |
| REFUGE_Valid | OD | +0.366820 | +0.000769 | -0.366051 | -1.147370 |
| REFUGE_Valid | OC | +0.468707 | +0.011069 | -0.457638 | +0.086020 |
| Drishti_GS | OD | -0.185767 | -0.003576 | +0.182191 | +0.172919 |
| Drishti_GS | OC | -0.475549 | -0.014255 | +0.461294 | -0.824546 |
| ORIGA | OD | -0.197041 | -0.013623 | +0.183418 | -0.361121 |
| ORIGA | OC | -0.372990 | -0.016673 | +0.356317 | -1.229036 |

The Drishti_GS/ORIGA losses shrink greatly relative to old shared filtering but remain negative relative to C in both channels. REFUGE_Valid OD/OC gains shrink from +0.366820/+0.468707pp to +0.000769/+0.011069pp. The new OD interval crosses zero. New minus C worst seed-mean domain/channel/order loss is -0.026574pp versus old shared minus C -0.847606pp; this reduction in harm comes with loss of most local benefits. This is not selective preservation of REFUGE_Valid gains.

## Mechanical evidence

SEARCH statistics below are arrival-weighted over 6102 paired seed/order observations, not domain-equal Dice averages.

| Diagnostic | Old shared | Acquisition weighted |
|---|---:|---:|
| joint.rank | 4 | 4 |
| joint.acquisition_capture_fraction | 1 | 1 |
| joint.anatomy_capture_fraction | 1 | 1 |
| joint.proposal_span_energy_fraction | 0.0490618267 | 0.0479948846 |
| joint.relative_metric_update_change | 0.165549814 | 0.00543620206 |
| joint.filter_eigenvalues.0 | 0.0152134871 | 0.953891939 |
| joint.filter_eigenvalues.1 | 0.0426264452 | 0.996016015 |
| joint.filter_eigenvalues.2 | 1 | 1 |
| joint.filter_eigenvalues.3 | 1 | 1 |
| joint.proposal_metric_norm | 0.000130519808 | 0.000129162769 |
| joint.filtered_metric_norm | 0.000127960988 | 0.000128964175 |
| update_direction.proposal_dot | -0.000433520728 | -0.000425435974 |
| update_direction.actual_dot | -0.000381561845 | -0.000421138263 |
| update_direction.dot_change | 5.19588835e-05 | 4.29771105e-06 |
| structure_evidence_mean.0 | 0.00938833437 | 0.00948861527 |
| structure_evidence_mean.1 | 0.00671483852 | 0.00680706394 |
| BN_gradient_norm | 0.308869209 | 0.305008129 |
| BN_update_norm | 0.00316470704 | 0.00310656437 |
| pseudo_loss | 0.0597623994 | 0.0595300754 |

The intervention changes from 16.554981% to 0.543620% of the native proposal in the Adam metric (about 30.5 times smaller). The two contracted eigenvalues move from 0.015213/0.042626 to 0.953892/0.996016; the new operator is near identity. Rank4 and capture1 are expected from constructing a basis spanning both sensitivity blocks, not evidence of anatomical validity. The normalized blocks preserve magnitude-independent directions; a small local proxy does not establish a safe direction.

Both filters make the native pseudo-loss directional derivative less negative on average: +5.195888e-5 for old shared and +4.297711e-6 for new. New proposal/actual averages are -4.254360e-4/-4.211383e-4. These paired within-step observations support reduced interference with that local pseudo-objective, but do not prove that the removed update would improve Dice. Cross-arm native proposals also differ after trajectories diverge. No claim that every individual update has a negative derivative follows from these means.

| Domain | Old update change % | New update change % | Old derivative change | New derivative change |
|---|---:|---:|---:|---:|
| Drishti_GS | 15.969242 | 0.327839 | 5.8889009e-05 | 1.18854007e-06 |
| ORIGA | 14.709236 | 0.543280 | 7.16628983e-05 | 8.2378563e-06 |
| REFUGE | 19.865404 | 0.410018 | 2.77147163e-05 | 6.45411972e-07 |
| REFUGE_Valid | 16.541111 | 0.615853 | 4.69907333e-05 | 2.98078372e-06 |

The OD/OC structure evidence remains similar across the two filters; for example SEARCH means are 0.009388/0.006715 versus 0.009489/0.006807. Thus this intervention does not show that structure evidence became more discriminative. Overall and domain-specific evidence, norms, base/LSO losses, BN statistics and available region gradients remain in both mechanism tables. C-based arms have no original region-logit hook: those gradients are unavailable, not zero. CW_LSO retains its recorded OD/OC region gradients and LSO0.1; no other arm acquires LSO.

## Execution, coverage and cost

All 24 unique online jobs completed 1951 arrivals, followed by 24 independent CPU scorers with 1951 rows each (46824 total). All 52 attempt receipts are present: four profiles, 24 online, 24 scores, with no same-round failure. The last online exit precedes the first scorer start. A terminal PID audit found no remaining recorded watchdog, supervisor, online or scorer processes. Full attempt charges and coverage checks are in RESOURCE_LEDGER.json and COMPLETION_AUDIT.json.

Preparation still includes the pre-freeze Python3.10 syntax failure: 0.34593280404806137 CPU seconds; successful CPU54 checks cost 43.01602158299647 seconds. Neither used GPU. These are separate from 481.2693135738373 CPU-worker scoring seconds. Original T0 and all budgets remain unchanged.

| Condition | Formal GPU-worker hours | Same-round profile peak reserved MiB | Calls/arrival |
|---|---:|---:|---|
| C | 2.379304 | 686 | 8F/1B/1Adam |
| Old shared | 2.663923 | 898 | 10F/5B/1Adam |
| Acquisition weighted | 2.760572 | 898 | 10F/5B/1Adam |
| CW_LSO | 2.947606 | 688 | 8F/1B/1Adam |

All billed GPU-worker time including profiles/initialization is 10.779942089253002 hours. Scoring completes about 3.58 hours after T0; terminal resource accounting is 12895.049118041992 wall seconds and 3306057146 bytes. Cumulative 15 closed/stopped rounds are 127.88115434428058 GPU-worker hours; with all prior isolated profiles the cumulative total is 128.12113314343827 hours. GPU sharing and phase-specific load confound timing ratios: these charges are actual costs, not a controlled speed benchmark. Call counts and same-round memory peaks are the appropriate architectural overhead evidence.

## Follow-up decision and external data

Recommendation: do not prioritize an equal-update-magnitude old-filter experiment or further tuning of this branch. New fails the standalone C gate and does not reliably beat the original filter. The much weaker intervention returns toward C, recovering damage while removing most benefits. This is compatible with an intervention-strength explanation but does not establish it causally. An equal-magnitude control could study mechanism, yet the present results offer no efficacy signal to justify it as the next improvement experiment. No such control was implemented or launched.

The minimum worthwhile next proposal is to validate the established CW_LSO gain on a genuinely held-out setting, with only source-only, C and unchanged CW_LSO, the original three seeds and a prospectively fixed stream order. This would test transfer of the positive core rather than expand the current filter search. It is a proposal only: select source/target domains and freeze architecture, preprocessing, checkpoint, metrics and resource budget before requesting launch confirmation. MRI transfer changes the task and needs a separately qualified adapter; it is not currently runnable. If the user instead prioritizes mechanism over efficacy, an equal-magnitude old-filter control would require a separate predeclared budget and explicit approval.

SSL_CL-aligned data preparation receipts report DATA_READY_NO_EXPERIMENT: 2932 HDF5 files (1843390563 bytes), 36 image-only views and 18 offline scorer views. Fundus has 660 cases across three domains; prostate has 116 patients/six sites/3887 slices; M&Ms has 345 patients/690 ED-ES volumes/7908 slices, of which only the original 320 patients/640 volumes are evaluation eligible. The 25 auxiliary patients remain unlabeled training only. Original split seeds0/1/2 and patient-level roles are preserved.

REFUGE and Drishti_GS overlap the existing CTTA development exposure, so this fundus package cannot be called independent external evidence. MRI needs single-channel input, appropriate output classes/loss, preprocessing and 3D/patient-level scoring, plus an appropriate source checkpoint. Source-only evaluation must be distinguished from label-free online adaptation; target labels stay with the offline scorer. This closeout reuses the completed preparation audits and does not rescan every voxel or launch external validation.

User confirmation is required before any next experiment. R50 monitoring ends after verified GitHub delivery, regardless of its failed scientific gates.
