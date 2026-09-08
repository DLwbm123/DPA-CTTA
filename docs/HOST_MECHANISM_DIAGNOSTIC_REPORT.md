# H1 host mechanism diagnostic — H1_DIAGNOSTIC_COMPLETE

The prescribed source-only diagnostic completed on 2026-09-08. This replaces the earlier resource-blocked status; the 2026-09-07 block consumed zero GPU steps and remains recorded in the historical [resource audit](../results/host_mechanism_diagnostic_v1/execution_audit.json). The latest user authorized GPUs 4–7; GPU 7 had 24124 MiB free and was selected. Existing frozen code and the original ordered, finite experiment were reused without increasing the budget or interfering with any other process.

**Actual execution commit:** `daa95d25ad03e9240621c832d4e45d8ab188b1b2`. The result-delivery commit is the later revision containing this report, not a different executed code snapshot. Original scientific configuration and dependency `dbff0d985c6c95345d9fb78f5b1daef57b392564` are unchanged. The receipt binds the clean commit, diagnostic configuration, original private registration and actual GPU UUID; the formal receipt additionally binds successful smoke evidence. This is direct task authorization, not independent reviewer signoff.

[Configuration](../configs/host_mechanism_diagnostic_v1.json) · [execution audit](../results/host_mechanism_diagnostic_v1/execution_20260908.json) · [diagnostic aggregates](../results/host_mechanism_diagnostic_v1/diagnostic_summary_20260908.json) · [scalar supplement](../results/host_mechanism_diagnostic_v1/supplement_20260908.json) · [implementation contract](HOST_MECHANISM_DIAGNOSTIC_CONTRACT.md) · [target-dev draft only](TARGET_DEV_PROTOCOL_DRAFT_H1.md).

## Main finding: normalization path dominates this source-clean loss

Along the frozen N→G→I→U replacement order, Fundus macro G−N is −8.385323 percentage points (20/20 negative), while I−G is +0.005736 and U−I is +0.055699. Polyp G−N is −0.509126 pp (25/32 negative), I−G is −0.000049 and U−I is −0.002252. Thus the large source-clean deficit appears before the current update and is concentrated in the AdaBN normalization path. This is a conditional fixed-order diagnosis, not a unique causal decomposition or a general claim about shifted domains.

N = standard source BN without prompt. G = native AdaBN with identity prompt. I = native AdaBN with this visit's actual memory initialization before Adam. U = final native A output. G/I are isolated observations, not separate evolving histories. Identity FFT/IFFT error passed the predeclared tolerance on procedural inputs; a residual numerical-path difference is not claimed literally zero.

| Prediction | Fundus OD Dice (%) | Fundus OC Dice (%) | Fundus macro Dice (%) | Polyp Dice (%) |
| --- | ---: | ---: | ---: | ---: |
| N | 97.746073 | 87.139392 | 92.442733 | 97.221359 |
| G | 95.794747 | 72.320073 | 84.057410 | 96.712233 |
| I | 95.812281 | 72.314011 | 84.063146 | 96.712184 |
| U | 95.876437 | 72.361253 | 84.118845 | 96.709932 |

## Per-channel and position-bin increments

Differences are percentage points. Signs use exact unrounded values. Full medians, p10/p90, tails and sign counts are available in the machine JSON.

| Task / channel | Position bin | G−N | I−G | U−I | U−N | G−N positive/zero/negative |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fundus/OD | all | -1.951326 | 0.017534 | 0.064156 | -1.869636 | 2/0/18 |
| fundus/OC | all | -14.819320 | -0.006062 | 0.047241 | -14.778140 | 3/0/17 |
| fundus/macro | all | -8.385323 | 0.005736 | 0.055699 | -8.323888 | 0/0/20 |
| fundus/OD | 1_5 | -1.271144 | 0.000000 | -0.101086 | -1.372230 | 0/0/5 |
| fundus/OC | 1_5 | -9.670971 | 0.000000 | -0.126154 | -9.797126 | 1/0/4 |
| fundus/macro | 1_5 | -5.471058 | 0.000000 | -0.113620 | -5.584678 | 0/0/5 |
| fundus/OD | 6_16 | -1.175586 | 0.000000 | 0.116289 | -1.059297 | 1/0/10 |
| fundus/OC | 6_16 | -16.324600 | 0.000000 | 0.112088 | -16.212511 | 0/0/11 |
| fundus/macro | 6_16 | -8.750093 | 0.000000 | 0.114189 | -8.635904 | 0/0/11 |
| fundus/OD | 17_plus | -4.934840 | 0.087668 | 0.127345 | -4.719826 | 1/0/3 |
| fundus/OC | 17_plus | -17.115234 | -0.030308 | 0.085656 | -17.059885 | 2/0/2 |
| fundus/macro | 17_plus | -11.025037 | 0.028680 | 0.106501 | -10.889856 | 0/0/4 |
| polyp/polyp | all | -0.509126 | -0.000049 | -0.002252 | -0.511427 | 7/0/25 |
| polyp/polyp | 1_5 | -0.242106 | 0.000000 | -0.011041 | -0.253147 | 1/0/4 |
| polyp/polyp | 6_16 | -0.792333 | 0.000000 | -0.001639 | -0.793973 | 2/0/9 |
| polyp/polyp | 17_plus | -0.397864 | -0.000098 | 0.000074 | -0.397889 | 4/0/12 |

Actual retrieval was observed at Fundus positions 17–20 and Polyp positions 17–32. The early positions 1–4 of the counterfactual subset had no retrieval; positions were not replaced. Reference and watched logits had maximum absolute difference 0.0 in both tasks at every tested visit. State/RNG/source-gradient comparisons passed. Every reference clean Dice and defined ASSD value equals its corresponding old-pilot clean value (all paired differences exactly zero), so no cross-run score tuning was performed.

## Six branches at the 16 frozen positions

Each task has eight positions: 1,2,3,4,17,18,19,20. Each isolated branch restores the actual pre-update Adam state and initialization and makes exactly one step, then is discarded. A_cf reproduced native prompt, Adam and prediction within rtol=1e-4/atol=1e-5. No branch changes the future native trajectory. O_native/O_small use source query labels only after non-oracle updates/predictions are fixed: they are offline local diagnostics, not an online method or performance upper bound.

| Task / channel | Branch | Dice after−I (pp) | Positive/zero/negative | Dice after−A_cf (pp) | Adam update L2 mean | Distance from A_cf L2 mean | Query alignment mean |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| fundus/OD | A_cf | -0.002428 | 4/0/4 | 0.000000 | 0.235157 | 0.000000 | -0.167936 |
| fundus/OC | A_cf | -0.052629 | 4/0/4 | 0.000000 | 0.235157 | 0.000000 | -0.167936 |
| fundus/macro | A_cf | -0.027528 | 4/0/4 | 0.000000 | 0.235157 | 0.000000 | -0.167936 |
| fundus/OD | B_cf | 0.012169 | 5/0/3 | 0.014597 | 0.234703 | 0.206941 | -0.087925 |
| fundus/OC | B_cf | 0.008064 | 5/0/3 | 0.060693 | 0.234703 | 0.206941 | -0.087925 |
| fundus/macro | B_cf | 0.010117 | 5/0/3 | 0.037645 | 0.234703 | 0.206941 | -0.087925 |
| fundus/OD | C_cf | 0.012169 | 5/0/3 | 0.014597 | 0.234703 | 0.206940 | -0.087926 |
| fundus/OC | C_cf | 0.008064 | 5/0/3 | 0.060693 | 0.234703 | 0.206940 | -0.087926 |
| fundus/macro | C_cf | 0.010117 | 5/0/3 | 0.037645 | 0.234703 | 0.206940 | -0.087926 |
| fundus/OD | A_small | 0.006071 | 3/0/5 | 0.008499 | 0.023516 | 0.211641 | -0.167937 |
| fundus/OC | A_small | -0.005067 | 4/0/4 | 0.047562 | 0.023516 | 0.211641 | -0.167937 |
| fundus/macro | A_small | 0.000502 | 4/0/4 | 0.028030 | 0.023516 | 0.211641 | -0.167937 |
| fundus/OD | O_native | 0.114042 | 7/0/1 | 0.116471 | 0.254734 | 0.342947 | 0.339642 |
| fundus/OC | O_native | 0.345166 | 8/0/0 | 0.397795 | 0.254734 | 0.342947 | 0.339642 |
| fundus/macro | O_native | 0.229604 | 7/0/1 | 0.257133 | 0.254734 | 0.342947 | 0.339642 |
| fundus/OD | O_small | 0.008298 | 5/0/3 | 0.010726 | 0.025473 | 0.236117 | 0.339642 |
| fundus/OC | O_small | 0.029074 | 6/1/1 | 0.081703 | 0.025473 | 0.236117 | 0.339642 |
| fundus/macro | O_small | 0.018686 | 7/0/1 | 0.046214 | 0.025473 | 0.236117 | 0.339642 |
| polyp/polyp | A_cf | -0.006240 | 1/4/3 | 0.000000 | 0.022616 | 0.000000 | -0.001298 |
| polyp/polyp | B_cf | -0.006991 | 2/2/4 | -0.000751 | 0.029679 | 0.034559 | 0.052763 |
| polyp/polyp | C_cf | -0.006991 | 2/2/4 | -0.000751 | 0.029679 | 0.034559 | 0.052763 |
| polyp/polyp | A_small | 0.000000 | 0/8/0 | 0.006240 | 0.002262 | 0.020354 | -0.001296 |
| polyp/polyp | O_native | -0.004881 | 3/3/2 | 0.001359 | 0.030075 | 0.034478 | 0.642578 |
| polyp/polyp | O_small | -0.001336 | 0/7/1 | 0.004904 | 0.003007 | 0.022249 | 0.642577 |

For Fundus macro, B_cf improves over A_cf by only +0.037645 pp, with 5/8 positive and 3/8 negative differences. A_small improves over A_cf at 4/8 positions and worsens at 4/8; its mean advantage is +0.028030 pp, insufficient to make smaller learning rate the priority over the much larger normalization effect. O_native improves over I at 7/8 Fundus positions (+0.229604 pp mean), but remains a limited local source-label result. Polyp oracle and proxy effects are tiny and mixed. C_cf−B_cf hard Dice is exactly zero at all 16 positions and every channel, despite nonzero boundary gradients. This does not invalidate all boundary supervision or establish a global prompt-space capacity bound.

## Region loss and foreground area before/after counterfactual update

Before is I on the selected eight positions, not the complete-stream I mean. Region is the unchanged balanced BCE + soft Dice loss per channel. Areas are foreground pixel counts on the final grid; means are across visits.

| Task/channel | Branch | Region before | Region after | Predicted area before | Predicted area after |
| --- | --- | ---: | ---: | ---: | ---: |
| fundus/OD | A_cf | 0.116451 | 0.119384 | 25033.12 | 24928.25 |
| fundus/OC | A_cf | 0.699119 | 0.704354 | 8051.75 | 8004.12 |
| fundus/OD | B_cf | 0.116451 | 0.118636 | 25033.12 | 24934.12 |
| fundus/OC | B_cf | 0.699119 | 0.701546 | 8051.75 | 8018.75 |
| fundus/OD | C_cf | 0.116451 | 0.118636 | 25033.12 | 24934.12 |
| fundus/OC | C_cf | 0.699119 | 0.701546 | 8051.75 | 8018.75 |
| fundus/OD | A_small | 0.116451 | 0.116693 | 25033.12 | 25024.00 |
| fundus/OC | A_small | 0.699119 | 0.699550 | 8051.75 | 8047.38 |
| fundus/OD | O_native | 0.116451 | 0.112845 | 25033.12 | 25037.75 |
| fundus/OC | O_native | 0.699119 | 0.685223 | 8051.75 | 8071.62 |
| fundus/OD | O_small | 0.116451 | 0.116127 | 25033.12 | 25034.00 |
| fundus/OC | O_small | 0.699119 | 0.697730 | 8051.75 | 8053.75 |
| polyp/polyp | A_cf | 0.084323 | 0.084255 | 7338.75 | 7338.88 |
| polyp/polyp | B_cf | 0.084323 | 0.084449 | 7338.75 | 7338.62 |
| polyp/polyp | C_cf | 0.084323 | 0.084449 | 7338.75 | 7338.62 |
| polyp/polyp | A_small | 0.084323 | 0.084316 | 7338.75 | 7338.75 |
| polyp/polyp | O_native | 0.084323 | 0.083773 | 7338.75 | 7340.50 |
| polyp/polyp | O_small | 0.084323 | 0.084265 | 7338.75 | 7338.88 |

## Gradient evidence

Raw norms and actual loss-weighted norms are L2; cosine values are not Adam effects. Alignment is −dot(gQ,u)/(norm(gQ)norm(u)+1e−12), with zero norms undefined. Gradient vectors were never saved.

| Task | Component | Raw norm mean | Actual weighted norm mean | Cosine with query gradient mean |
| --- | --- | ---: | ---: | ---: |
| fundus | H | 0.069231723 | 0.069231723 | -0.432977 |
| fundus | R | 0.053890008 | 0.0053890009 | 0.048774 |
| fundus | D | 2.355151e-05 | 2.355151e-07 | 0.007659 |
| fundus | Q | 0.40693468 | 0.40693468 | undefined |
| polyp | H | 0.0048310619 | 0.0048310619 | -0.017686 |
| polyp | R | 0.23503122 | 0.023503122 | 0.250995 |
| polyp | D | 3.1988511e-06 | 3.198851e-08 | 0.249073 |
| polyp | Q | 0.038461639 | 0.038461639 | undefined |

The effective boundary gradient is small here (mean weighted norm ≈2.36e−7 Fundus and 3.20e−8 Polyp). Proxy/query gradient directions vary; their cosines or larger norms alone do not establish effectiveness. All pairwise cosine distributions, actual one-step update/alignment distributions and region/Dice outcomes remain in the JSON.

## ASSD, empty/full and area diagnostics

ASSD uses the same final-grid 4-connected surfaces, in pixels. Conditional ASSD means are descriptive; paired increments in the JSON use only jointly defined cohorts and expose undefined counts. Macro ASSD is not defined. Empty/full masks remain in Dice. Rates below are means of per-visit ratios, not pooled-pixel ratios; FP denominator is GT-negative pixels, FN denominator is GT-positive pixels. OC-outside-OD divides by predicted OC pixels; zero denominators yield null.

| Task/channel | Prediction | Conditional ASSD (px) | Valid/undefined | Pred empty/full | GT empty/full | Boundary-defined visits |
| --- | --- | ---: | --- | --- | --- | ---: |
| fundus/OD | N | 1.773058 | 20/0 | 0/0 | 0/0 | 20 |
| fundus/OC | N | 3.705473 | 20/0 | 0/0 | 0/0 | 20 |
| fundus/OD | G | 7.714473 | 20/0 | 0/0 | 0/0 | 20 |
| fundus/OC | G | 12.745817 | 20/0 | 0/0 | 0/0 | 20 |
| fundus/OD | I | 7.592411 | 20/0 | 0/0 | 0/0 | 20 |
| fundus/OC | I | 12.753855 | 20/0 | 0/0 | 0/0 | 20 |
| fundus/OD | U | 7.458771 | 20/0 | 0/0 | 0/0 | 20 |
| fundus/OC | U | 12.676951 | 20/0 | 0/0 | 0/0 | 20 |
| polyp/polyp | N | 0.770857 | 32/0 | 0/0 | 0/0 | 32 |
| polyp/polyp | G | 0.894972 | 32/0 | 0/0 | 0/0 | 32 |
| polyp/polyp | I | 0.894925 | 32/0 | 0/0 | 0/0 | 32 |
| polyp/polyp | U | 0.895901 | 32/0 | 0/0 | 0/0 | 32 |

| Task/channel | Prediction | GT area mean | Pred area mean | FP pixels mean | FN pixels mean | FP rate mean (%) | FN rate mean (%) | OC outside OD mean (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/OD | N | 26140.25 | 26529.95 | 788.25 | 398.55 | 0.3362 | 1.6124 | 0.0000 |
| fundus/OC | N | 8780.30 | 8530.30 | 504.10 | 754.10 | 0.1989 | 13.3310 | 0.0000 |
| fundus/OD | G | 26140.25 | 26729.10 | 1333.55 | 744.70 | 0.5575 | 2.6713 | 0.0000 |
| fundus/OC | G | 8780.30 | 8211.90 | 1812.80 | 2381.20 | 0.7000 | 14.9468 | 0.0000 |
| fundus/OD | I | 26140.25 | 26717.00 | 1323.40 | 746.65 | 0.5534 | 2.6792 | 0.0000 |
| fundus/OC | I | 8780.30 | 8206.45 | 1810.30 | 2384.15 | 0.6991 | 14.9683 | 0.0000 |
| fundus/OD | U | 26140.25 | 26617.10 | 1259.75 | 782.90 | 0.5269 | 2.8090 | 0.0000 |
| fundus/OC | U | 8780.30 | 8153.25 | 1778.00 | 2405.05 | 0.6865 | 15.1319 | 0.0000 |
| polyp/polyp | N | 6008.09 | 5991.44 | 118.94 | 135.59 | 0.1029 | 2.5648 | not applicable |
| polyp/polyp | G | 6008.09 | 5969.44 | 124.94 | 163.59 | 0.1078 | 3.2587 | not applicable |
| polyp/polyp | I | 6008.09 | 5969.50 | 125.00 | 163.59 | 0.1079 | 3.2576 | not applicable |
| polyp/polyp | U | 6008.09 | 5969.75 | 125.16 | 163.50 | 0.1080 | 3.2586 | not applicable |

## Mechanical evidence, compute and resources

Both stages used loaded registered real source weights and the original K=4 source proxy. Smoke query images/masks were procedural; formal query images/masks were the original source-clean 20/32 groups. Both smoke identity FFT/IFFT checks passed (maximum absolute errors 9.83e−7 Fundus and 3.10e−6 Polyp). Same-device native/watched equivalence and A_cf restoration passed with the frozen tolerance. Models, buffers, clone independence, counters, source-gradient behavior and RNG remained isolated; historical memory is compared without substituting new histories.

| Counter | Smoke | Formal | Total |
| --- | ---: | ---: | ---: |
| reference_steps | 34 | 52 | 86 |
| watched_steps | 34 | 52 | 86 |
| counterfactual_steps | 12 | 96 | 108 |
| reference_pushes | 34 | 52 | 86 |
| watched_pushes | 34 | 52 | 86 |
| gradient_calls | 8 | 64 | 72 |
| reference_model_forwards | 68 | 104 | 172 |
| reference_prompt_forwards | 70 | 124 | 194 |
| watched_model_forwards | 68 | 104 | 172 |
| watched_prompt_forwards | 70 | 124 | 194 |
| diagnostic_model_forwards | 84 | 232 | 316 |
| diagnostic_prompt_forwards | 88 | 250 | 338 |
| N_model_forwards | 34 | 52 | 86 |
| proxy_model_forwards | 2 | 16 | 18 |
| proxy_images | 8 | 64 | 72 |

Adam calls total **80 smoke + 200 formal = 280**, exactly the authorized budget. Extra `gradient_calls` are isolated autograd.grad calls, separate from native backward/Adam. Network and prompt forwards are separately counted; a K=4 proxy network forward counts once and four proxy images. These are not FLOP counts. No repeated pilot, stability run, source training, DD, target inference or parameter search occurred.

| Phase/task | Elapsed seconds | Peak process allocated GiB |
| --- | ---: | ---: |
| smoke/fundus | 25.135 | 3.767 |
| smoke/polyp | 19.272 | 2.173 |
| formal/fundus | 40.338 | 4.123 |
| formal/polyp | 42.044 | 2.181 |

| Task | Mean full-visit seconds | Mean isolated-diagnostics seconds | Native Adam update L2 mean |
| --- | ---: | ---: | ---: |
| fundus | 2.0037 | 1.3339 | 0.1995357 |
| polyp | 1.2953 | 0.4433 | 0.0170541 |

Total stage wall time including assembly was 54.297 s smoke and 89.109 s formal. Timings include adaptation and diagnostic scoring and are not pure inference latency. Peaks are process allocated memory, not reserved memory or whole-device occupancy; no empty_cache was used. Python 3.10.6, Torch 2.2.1+cu121, CUDA 12.1, cuDNN 8902; TF32 off, cuDNN benchmark off, cuDNN deterministic on, deterministic-algorithm mode False. Seed alone is not proof of device-wide determinism. No environment was upgraded.

Smoke, formal and independent CPU recomputation exits were all **0**. The independent process reread all 52 complete formal visit records (including 16 six-branch records), validated exact registration/order/counters/metrics and produced the public aggregate without initializing CUDA. No failure files, resumed steps or incomplete prefixes exist. Private directory/file modes were checked as 0700/0600. The final file-byte total is 1420785 bytes, derived from the measured pre-export total plus the two public export sizes; budget 256 MiB. Real paths, group/sample IDs, raw per-visit JSONL and registration/model digests stay private; no model, optimizer, logits, features, images, masks or gradient vectors were saved.

## Existing 832-record analysis — completed, no model run

The original private JSONL passed ordered coverage/channel/lifecycle validation. Analysis exited 0 with CUDA uninitialized and no Adam calls. Exact-zero signs, linear quantiles and ascending ceil(10% n) tails were fixed before analysis. Differences below are percentage points. ASSD differences use jointly defined pairs; macro ASSD is not defined. Original logs contain no actual retrieval events, prediction area or confusion matrices; none are inferred.

| Task / channel | Clean A−N mean | Median | p10 | p90 | Positive / zero / negative | Minimum | Worst 10% mean |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| fundus / OD | -1.869636 | -1.409808 | -2.053575 | -0.203453 | 2 / 0 / 18 | -15.552919 | -9.458257 |
| fundus / OC | -14.778140 | -15.376423 | -29.663198 | 0.265303 | 3 / 0 / 17 | -41.902986 | -36.439562 |
| fundus / macro | -8.323888 | -8.025009 | -15.377249 | -1.361080 | 0 / 0 / 20 | -22.633290 | -19.172978 |
| polyp / polyp | -0.511427 | -0.249367 | -0.767138 | 0.121092 | 7 / 0 / 25 | -4.534415 | -2.775470 |

| Task / macro or single channel | Clean positions 1–5 | 6–16 | 17 onward |
| --- | ---: | ---: | ---: |
| fundus | -5.584678 | -8.635904 | -10.889856 |
| polyp | -0.253147 | -0.793973 | -0.397889 |

All A−N/B−N/C−N/B−A/C−B task/channel/segment distributions, zero/positive/negative counts, signed lower tails, common-ASSD counts and same-content cross-segment differences are in the linked JSON. Complete group-level four-segment measurements remain private. The position bins are not evidence of retrieval. These source development groups are not new validation data or verified independent patients; the descriptive trends do not establish significance or mechanism.

## Local regression

Full supported-environment regression on 2026-09-07 (unchanged executed code, reused without rerunning on 2026-09-08): **179 tests**, 0 failures, 0 errors, 0 skips; exit 0; 374.47 seconds. Python 3.12.9, Torch 2.6.0. Old tests and assertions are retained. [Machine audit](../audit/host_diagnostic_results.json), [full log](../audit/host_diagnostic_cpu.log). Four earlier targeted tests also passed. Real source images/weights and CUDA were denied during this CPU suite; blocked API attempts were empty.

## One next-round priority and explicit limits

**Propose only H_source_stats:** a separately named host candidate that fixes source normalization statistics. This is the best-supported next diagnostic candidate because G−N dominates this clean-source deficit, particularly Fundus OC. Do not silently disable BN matching: the future protocol must retain and test a nonconstant differentiable statistic-matching objective with nonzero prompt gradients and separately specify its relationship to normalization. No H implementation or target run is part of this task. The target-dev draft contains N, original A, original B and at most this one H; it still needs separately authorized target registrations and a finite budget.

These 52 previously used source development content groups are not 52 verified independent patients or a fresh validation set. Eight local counterfactual positions per task and one seed do not establish significance, unseen generalization, method effectiveness or SOTA. The observations do not rule out all proxy styles, all DD, all boundary supervision or global prompt capacity. Source performance above N is not imposed as a prerequisite for genuine shifted target development. Engineering completion is distinct from a useful method.

**H1_DIAGNOSTIC_COMPLETE. Stop.** No further model runs or automatic follow-up were created. CI remains NOT_CONFIGURED.
