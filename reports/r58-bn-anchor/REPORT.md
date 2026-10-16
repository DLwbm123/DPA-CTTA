# R58 fixed half-source BN anchor

Frozen gate: False

MIX/SOURCE/BATCH are each one stateless stream mapped across paired C orders/seeds; these are not independent replicated inference runs. Primary MIX-BATCH only; comparisons to C and SOURCE descriptive. No weight search, parameter update, independent validation or superiority over adaptive methods is established by this gate.

MIX macro Dice75.801158% versus BATCH75.079176%: +0.721982pp (95% content CI[0.515164,0.941350]). However imageweighted delta is-0.258703pp and the worst cell is-2.073848pp, so the frozen gate fails. REFUGE_Valid OC worsens from66.5936 to64.5198; the motivating deficit is not repaired. MIX remains2.636088pp below C. No alpha sweep, secondary-arm promotion or success claim.

Completion:1951 control images,46824 scored rows; GPU retired exit0,stateless replay passed,115.501084 GPU-process seconds and138.757361 wall seconds. No retry. Next registered question: whether cumulative affine/optimizer history contributes to C's trade-off, tested with a per-arrival reset while preserving current-image update and arrival RNG progression. This is a diagnostic of persistent state, not a novel-method claim.

Numerical-policy caveat discovered during R59 qualification: new stateless controls used default GPU numerical settings whereas reused C disabled TF32 and enabled deterministic algorithms. Within-new-control comparisons retain a shared policy, but attribution against historical C is provisional pending a matched-policy sensitivity audit. Do not interpret C-versus-control differences as isolating BN/update mechanisms alone. R59 now matches C's policy and first-arrival parity passes.
