# C_RBE: method conformance

All rows implement the supplied design, not a claim of exact original-paper reproduction. Real source assets are PENDING.

| Specification | Production mapping | Formula and boundary |
|---|---|---|
| C1 | `r7_c_rbe.RBE.observe` | 64x64 tokens; Pc32x64, Pa16x64, conditional ka16, O16x8 and perpatch normalized8x32 H;512 observations. |
| C2 | `RBE.cal_loss; SourceTrainer.cal_step/constant_variance` | Task R=1; only h_R trains in cal; RGB .04progress/token.02progress independent noise; nominal proxy; all unperturbed cal observations geometric mean8. |
| C3 | `r7_c_rbe.correct / energy` | Exactly3 Cholesky IRLS; Huber1, nu1 atzero, mean/512 and fixedridge.1, no fallback. |
| C4 | `RBE.fit_loss; SourceTrainer` | Query +.1state +.1obs +.05reconstruction +.05 clean/styled coefficients +.001 within-basis coherence average. Independent STATIC. |

Shared: original up1/up3 FiLM1024, current BN, fit-only observer, float32 backbone/MLP and differentiable float64 inference, source fit/cal/val and support/query isolation. Full network and mathematical comparisons are in the final CPU logs.

Persistent online state: z32, counter. No image, patient token, probability map or identity persists. States reset per trajectory; STATIC resets every visit; failures do not commit.

SOURCE_PREP, TARGET_SCREEN, TARGET_MECHANISM and TARGET_EXTENSION are NOT_RUN; external_review=NOT_RUN.
