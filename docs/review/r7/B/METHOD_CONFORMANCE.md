# B_RCA: method conformance

All rows implement the supplied design, not a claim of exact original-paper reproduction. Real source assets are PENDING.

| Specification | Production mapping | Formula and boundary |
|---|---|---|
| B1 | `r7_b_rca.RCA.update` | U32 source-only uncentered SVD; stable W initialized.9I giving A=.855I; bounded G; only descriptor difference. |
| B2 | `r7_b_rca.correct / energy` | Exact5 proximal steps from delta0; kappa squared in residual and eta; lambda.01,mu.1; finite approximation. |
| B3 | `RCA.fit_loss / cal_loss; SourceTrainer` | Query Lseg +.1 codeMSE +.1 observationMSE +.001 offdiagonal normalized dictionary Gram. Kappa-only calibration. |
| B4 | `OnlineHost / inference_from_tensors` | Frozen calibrated eta checked; state z,d,counter restored with binding; PRED_ONLY is full-weight deployment ablation. |

Shared: original up1/up3 FiLM1024, current BN, fit-only observer, float32 backbone/MLP and differentiable float64 inference, source fit/cal/val and support/query isolation. Full network and mathematical comparisons are in the final CPU logs.

Persistent online state: z32, previous descriptor32, counter. No image, patient token, probability map or identity persists. States reset per trajectory; STATIC resets every visit; failures do not commit.

SOURCE_PREP, TARGET_SCREEN, TARGET_MECHANISM and TARGET_EXTENSION are NOT_RUN; external_review=NOT_RUN.
