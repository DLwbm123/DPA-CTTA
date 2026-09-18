# A_PSF: method conformance

All rows implement the supplied design, not a claim of exact original-paper reproduction. Real source assets are PENDING.

| Specification | Production mapping | Formula and boundary |
|---|---|---|
| A1 | `source.a_basis / pooled_jacobian; numerics.predictive_basis/project` | Disjoint16+16 fit groups, 32 pooled readouts, 1024-dimensional GGN, rank16 predictive truncation; no random padding. |
| A2 | `r7_a_psf.PSF.observe` | 8x32 style and32x64 content codebooks, cosine/.2,128→64→32, bounded R. |
| A3 | `r7_a_psf.gaussian_filter / PSF.update` | Full stable16x16 F, diagonal Q, full SPD P, precision Cholesky; causal only. |
| A4 | `PSF.fit_loss / cal_loss; SourceTrainer` | Specified .1/.01/.05/.05 losses; content terms averaged by2; tau squared in R; independent STATIC. |

Shared: original up1/up3 FiLM1024, current BN, fit-only observer, float32 backbone/MLP and differentiable float64 inference, source fit/cal/val and support/query isolation. Full network and mathematical comparisons are in the final CPU logs.

Persistent online state: m16, P16x16, counter. No image, patient token, probability map or identity persists. States reset per trajectory; STATIC resets every visit; failures do not commit.

SOURCE_PREP, TARGET_SCREEN, TARGET_MECHANISM and TARGET_EXTENSION are NOT_RUN; external_review=NOT_RUN.
