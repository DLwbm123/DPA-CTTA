# R61 image-derived feedback: evidence and feasibility boundary

2026-10-17. Literature and local source inspection only; no experiment registered, no new target scoring or GPU work. Follows [the duplicate-mechanism audit](R61_EVIDENCE_DESIGN.md).

## Primary evidence

| Primary source | Relevant observation | Limit for this project |
|---|---|---|
| [Wong et al., ISBI 2012, vessel kinking](https://iacl.ece.jhu.edu/proceedings/isbi2012/pdfs/0001647.pdf), sections 1–2 | Pallor can differ from actual cup extent. The algorithm combines vessel kinks with pallor. | Its vessel classifier uses manually classified segments from five images and parameter search. It also uses eye-side information and high-resolution images. It is not a ready label-free component for this checkpoint/ROI protocol. |
| [Hu et al., 2016, color and vessel evidence](https://www.tandfonline.com/doi/abs/10.1080/21655979.2016.1227144), Methods and Experimental design | Combines radial color differences with vessel-bend confidence after vessel removal. | Evaluation uses ten high-confidence images chosen by an ophthalmologist from DRIVE. This does not establish robustness across our four domains. The pipeline needs vessel extraction, inpainting and boundary fitting; replacing those by raw edge magnitude is not a reproduction. |
| [Zhou et al., 2019, LSACM-SP](https://onlinelibrary.wiley.com/doi/10.1155/2019/8973287), sections 2–4 | Uses local intensity distributions and bias fields with nested disc/cup regions. Preprocessing removes vessels with B-COSFIRE and median inpainting; contour initialization uses Canny and circular Hough detection. | Source preprocessing uses a 400-pixel ROI, whereas our model protocol uses 512-pixel prepared inputs. Scale conversion, vessel-filter settings, energy coefficients and stopping rules must be resolved before implementation. A lower image-fitting energy is not evidence of higher anatomical Dice. |

The two PMC pages returned browser challenges; source details above came from the accessible publisher text and the conference PDF. No unavailable paper was treated as read. This is a bounded selection of relevant primary work, not an exhaustive novelty review.

## Mechanism distinction

The local search found prediction-derived Sobel structure fields in R36/R37/R45/R46. It found no named Chan-Vese, B-COSFIRE, vessel-inpainting or local intensity-likelihood implementation in the searched `src` and protocol files. This bounded search is not proof that no analogous component exists anywhere in the project.

Intensity distributions and vessel geometry are observable image properties outside the existing pseudo-label confidence and historical feature votes. They can suggest inward or outward contour changes depending on the local fit. They remain ambiguous: vessel boundaries are strong edges, illumination changes region statistics, and pallor does not uniquely identify the anatomical cup. An added energy term cannot be called reliable supervision merely because it is label-free.

## Selected feasibility question

Investigate only whether the published LSACM-SP preprocessing and energy can be specified reproducibly at the existing input resolution without target-label fitting or a new learned detector. Do not implement the supervised kinking classifier, substitute a tuned green-channel Sobel score, or scan morphology parameters.

The next design step has a finite scope: inspect the primary paper's equations/appendix and its cited B-COSFIRE implementation/defaults; record coefficients, termination, vessel extraction settings and a deterministic resolution policy. Inspect at most these dependencies, not a new broad literature search. If essential choices remain unspecified or require target-label tuning, mark this particular reproduction path infeasible under the current protocol and retain the gap explicitly.

If feasible, first register one CPU-only diagnostic on existing exposed SEARCH images with archived source/student candidates. Read images and generate all image-only rankings before opening labels. Freeze a matched comparator, a single configuration, costs, numeric allocation and adverse-cell gates before execution. No full adaptation trajectory is justified until the cue improves paired candidate decisions under that diagnostic. Using archived predictions avoids a new GPU replay; availability and identity matching must be checked before registration. No new numeric gate is claimed to have been registered by this design note.

Any resulting selector would be a new diagnostic inspired by an image model, not a faithful reproduction of the paper's segmentation performance. Even a positive result would remain development evidence. The current result is a more specific feasibility question, not an improvement claim.
