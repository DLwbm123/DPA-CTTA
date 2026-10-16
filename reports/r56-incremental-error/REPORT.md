# R56 frozen confusion-count diagnostic

Baseline: C

{
  "oversegmentation_hypothesis_supported": false,
  "mean_fp_minus_fn_excess": 0.007010670454430108,
  "positive_pairs": 3,
  "independent_validation": false,
  "adaptation_runs": 0
}

All domains, channels, seeds and orders are retained in the tables. Rates are ratios of pooled pixel counts within each cell, not mean per-image rates or Dice. This is descriptive development evidence; no causal mechanism, deployed improvement or independent validation is established. No images, masks or new labels were read.

REFUGE_Valid OC mean pooled predicted/GT: CW_LSO1.998238 versus C1.991227. Incremental FP/GT is+0.007066 and FN/GT is+0.000055. Primary d=0.007011 is below the fixed0.05 threshold and only3/6 pairs have positive d. The incremental excess hypothesis fails its gate. This is not equivalence, nor separate identification of W and LSO. The large excess seen relative to VPTTA is already present in C; focus subsequent investigation on the shared adaptive path versus source-model bias.

No further changes to fusion, morphology, or W/LSO are justified by these diagnostics alone. The next scheduled research step must first inspect existing source/checkpoint controls and exact C inference/update semantics, reuse compatible evidence if available, then register a matched source-versus-adaptive diagnostic before any new execution. No new GPU run is claimed here.
