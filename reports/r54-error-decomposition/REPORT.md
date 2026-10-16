# R54 frozen confusion-count diagnostic

{
  "oversegmentation_hypothesis_supported": true,
  "mean_fp_minus_fn_excess": 0.6624280719175436,
  "positive_pairs": 6,
  "independent_validation": false,
  "adaptation_runs": 0
}

All domains, channels, seeds and orders are retained in the tables. Rates are ratios of pooled pixel counts within each cell, not mean per-image rates or Dice. This is descriptive development evidence; no causal mechanism, deployed improvement or independent validation is established. No images, masks or new labels were read.

REFUGE_Valid OC: mean paired FP/GT excess +0.595111, FN/GT excess -0.067317; predicted/GT is1.998238 for CW_LSO versus1.335810 for VPTTA. The fixed diagnostic is supported in6/6 pairs. ORIGA OC has predicted/GT0.772609 for CW_LSO versus0.423626 for VPTTA; CW_LSO has much less missing foreground there. Uniform mask shrinkage is therefore not justified by this evidence. These ratios are pooled within domain/seed/order and then averaged; they are not average per-image area ratios.

Next: a separately registered spatial diagnostic will distinguish disconnected foreground fragments from false positives within the largest connected component. This does not establish that removing components improves Dice, nor authorize selecting per-domain rules.
