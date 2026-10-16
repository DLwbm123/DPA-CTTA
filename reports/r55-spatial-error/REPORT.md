# R55 spatial error attribution

{
  "fragment_hypothesis_supported": false,
  "passing_pairs": 0,
  "total_pairs": 6,
  "independent_validation": false
}

Fixed 8-connected largest-component diagnostic; all domain/channel/order/seed cells retained. No new training and no prediction replacement. Removing fragments is not established as an efficacy improvement. All results are development evidence.

REFUGE_Valid OC: disconnected components contain only1.315%-1.720% of false-positive pixels across the six pairs, far below the fixed50% threshold. Removed true-positive pixels are0.046%-0.085% of GT. All six pairs fail the primary diagnostic. Thus most false positives remain in the largest connected region; this does not identify whether the cause is calibration, shape, alignment, or adaptation drift. Do not promote largest-component filtering as a solution.

Next fixed diagnostic: compare existing C and CW_LSO confusion counts to determine whether the incremental W/LSO change introduces the excess, before changing that mechanism.
