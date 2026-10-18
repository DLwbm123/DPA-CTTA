# R75 preparation and boundaries

CPU covariance / blocked accumulation / regularized whitening / effective-head equivalence checks and the complete upstream import check passed. CUDA was not initialized during preparation.

One CPU-only preparation failure is retained: the synthetic whitening identity test mixed a float32 identity with float64 statistics at a strict float64 tolerance. The identity tensor was corrected to float64; the estimator and scientific settings were unchanged. Failed preparation cost 1.871509552 s; subsequent passing preparation 5.230889797 s. Later code edits are documented before target inference, with the full preflight repeated on the frozen code.

The first experiment is a readout-only transplant of WCW, not an official dense-segmentation reproduction or feedback-coupled learning system. All four outputs share the complete native CW_LSO adaptation trajectory. This deliberately preserves the existing strong baseline and isolates feature-readout headroom. Positive results would justify a separately specified learning-coupled follow-up; negative results cannot be described as failure of every possible WCW implementation.

The first GPU-process attempt failed the device-binding check before model loading, CUDA initialization, or target-image access: the neutral entry reapplied an empty `CUDA_VISIBLE_DEVICES` from the CPU preparation environment over the worker assignment. The entry now preserves the assignment supplied by the supervisor. The failed process and costs remain in the original run; a separate repaired run inherits its attempt receipt and the original deadline/budget. This is an entry-environment repair, not a scientific retry. Controller accounting accepts inherited receipts so setup cost cannot disappear.
