# Cross-patient boundary objective: execution notes

Status: implementation and source-development execution preparation. Results are not available yet.

The user explicitly authorized content-disjoint development because the official RIM-ONE-r3 release did not establish patient linkage. This does not establish independent patients or absence of source-backbone pretraining exposure. See `PROTOCOL_ZH.md` for the amendment and fixed design. RDC and ENGRAM are disabled.

The implementation is in `src/dpa_ctta/r78_cross_patient/`. It uses the existing pinned ResUNet and CW_LSO integrations, requires the project's existing PyTorch/NumPy/SciPy/Pillow/batchgenerators environment, and introduces no dependency.

To reproduce preparation with locally authorized source data, supply records containing `sample_id`, `group_id` (existing image-content group), `image_relative`, `mask_relative`, and `image_size`. `prepare_metadata(records, source_root)` returns the training/image panel and the separate final scorer panel. Save them under the run's `private/panel.json` and `scorer/panel.json`. `configs/r78_cross_patient_v1.json` declares the public settings; bind private `output_root`, `source_root`, `checkpoint_path`, `gpu`, `gpu_cap_seconds=3600`, and `cpu_cap_seconds=900` locally. No patient identities or private asset paths are included in this delivery.

Run `dpa_ctta.r78_cross_patient.run` with `DIAG_CONFIG` pointing to the bound configuration and `DIAG_MODE=self_check` first, with CUDA hidden. Then use `DIAG_MODE=controller` and `DIAG_ENTRY` pointing to a neutral entry script for the same module. The existing subprocess runner executes one bounded GPU worker and, only after it retires successfully, one independent CPU scorer. It does not retry or expand the experiment. Keep interpreter and entry paths neutral in visible process arguments.

The CPU check exercises synthetic loss/state/gradient tests, the actual 512px model interface, matched-capacity consistency and native CW_LSO, exact zero-state parity, and native restoration. These are engineering checks, not method results. Final source labels are denied to the GPU worker. The scorer requires all 1,000 outer episodes and 736 readouts before opening final labels.

Public delivery contains only source, protocol, anonymized aggregates, resource summaries, and this report. Official dataset files, model weights, per-image predictions/scores, and identifying metadata remain private. The official archive README permits research use and prohibits redistribution without consent.

Official sources inspected: [release page](https://medimrg.webs.ull.es/), its linked `RIM-ONE-r3.zip` README and file directory, and [the authors' database paper](https://www.researchgate.net/publication/280673617_Interactive_Tool_and_Database_for_Optic_Disc_and_Cup_Segmentation_of_Stereo_and_Monocular_Retinal_Fundus_Images). The paper defines K as an image number, not a patient identifier. This establishes unresolved grouping, not proof that the dataset contains repeated patients.
