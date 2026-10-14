# SSL_CL data preparation for future CTTA validation

This prepares data only. It does not change R49, start an experiment, select a
source/target order, or establish a new independent test set.

| Dataset | Domains | Frozen payload | Labels / input |
|---|---|---|---|
| Fundus | REFUGE 400, RIM_ONE_r3 159, Drishti_GS 101 | 660 RGB images, 384×384 | 0 background / 1 disc rim / 2 cup; uint8 RGB divided by 255 |
| Prostate | RUNMC 30, BMC 30, I2CVB 19, HK 12, UCL 13, BIDMC 12 | 116 patient volumes, slices 256×256 | 0 background / 1 whole prostate; one input channel |
| M&Ms | Siemens 190, Philips 250, GE 150, Canon 100 phase volumes | 345 patients × ED/ES = 690 volumes, slices 384×384 | 0 background / 1 LV / 2 MYO / 3 RV; one input channel |

M&Ms includes 320 canonical patients (640 phase volumes) and 25 auxiliary
patients (50 GE phase volumes). Auxiliary25 remains unlabeled training only;
the held-out scorer views exclude it. MRI values retain the existing per-volume
0.5/99.5 percentile clipping and scaling to [-1,1]. M&Ms keeps the previously
frozen 320 mm field of view; preparation does not revisit its historical choice.

## Private package

The new package copies the frozen `h5/v1`, three training and three diagnostic
manifests, and original split files. It never edits their source. Original HDF5
metadata, preprocessing identifiers, and existing checksums remain intact; this
preparation does not recompute hashes or claim bytewise equivalence.

`views/seed{0,1,2}/{fundus,prostate,mnms}/{role}/online.csv` contains image paths
and original patient/case, split, domain, phase, and cohort metadata. It contains
no label paths or label hashes. Roles are `train_labeled`, `train_unlabeled`,
`val`, and `test`, preserving the original 20% labeled protocol.

`val/scorer.csv` and `test/scorer.csv` additionally contain the corresponding
label path for offline scoring. This is manifest separation, not an operating
system access-control guarantee: a future online worker must receive only its
image inputs. Original diagnostic manifests and labels remain private.

Paths in these CSVs are relative to `h5/v1`, e.g. `images/...` and `labels/...`.
Images use HDF5 key `image`; labels use `label`. Fundus is `[3,H,W]` / `[H,W]`;
MRI is `[Z,H,W]` for both. Convert MRI slices to `[1,H,W]` without renormalizing.
Retain patient/phase grouping and HDF5 processed spacing to reconstruct MRI
volumes for evaluation; slices and ED/ES are not independent patients. Empty
slices must not inflate the patient/volume score.

For a two-sigmoid fundus head, derive OD as `label > 0` and OC as `label == 2`.
This conversion is an offline label/metric convention, never an online decision.
MRI requires dataset-specific heads and losses; it cannot be passed directly to
the existing 512×512 RGB OD/OC host. No model adapter or checkpoint is created
by this data-only preparation.

## Reproduce preparation

Use an existing Python >=3.10 environment with numpy and h5py. Put the frozen
SSL_CL release at `PREP_SOURCE`, and choose a **new** `PREP_TARGET` directory on
approved mounted storage. Then run `scripts/prepare_sslcl_data.py` with those
environment variables. The target is create-only; partial output retains
`PREPARING` and must not be consumed. `READY.json` is written only after checks
finish. Copy the four frozen label-map/preprocess YAML files into `contracts/`
alongside it for interpretation; do not overwrite source metadata.

Validation covers patient disjointness across train/val/test, cohort eligibility,
role consistency, seed payload identities, original hidden-label policy, all
paired HDF5 identities/shapes, positive MRI spacing, and one decoded middle
slice/image per domain/phase. It does not decode every voxel or recompute hashes.
Synthetic checks cover those manifest boundaries, online label exclusion, and
refusal to overwrite an existing target. This is a data qualification, not a
backward pass, GPU profile, or accuracy experiment.

Only `SUMMARY.json` and this documentation are public. HDF5 files, original and
derived per-case manifests, split lists, identities, private locations, and logs
are not published.

## Evaluation boundary

REFUGE and Drishti_GS already occur in the CTTA development history. Changing
preprocessing or importing SSL_CL splits does not make them independent data.
Before any future experiment, freeze source training, CTTA development versus
held-out evaluation, domain orders, patient-level statistics, and budgets.
The existing three split seeds are data partitions, not the CTTA optimizer seeds.
The current R49 stream, protocol, source freeze and monitor remain unchanged.
