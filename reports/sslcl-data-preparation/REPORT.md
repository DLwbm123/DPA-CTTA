# SSL_CL data preparation — 2026-10-14

Status: **DATA_READY_NO_EXPERIMENT**. The private package is on the approved
NAS, with a completed READY receipt and no remaining preparation process.
No GPU, training, adaptation, or model scoring was run. R49 was not modified.

| Dataset | Patient/case groups | Images or phase volumes | Images/slices |
|---|---:|---:|---:|
| Fundus (3 domains) | 660 | 660 images | 660 |
| Prostate (6 sites) | 116 | 116 volumes | 3,887 |
| M&Ms (4 vendors) | 345 | 690 ED/ES volumes | 7,908 |

M&Ms evaluation eligibility is limited to the original 320 canonical patients
(640 phase volumes); 25 auxiliary patients remain unlabeled training only.
All original split seeds 0/1/2, class semantics, MRI normalization, spatial
metadata, and preprocessing are preserved. Fundus OD is rim ∪ cup; OC is cup.

Prepared payload: **2,932 HDF5 files, 1,843,390,563 bytes**, plus frozen source
manifests/splits and interpretation contracts. There are 36 image-only manifest
views and 18 offline val/test scorer views. Patient/case identities, paths,
split lists, images, labels, and private logs are not published.

Checks passed:

- Synthetic release preparation and rejection of duplicate cases, unsafe paths,
  patient split/role leakage, auxiliary evaluation leakage, and target overwrite.
- All three source seed manifests preserve patient train/val/test and
  labeled/unlabeled separation; cross-seed payload identities agree.
- All 1,466 image/label pairs have readable HDF5 metadata, matching identities
  and shapes; MRI processed spacing is positive and finite.
- 17 domain/phase samples decode with finite intensities and valid label values.
- Derived online manifest schemas exclude label paths and label hashes.

The data were copied to the approved NAS after a successful NFS mount,
capacity, and create/write/read check. No bulk copy was made to the GPU host:
its inspected destination resolved to the root overlay filesystem. Existing
files and other jobs were left untouched. This package is data-ready on NAS,
not yet deployed as a new runnable dataset on that GPU host.

Limits: no every-voxel decode, new full-file hashes, model adapter, source
checkpoint training, GPU profile, or accuracy result. Manifest separation does
not enforce OS-level label access control. The current fundus host is RGB
512×512 with two sigmoid channels; MRI needs a separate input/head/loss/volume
scoring adaptation. REFUGE and Drishti_GS overlap existing CTTA development
exposure, so this preparation does not establish independent validation.

See [data contract](../../docs/SSLCL_DATA_PREPARATION.md) and
[aggregate summary](SUMMARY.json). No new experimental matrix or source/target
domain order has been selected or launched.
