# Pre-training audit

Status: **PASS for implementation-only review**. Training remains blocked pending `CODE_REVIEW_PASS`.

- Runtime: Python 3.12.9; PyTorch 2.6.0; CPU only; CUDA initialized: false.
- Reference: `DLwbm123/CTTA` at `dbff0d985c6c95345d9fb78f5b1daef57b392564`.
- Fundus injection paths: `up1, up3, seg_head`.
- Polyp injection paths: `ra4_conv5, ra3_conv4, ra2_conv4`.
- Tests: two independent runs, 143 tests each, 0 failures, 0 errors.
- Scientific payload SHA-256: `2751d89b26fb8f44e513697e17e4a6363af35a6902d4993e9031b65110c64f67` in both runs.
- Access boundary: source images=false, target images=false, source checkpoints=false, training=false.
- Audited pre-generated Git tree: `749f57705a1c10bae300138ffdeb6c275f2df9bf`.
- Per-file SHA-256 manifest: `audit/file_manifest.json` (43 entries).

## Known limitations

- No source-statistics scaler has been fitted.
- No real-data training or target evaluation has been run.
- Diagonal precision only.
- Public reference checkout omits ctta-repro-suite/third_party; integration uses a temporary symlink view of VPTTA sources from the same pinned commit.
- Final commit tree is reported by GitHub because embedding it in its own tree is self-referential.
