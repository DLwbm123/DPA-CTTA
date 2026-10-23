# Independent latent inference pilot

Source-supervised, spatial anatomy / acquisition-style representation with a signed-distance segmentation decoder and conditional image renderer. Test-time inference freezes all network weights and changes only current-case latent variables; optional continual state consists of bounded style mixture sufficient statistics.

Implementation: `src/train.py`; durable controller: `src/supervise.py`; content-paired analysis: `src/analyze.py`. Runtime requires PyTorch, NumPy, SciPy and Pillow. Set `WORK_ROOT` to a dedicated run folder containing a private configuration, then use the neutral-entry controller on the authorized host. The controller refuses a completed or live duplicate run and preserves its original budget across technical retries.

This pilot trains from scratch on an existing, repeatedly used source-development split. It does not establish independent-patient generalization or superiority to published methods. Synthetic gradient checks are separate from real-image training and scientific evaluation. See `PROTOCOL_ZH.md` for exact limits and gates.

No private configuration, per-content rows, raw data, image paths, checkpoints, or third-party papers belong in public delivery. Only source, protocol, anonymous aggregates and sanitized execution receipts may be published.
