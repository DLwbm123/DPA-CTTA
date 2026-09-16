# Bounded R5 review probes

This is not the original repository or its full test suite. It contains two Git-blob-verified source files read through the GitHub connector, and clearly marked verbatim function excerpts used in isolated CPU probes.

Run `python probe_review.py` for four positive invariant probes and two reproduced audit defects. Run `python probe_join.py` for the additional 33-row replay/join demonstration; this also reruns `probe_review.py`.

Dependencies: Python, PyTorch, NumPy. No network, real checkpoint, target/source images, labels, or CUDA are required. Reference run: Python 3.13.5 / Torch 2.10.0+cpu. Source implementation: 0c08cece6a91bdf5e3d06c9862dcd192df7787d8.

`passed=true` on a defect probe means the defect was successfully reproduced, NOT that production passed review. The smoke probe tests the original orchestration with mock dependency modules and a small real CPU Linear model; it is not an original ResUNet/GraTa run. The join probe uses the original replay/join excerpts and inherited validator, with a fixture-only identity comparator; it does not execute full recompute or a scientific gate.
