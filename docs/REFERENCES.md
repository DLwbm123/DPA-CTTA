# References and provenance

- DPA-CTTA v0 formulas and experimental gates: the reviewed repository bootstrap prompt and pre-training experiment plan supplied for this implementation.
- Integration source: `DLwbm123/CTTA`, commit `dbff0d985c6c95345d9fb78f5b1daef57b392564`, package `ctta-repro-suite`.
- Upstream model families used only through that pinned checkout: ResUNet34 for Fundus and PraNet for Polyp.
- PyTorch primitives used for autodiff JVPs, forward hooks, FFT descriptors, modules, and serialization: https://pytorch.org/docs/stable/

No claim about M0, M1, D0, or target performance is made in this implementation-only release.
