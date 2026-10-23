# Independent latent metric r03

Read PROTOCOL_ZH.md before execution. `src/train.py` imports the frozen r01 implementation using private `parent_root`; r01 code and final source checkpoints are required, not copied. Configure the same source dataset and split, GPU1, two fixed seeds, 500 episodes each. Run via the neutral supervisor with WORK_ROOT set in the environment. This run learns only a positive latent metric on source-fit labels; no target labels or direct-comparator mechanisms.

`src/analyze.py ROOT` performs aggregate-only analysis after completion without importing a model. Private detailed rows and checkpoints stay on the server. No performance or superiority claim before frozen gates pass.
