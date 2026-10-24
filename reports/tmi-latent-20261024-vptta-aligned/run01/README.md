# Five-source latent adaptation experiment

See REPORT_ZH.md for the negative result and PROTOCOL_ZH.md for the frozen training configuration. Data/evaluation conventions are in ../DATA_PROTOCOL_ZH.md. No medical images, masks, checkpoints, or per-image outputs are distributed.

To reproduce, supply the VPTTA-preprocessed Fundus directory and the nine CSV manifests. Copy config.example.json to a private run directory as config.private.json, replace its data_root placeholder, copy src/ into that directory, and set WORK_ROOT to that run directory. Dependencies include PyTorch, NumPy, Pillow, SciPy, scikit-learn and MedPy. Run src/worker.py with an available CUDA device and an independently enforced timeout/storage limit. The experiment has five source models, one seed, 200 epochs per source, and evaluates only after all five models are trained. Do not present partial output as the full five-source result.

src/metrics.py retains the upstream metric license header. The guardian/probe are supplied as the experiment's control implementation; private host-specific launch entries and credentials are excluded. The wrapper exit receipt confirms observed normal termination, not scientific effectiveness.
