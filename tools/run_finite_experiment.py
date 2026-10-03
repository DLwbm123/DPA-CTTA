"""Copy this tiny entry to a neutral path and supply private environment bindings."""
import json
import os
from pathlib import Path
import sys

config = Path(os.environ['RUN_ENV'])
for key, value in json.loads(config.read_text()).items():
    os.environ.setdefault(key, str(value))
sys.path.insert(0, os.environ['PYTHONPATH'])
from dpa_ctta.r16_evidence_correction.run import main
main()
