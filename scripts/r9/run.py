"""Environment-only private configuration; disabled inputs fail before private asset I/O."""
import json
import os
from dpa_ctta.r9_current_first.queue import run
from dpa_ctta.r9_current_first.admission import require_authorized
if __name__=='__main__':
    with open(os.environ['R9_CONFIG']) as f:config=json.load(f)
    require_authorized(config)
    run(config,os.environ['R9_WORKER_ENTRY'],os.environ['R9_NEUTRAL_PYTHON'])
