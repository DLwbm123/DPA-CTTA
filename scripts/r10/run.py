import os,json
from dpa_ctta.r10_use_write_rl.queue import run
config=json.load(open(os.environ['R10_CONFIG']))
run(config,os.environ['R10_WORKER_ENTRY'],os.environ['R10_NEUTRAL_PYTHON'])
