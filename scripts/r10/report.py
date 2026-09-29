import os,json
from pathlib import Path
from dpa_ctta.r10_use_write_rl.report import full_report
from dpa_ctta.r9_current_first.storage import write_json
root=Path(json.load(open(os.environ['R10_CONFIG']))['output_root'])
write_json(root/'public-aggregate.json',full_report(root))
