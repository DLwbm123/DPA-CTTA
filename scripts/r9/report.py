import json
import os
from pathlib import Path
from dpa_ctta.r9_current_first.report import full_report
if __name__=='__main__':
    result=full_report(os.environ['R9_OUTPUT_ROOT'])
    Path(os.environ['R9_PUBLIC_REPORT']).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
