"""Generate disabled configuration and parsed DAG without model/data access."""
import json
from dpa_ctta.r9_current_first.protocol import ROOT, graph, disabled_config
out=ROOT/'docs/review/r9_current_first'
for name,data in [('TASK_GRAPH.json',graph()),('LAUNCH.disabled.json',disabled_config())]:
    (out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
print('R9: 43 sources, 610 core + 161 sensitivity slots; execution disabled')
