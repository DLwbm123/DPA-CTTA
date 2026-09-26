import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dpa_ctta.r9_current_first import queue,assets
from dpa_ctta.r9_current_first.ledger import Ledger
from dpa_ctta.r9_current_first.protocol import CAPS
from dpa_ctta.r9_current_first.storage import write_json

class Queue(unittest.TestCase):
    def test_finite_dependencies_single_infra_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'r9';root.mkdir();identity={'runtime':'synthetic'}
            nodes=[dict(id='first',kind='source',needs=[],resource='gpu'),dict(id='last',kind='bind_assets',needs=['first'],resource='cpu')]
            budget=dict.fromkeys(CAPS,0);budget['gpu_seconds']=10
            config=dict(profile={'node_budgets':{n['id']:budget for n in nodes},'peak_vram_bytes':{'first':1024}},gpu_assignments=[{'physical_id':9,'uuid':'GPU-synthetic'}])
            executed=[]
            class Process:
                pid=12345
                def __init__(self,args,env,**kwargs):
                    assert args==['/tmp/e/bin/python','/tmp/w.py']
                    packet=json.loads(Path(env['R9_PACKET']).read_text());node=packet['node']['id'];executed.append(node)
                    fail=node=='first' and executed.count('first')==1
                    f={'class':'INFRASTRUCTURE','reason':'synthetic EIO','evidence':'test receipt'} if fail else None
                    ledger=Ledger(root/'ledger',identity,[9]);actual=dict.fromkeys(CAPS,0)
                    ledger.observe(packet['attempt'],packet['token'],actual,settle=True,failed=fail)
                    (root/'attempts').mkdir(exist_ok=True)
                    write_json(root/'attempts'/(packet['attempt']+'.json'),dict(schema='R9_ATTEMPT_V1',identity=identity,node=node,attempt=packet['attempt'],status='FAILED' if fail else 'COMPLETE',failure=f))
                def wait(self,timeout=None):return 0
            with patch.object(queue,'require_authorized',return_value=root),patch.object(queue,'verify_runtime',return_value=identity),patch.object(queue,'graph',return_value={'nodes':nodes}),patch.object(queue.subprocess,'Popen',Process),patch.object(assets,'available_memory',return_value=1024):
                state=queue.run(config,'/tmp/w.py','/tmp/e/bin/python')
            self.assertEqual(executed,['first','first','last']);self.assertEqual(state['status'],'COMPLETE')
            ledger=Ledger(root/'ledger',identity,[9]);self.assertEqual(ledger.total(ledger._read())['gpu_seconds'],10)

if __name__=='__main__':unittest.main()
