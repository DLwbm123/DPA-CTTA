"""One isolated filesystem scan. A guardian must bound this process externally."""
import json, os
from pathlib import Path

root = Path(os.environ['WORK_ROOT'])
if not root.is_dir():
    raise RuntimeError('Work storage unavailable')
total = 0
def fail(error): raise error

for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
    for name in files:
        try:
            total += os.lstat(os.path.join(directory, name)).st_blocks * 512
        except FileNotFoundError:
            pass
print(json.dumps({'allocated_bytes': total}), flush=True)
