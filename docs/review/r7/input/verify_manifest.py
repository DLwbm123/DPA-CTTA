"""Verify every delivered file listed in MANIFEST.json without rewriting it."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
m=json.loads((root/'MANIFEST.json').read_text())
errors=[]
for row in m['files']:
 p=root/row['path']
 if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
  errors.append(row['path']+': missing or unsafe');continue
 b=p.read_bytes()
 if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:
  errors.append(row['path']+': byte/digest mismatch')
print(json.dumps({'verified_entries':len(m['files']),'errors':errors,'status':'PASS' if not errors else 'FAIL'},indent=2))
sys.exit(bool(errors))
