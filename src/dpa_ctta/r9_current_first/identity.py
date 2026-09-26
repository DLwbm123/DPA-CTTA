"""Exact executable inventory; no private runtime paths appear in public artifacts."""
import hashlib
import subprocess
from pathlib import Path
from .protocol import ROOT,SPEC_SHA,digest


def inventory(root=ROOT):
    root=Path(root)
    files=subprocess.check_output(['git','ls-files','src','scripts','configs'],cwd=root,text=True).splitlines()
    files.append('docs/review/r9_current_first/input/R9_SPEC_AND_MATRIX.json')
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(set(files)) if (root/p).is_file()}


def verify_runtime(config):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=config['code_sha'] or inventory()!=config['source_inventory']:raise ValueError('R9 exact runtime SHA/inventory mismatch')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','src','scripts','configs'],cwd=ROOT,text=True)
    if dirty:raise ValueError('R9 runtime source changes after binding')
    return dict(code_sha=head,spec_sha256=SPEC_SHA,inventory_sha256=digest(config['source_inventory']))
