"""File/network boundary for score generation; pure callback is CPU-testable."""
import os,sys
from pathlib import Path

def callback(c,panel):
    root=Path(c['output_root']).resolve();data=Path(c['source_root']).resolve();checkpoint=Path(c['checkpoint_path']).resolve()
    images={Path(r['image_path']).resolve() for r in panel};opened=set()
    denied=[Path(v).resolve() for v in c['denied_prior_roots']]+[(root/'scorer').resolve()]
    def audit(event,args):
        if event=='socket.connect':raise PermissionError('network not permitted in score generation')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        utility=any(p.name==n+'.py' or (p.suffix=='.pyc' and p.name.startswith(n+'.')) for n in ('signal_score','e7_score','e8_score'))
        if utility or any(p.is_relative_to(v) for v in denied):raise PermissionError('labels, outcomes, oracle and scorer denied')
        if p.is_relative_to(data) and p!=checkpoint:
            if p not in images:raise PermissionError('only registered eight RGB')
            opened.add(str(p))
    return audit,opened

def install(c,panel):
    audit,opened=callback(c,panel);sys.addaudithook(audit);return opened
