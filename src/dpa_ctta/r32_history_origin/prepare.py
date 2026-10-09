"""Build fixed diagnostic index lists from existing metadata, without masks."""
import json
import os
from pathlib import Path
from collections import Counter


def prepare(old, root):
    split=json.loads((old/'scorer/SPLIT.private.json').read_text()); online={}; scorer={}; public=[]
    for order in (0,1):
        rows=json.loads((old/f'scorer/FULL_o{order}.json').read_text()); cases=[]; named=[]
        image_rows=json.loads((old/f'private/ONLINE_o{order}.json').read_text())
        assert len(rows)==len(image_rows)==1951
        for i,domain in enumerate(('ORIGA','REFUGE_Valid')):
            positions=[j for j,r in enumerate(rows,1) if r['domain']==domain]
            assert positions==list(range(positions[0],positions[-1]+1))
            start,end=positions[0],positions[-1];previous=rows[start-2]['domain']
            same=list(range(start,start+64));cross=list(range(start-64,start));query=list(range(start+64,end+1))
            assert all(rows[j-1]['domain']==previous for j in cross)
            ids=[{rows[j-1]['image_sha256'] for j in seq} for seq in (same,cross,query)]
            assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
            assert all(rows[j-1]['image_sha256']==image_rows[j-1]['image_sha256'] for j in same+cross+query)
            case=dict(id=f'd{i}',same=same,cross=cross,query=query,pivot=same[-1]);cases.append(case)
            named.append(dict(case,domain=domain,cross_domain=previous))
            counts=Counter(split.get(rows[j-1]['image_sha256'],'CONTEXT') for j in query)
            public.append(dict(case_id=f'd{i}',order=order,domain=domain,cross_domain=previous,history_length=64,query_arrivals=len(query),query_start_within_domain=65,
                               original_prefix_updates=same[-1],query_SEARCH=counts['SEARCH'],query_REVIEW=counts['SEALED_REVIEW'],query_CONTEXT=counts['CONTEXT']))
        online[str(order)]=cases;scorer[str(order)]=named
        (root/f'private/ONLINE_o{order}.json').write_text(json.dumps(image_rows))
        (root/f'scorer/FULL_o{order}.json').write_text(json.dumps(rows))
    (root/'private/PLAN.json').write_text(json.dumps(online))
    (root/'scorer/PLAN.json').write_text(json.dumps(scorer))
    (root/'scorer/SPLIT.private.json').write_text(json.dumps(split))
    (root/'public/QUERY_REGISTRATION.json').write_text(json.dumps(dict(cases=public,domain_metadata_used_only_for_offline_design=True),indent=2))
    c=json.loads((old/'RESOLVED_CONFIG.json').read_text())
    for key in ('b_jobs','a_jobs'):
        c.pop(key,None)
    c.update(experiment_id='R32_HISTORY_ORIGIN',output_root=str(root),snapshot_input_root=str(old/'target'),protocol_path=str(root/'code/docs/protocols/R32_HISTORY_ORIGIN.md'),
             jobs=[dict(id=f's{s}_o{o}',seed=s,order=o) for s in c['seeds'] for o in (0,1)])
    (root/'private/config.json').write_text(json.dumps(c,indent=2))
    return public


if __name__=='__main__':
    print(json.dumps(prepare(Path(os.environ['R32_INPUT']),Path(os.environ['R32_OUTPUT'])),indent=2))
