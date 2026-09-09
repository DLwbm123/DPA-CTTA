"""Full legal manifest pools; P1 bytes reused, only newly selected assets verified."""
import collections
import json
from pathlib import Path
from PIL import Image
from .m1_data import DOMAINS
from .m2_registration import anchor
from .p1_data import load_proxy,expected
from .source_io import read_pixels,source_proxy
from .source_pilot_release import digest,check_registered_files,file_identity
from .host_diagnostic_run import private_json

PARENTS=('A','O2','D4')
VIEWS={'EA':'A','EO2':'O2','ED4':'D4'}
ARMS=('N','A','EA','O2','EO2','D4','ED4')
SUBSETS=('remaining_dev','legacy_dev','p1_extension_dev','all_dev')


def select_full(rows,task,source,p1,old_exclusions=(),old_roles=()):
    protected={'sealed_final','forbidden_development'};sources={r['image_sha256'] for r in source}
    blocked={r['image_sha256'] for r in old_exclusions if r['reason']!='later_domain_alias'}|{r['group_id'] for r in old_roles if r['role'] in protected}
    links={k:{str(r[k]) for r in source+rows if r.get(k) not in [None,'','UNKNOWN'] and (r in source or r.get('sealed_final') or r.get('role') in protected)} for k in ['patient_id','video_id']}
    old={r['image_sha256']:r for r in p1};aliases=collections.defaultdict(list);excluded=[];selected=[]
    for i,r in enumerate(rows):
        if r['domain'] in DOMAINS[task]:aliases[r['image_sha256']].append(dict(r,manifest_index=i))
    for sha,rs in aliases.items():
        reason='source_overlap' if sha in sources else 'previous_exclusion' if sha in blocked else None
        if len({r['mask_sha256'] for r in rs})!=1:reason='mask_conflict'
        if any(r.get('sealed_final') or r.get('role') in protected for r in rs):reason='protected_role'
        if any(str(r.get(k,'UNKNOWN')) in values for r in rs for k,values in links.items()):reason='protected_association'
        if reason:excluded.append(dict(image_sha256=sha,reason=reason,previously_in_P1=sha in old));continue
        domain=min((r['domain'] for r in rs),key=DOMAINS[task].index);r=min((r for r in rs if r['domain']==domain),key=lambda r:r['sample_id'])
        if sha in old and any(r[k]!=old[sha][k] for k in ['domain','sample_id','mask_sha256','manifest_index']):raise ValueError('previous P1 representative drift')
        subset=('legacy_dev' if old[sha]['subset']=='legacy_dev' else 'p1_extension_dev') if sha in old else 'remaining_dev'
        selected.append(dict(r,group_id=sha,subset=subset,original_split=r['split'],patient_linkage=r.get('patient_id','UNKNOWN'),video_linkage=r.get('video_id','UNKNOWN')))
    selected.sort(key=lambda r:(DOMAINS[task].index(r['domain']),r['manifest_index']))
    counts={d:{s:sum(r['domain']==d and (s=='all_dev' or r['subset']==s) for r in selected) for s in SUBSETS} for d in DOMAINS[task]}
    missing=[dict(group_id=sha,reason=next((e['reason'] for e in excluded if e['image_sha256']==sha),'absent_from_manifest')) for sha in old if sha not in {r['group_id'] for r in selected}]
    return selected,counts,excluded,missing


def register(p1,spec_path,manifests,out):
    p1=Path(p1);out=Path(out);receipt=json.loads((p1/'receipt.run.json').read_text())
    if receipt['commit']!='f65e119016f2d6a32cd0cffbee2bb2e94f570c2d' or json.loads((p1/'verification.json').read_text())['status']!='P1_NO_DD_COMPARISON_COMPLETE':raise ValueError('P1 reference')
    if digest(p1/'registration.json')!=receipt['registration_sha256']:raise ValueError('P1 registration identity')
    prior=json.loads((p1/'registration.json').read_text());check_registered_files(prior)
    oldid={i['path']:i for i in prior['identities']};spec=json.loads(Path(spec_path).read_text())
    m4=json.loads((Path(prior['m4_directory'])/'registration.json').read_text());m1=json.loads((Path(m4['old_directory'])/'registration.json').read_text())
    reg=dict(tasks={},identities=[],p1_directory=str(p1),scheduler='shared_current_visit',checks=dict(reused_P1_files=0,new_verified_files=0),limitations=['exploratory full development pool, not globally unseen','UNKNOWN patient/video links','single seed; two orders share contents'])
    seen=set()
    for task,r in prior['tasks'].items():
        entry=spec['tasks'][task];manifest=Path(manifests[task]);rows=json.loads(manifest.read_text());source=json.loads(Path(entry['manifest']).read_text())
        selected,counts,excluded,missing=select_full(rows,task,source,r['target'],m1['tasks'][task]['target_exclusions'],m1['tasks'][task]['target_roles'])
        for row in selected:
            fresh=False
            for field in ['image','mask']:
                path=Path(row[field+'_path']).resolve();key=str(path)
                if not path.is_relative_to(Path(entry['data_root']).resolve()/row['domain']):raise ValueError('target path')
                if key in seen:continue
                seen.add(key)
                if key in oldid:
                    item=oldid[key]
                    if item['sha256']!=row[field+'_sha256']:raise ValueError('old selected digest drift')
                    reg['checks']['reused_P1_files']+=1
                else:
                    item=file_identity(path,row[field+'_sha256']);fresh=True
                    with Image.open(path) as im:
                        if list(im.size)!=row['image_size'] or (field=='mask' and im.mode not in ['L','P','RGB','RGBA','1']):raise ValueError('selected encoding/geometry')
                        im.verify()
                    reg['checks']['new_verified_files']+=1
                reg['identities'].append(item)
            if fresh:read_pixels(row['image_path'],task,row['image_size'])
        assets={a:r['artifacts'][a] for a in ['O2','D4']}
        needed={r['checkpoint']['path'],r['history']['path'],*[x['path'] for x in assets.values()],*[v[f+'_path'] for v in r['proxy'] for f in ['image','mask']]}
        reg['identities'] += [oldid[p] for p in needed if p not in seen]
        reg['identities'] += [anchor(manifest),anchor(entry['manifest'])]
        reg['tasks'][task]=dict(target=selected,counts=counts,exclusions=excluded,missing_P1_groups=missing,checkpoint=r['checkpoint'],history=r['history'],proxy=r['proxy'],artifacts=assets)
        real=source_proxy(r['proxy'],task)
        for a in assets:load_proxy(reg['tasks'][task],task,a,real)
    reg['identities'].append(anchor(p1/'registration.json'))
    G=sum(len(r['target']) for r in reg['tasks'].values())
    if not 0<G<=3759:raise ValueError('full-pool size outside inventory')
    reg['budget']=dict(groups=G,records=14*G,online=6*G,teacher_forwards=2*G,smoke_online=12,outer=0,inner=0)
    private_json(out/'registration.json',reg);return reg
