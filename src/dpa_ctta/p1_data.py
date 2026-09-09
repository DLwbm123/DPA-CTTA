"""Score-blind P1 extension from existing manifests, with selected-byte registration."""
import collections
import json
from pathlib import Path
from PIL import Image
import torch
from .m1_data import DOMAINS,ranked
from .m4_registration import load_registered as load_m4
from .m2_registration import anchor
from .source_pilot_release import digest,file_identity,check_registered_files
from .source_io import source_proxy,read_pixels
from .proxy_loss import FixedProxy,ProxyProvenance
from .host_diagnostic_run import private_json

ARMS=('N','A','ENS_A','SA','ENS_SA','O2','D4','L4')
PARENTS={'N':'N','A':'A','ENS_A':'A','SA':'SA','ENS_SA':'SA','O2':'O2','D4':'D4','L4':'L4'}


def select_extension(rows,task,source,legacy,old_exclusions=(),old_roles=()):
    sources={r['image_sha256'] for r in source};used={r['image_sha256'] for r in legacy}
    protected={'sealed_final','forbidden_development'}
    blocked={r['image_sha256'] for r in old_exclusions if r['reason']!='later_domain_alias'}
    blocked|={r['group_id'] for r in old_roles if r['role'] in protected}
    links={key:{str(r[key]) for r in source+rows if key in r and r[key] not in [None,'','UNKNOWN'] and (r in source or r.get('sealed_final') or r.get('role') in protected)} for key in ['patient_id','video_id']}
    aliases=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r['domain'] in DOMAINS[task]:aliases[r['image_sha256']].append(dict(r,manifest_index=i))
    pools={d:[] for d in DOMAINS[task]};exclusions=[]
    for sha,rs in aliases.items():
        reason='legacy_dev' if sha in used else 'source_overlap' if sha in sources else 'M1_exclusion' if sha in blocked else None
        if len({r['mask_sha256'] for r in rs})!=1:reason='mask_conflict'
        if any(r.get('sealed_final') or r.get('role') in protected for r in rs):reason='protected_role'
        if any(str(r.get(k,'UNKNOWN')) in values for r in rs for k,values in links.items()):reason='protected_known_association'
        if reason:exclusions.append(dict(image_sha256=sha,reason=reason));continue
        domain=min((r['domain'] for r in rs),key=DOMAINS[task].index)
        r=min((r for r in rs if r['domain']==domain),key=lambda r:r['sample_id'])
        pools[domain].append(dict(r,group_id=sha,original_split=r['split'],patient_linkage=r.get('patient_id','UNKNOWN'),video_linkage=r.get('video_id','UNKNOWN'),subset='extension_dev'))
    result=[];counts={}
    for domain,pool in pools.items():
        keys=set(ranked([r['group_id'] for r in pool],f'P1:20260907:{task}:{domain}:')[:32])
        old=[dict(r,subset='legacy_dev') for r in legacy if r['domain']==domain]
        new=[r for r in pool if r['group_id'] in keys]
        result+=sorted(old+new,key=lambda r:r['manifest_index'])
        counts[domain]=dict(legacy_dev=len(old),extension_dev=len(new),eligible_extension=len(pool),combined=len(old)+len(new))
    if len({r['image_sha256'] for r in result})!=len(result):raise ValueError('cross-domain duplicate')
    return result,counts,exclusions


def expected(reg,task,order):
    from .m4_sequences import target_order
    return target_order(reg['tasks'][task]['target'],order)


def register(m4,spec_path,manifests,out):
    m4=Path(m4);out=Path(out);old,registered=load_m4(m4)
    if json.loads((m4/'verification.json').read_text())['status']!='M4_TRAJECTORY_COMPARISON_COMPLETE':raise ValueError('M4 incomplete')
    receipt=json.loads((m4/'receipt.run.json').read_text())
    if receipt['commit']!='f88e99d212e914b74a7c521bac8c418fc73e5d6c' or digest(m4/'registration.json')!=receipt['registration_sha256']:raise ValueError('M4 identity')
    spec=json.loads(Path(spec_path).read_text());reg=dict(tasks={},identities=[],m4_directory=str(m4),limitations=['exploratory development; extension unused by M1-M4, not project-wide untouched','UNKNOWN patient/video linkage','single seed; same contents across orders'])
    m1=json.loads((Path(old['old_directory'])/'registration.json').read_text());seen=set()
    for task,r in registered['tasks'].items():
        entry=spec['tasks'][task];manifest=Path(manifests[task]);source=json.loads(Path(entry['manifest']).read_text());rows=json.loads(manifest.read_text())
        target,counts,excluded=select_extension(rows,task,source,r['target'],m1['tasks'][task]['target_exclusions'],m1['tasks'][task]['target_roles'])
        for row in target:
            for f in ['image','mask']:
                path=Path(row[f+'_path']).resolve()
                if not path.is_relative_to(Path(entry['data_root']).resolve()/row['domain']):raise ValueError('domain path')
                if str(path) not in seen:
                    reg['identities'].append(file_identity(path,row[f+'_sha256']));seen.add(str(path))
                    with Image.open(path) as im:
                        if list(im.size)!=row['image_size'] or (f=='mask' and im.mode not in ['L','P','RGB','RGBA','1']):raise ValueError('selected encoding/geometry')
                        im.verify()
            read_pixels(row['image_path'],task,row['image_size'])
        artifacts={}
        for arm in ['O2','D4','L4']:
            item=old['artifacts'][task]['O2'] if arm=='O2' else dict(path=str(m4/f'{task}_{arm}_150.pt'),sha256=json.loads((m4/'final_artifacts.frozen.json').read_text())[task][arm])
            artifacts[arm]=file_identity(item['path'],item['sha256']);reg['identities'].append(artifacts[arm])
        reg['identities'] += [identity for identity in m1['identities'] if identity['path'] in {v[f+'_path'] for v in r['proxy'] for f in ['image','mask']}]
        reg['identities'] += [r['checkpoint'],anchor(manifest),anchor(entry['manifest']),anchor(Path(old['old_directory'])/'registration.json')]
        history=old['tasks'][task]['history'];reg['identities'].append(history)
        reg['tasks'][task]=dict(target=target,counts=counts,exclusions=excluded,checkpoint=r['checkpoint'],proxy=r['proxy'],artifacts=artifacts,history=history)
        real=source_proxy(r['proxy'],task)
        for arm in artifacts:load_proxy(reg['tasks'][task],task,arm,real)
    G=sum(len(r['target']) for r in reg['tasks'].values());reg['budget']=dict(groups=G,records=16*G,online=10*G,smoke_online=12,outer=0,inner=0)
    private_json(out/'registration.json',reg);return reg


def load_proxy(reg,task,arm,real=None):
    item=reg['artifacts'][arm];check_registered_files({'identities':[item]})
    artifact=torch.load(item['path'],map_location='cpu',weights_only=True)
    if artifact.get('outer_step',artifact.get('episode'))!=(600 if arm=='O2' else 150):raise ValueError('proxy checkpoint step')
    real=source_proxy(reg['proxy'],task) if real is None else real
    pixels=artifact['pixels']
    if pixels.shape!=real.pixel_rgb.shape or pixels.dtype!=torch.float32 or len(pixels)!=4 or not torch.isfinite(pixels).all() or ((pixels<0)|(pixels>1)).any():raise ValueError('proxy shape/values')
    return FixedProxy(pixels,real.mask,real.signed_distance,ProxyProvenance.SOURCE)
