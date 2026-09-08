"""M1-only exploratory registration; old source whitelist and splits remain intact."""
import collections
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
from .source_io import registered_source, read_pixels
from .source_pilot import DEFAULT_CONFIG, SEGMENTS
from .source_pilot_release import file_identity, check_registered_files, digest

DOMAINS={'fundus':['REFUGE','ORIGA','REFUGE_Valid','Drishti_GS'],
         'polyp':['CVC-ClinicDB','ETIS-LaribPolypDB','Kvasir-SEG']}
SEED=20260907


def ranked(values, prefix):
    return sorted(values,key=lambda v:(hashlib.sha256((prefix+v).encode()).hexdigest(),v))


def select_targets(rows, task, source_images):
    """No label decoding or score input. Conflict/dedup decisions precede sampling."""
    by_image=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r['domain'] in DOMAINS[task]: by_image[r['image_sha256']].append(dict(r,manifest_index=i))
    pools={d:[] for d in DOMAINS[task]};excluded=[]
    for sha,aliases in by_image.items():
        reason=None
        if len({r['mask_sha256'] for r in aliases})!=1: reason='conflicting_mask_digest'
        elif sha in source_images: reason='source_image_overlap'
        elif any(r.get('sealed_final') is True or r.get('role') in {'sealed_final','forbidden_development'} for r in aliases): reason='explicit_protected_role'
        if reason:
            excluded.append(dict(image_sha256=sha,reason=reason));continue
        domain=min((r['domain'] for r in aliases),key=DOMAINS[task].index)
        representative=min((r for r in aliases if r['domain']==domain),key=lambda r:r['sample_id'])
        pools[domain].append(dict(representative,group_id=sha,original_split=representative['split'],
            patient_linkage=representative.get('patient_id','UNKNOWN'),video_linkage=representative.get('video_id','UNKNOWN')))
        if len({r['domain'] for r in aliases})>1: excluded.append(dict(image_sha256=sha,reason='later_domain_alias',kept_domain=domain))
    selected=[]; roles=[];counts={}
    for domain,pool in pools.items():
        keys=ranked([r['group_id'] for r in pool],f'M1:{SEED}:{task}:{domain}:')[:32]
        chosen=sorted((r for r in pool if r['group_id'] in keys),key=lambda r:r['manifest_index'])
        selected.extend(dict(r,experiment_role='M1_TARGET_DEV') for r in chosen)
        roles.extend(dict(sample_id=r['sample_id'],group_id=r['group_id'],role='M1_TARGET_DEV' if r['group_id'] in keys else 'NOT_USED_IN_M1') for r in pool)
        counts[domain]=dict(eligible=len(pool),selected=len(chosen))
    return selected,roles,excluded,counts


def register(old_directory,target_manifests):
    old=Path(old_directory); prior=json.loads((old/'registration.json').read_text())
    receipt=json.loads((old/'receipt.pilot.private.json').read_text())
    if digest(old/'registration.json')!=receipt['registration_sha256']: raise ValueError('prior registration mismatch')
    check_registered_files(prior)
    spec=json.loads((old/'source_spec.private.json').read_text());cfg=json.loads(DEFAULT_CONFIG.read_text())
    result=dict(tasks={},identities=[],experiment_role='exploratory_target_development',
        historical_target_exposure='documented_previous_evaluation',untouched_final_test_claim=False,
        clinical_validation_claim=False,known_association_source='inspected existing manifests; absent fields UNKNOWN')
    seen=set()
    for task,entry in spec['tasks'].items():
        selected=registered_source(entry['manifest'],entry['split'],cfg['tasks'][task]['csv_specs'],entry['data_root'],task)
        if selected!=prior['tasks'][task]['selected']: raise ValueError('source selection changed')
        source=json.loads(Path(entry['manifest']).read_text()); split=json.loads(Path(entry['split']).read_text())
        by_id={r['sample_id']:r for r in source}
        groups={}
        for role in ('basis_train','critic_train'):
            gs=collections.defaultdict(list)
            for item in split['splits'][role]['records']:
                r=dict(by_id[item['sample_id']],split_role=role)
                for field in ('image','mask'):
                    parts=Path(r[field+'_path']).parts; relative=Path(*parts[parts.index(entry['source']):])
                    path=(Path(entry['data_root'])/relative).resolve()
                    if not path.is_relative_to(Path(entry['data_root']).resolve()/entry['source']): raise ValueError('source path escape')
                    r[field+'_path']=str(path)
                gid=hashlib.sha256((r['image_sha256']+r['mask_sha256']).encode()).hexdigest()
                r['group_id']=gid;gs[gid].append(r)
            groups[role]={g:min(rs,key=lambda r:r['sample_id']) for g,rs in gs.items()}
        train=[groups['critic_train'][g] for g in ranked(groups['critic_train'],f'M1:{SEED}:{task}:critic_train:')[:64]]
        if not train: raise ValueError('EMPTY_LEGAL_SOURCE_TRAIN_QUERY_POOL')
        proxy_keys={r['group_id'] for r in selected['proxy']}
        history_pool=[groups['basis_train'][g] for g in ranked(groups['basis_train'],f'M1:{SEED}:{task}:history:') if g not in proxy_keys]
        fallback=not history_pool
        if fallback: history_pool=train
        history=[history_pool[i%len(history_pool)] for i in range(32)]
        if {r['image_sha256'] for r in train}&{r['image_sha256'] for r in selected['proxy']}: raise ValueError('query proxy image overlap')
        targets=json.loads(Path(target_manifests[task]).read_text())
        chosen,roles,excluded,counts=select_targets(targets,task,{r['image_sha256'] for r in source})
        for r in chosen:
            for field in ('image','mask'):
                p=Path(r[field+'_path']).resolve()
                if not p.is_relative_to(Path(entry['data_root']).resolve()/r['domain']): raise ValueError('target path outside declared domain')
        # Explicit M1 selected-byte checks; prior metadata-only checks are not relabeled.
        for r in selected['proxy']+selected['query']+train+history+chosen:
            for field in ('image','mask'):
                p=r[field+'_path']
                if p not in seen:
                    result['identities'].append(file_identity(p,r[field+'_sha256']));seen.add(p)
                    with Image.open(p) as im:
                        if list(im.size)!=r['image_size']: raise ValueError('registered geometry mismatch')
                        if field=='mask' and im.mode not in {'L','P','RGB','RGBA','1'}: raise ValueError('unsupported mask container encoding')
                        im.verify()  # Container integrity only; target mask semantics belong to evaluator.
            if r in chosen: read_pixels(r['image_path'],task,r['image_size'])
        checkpoint=file_identity(entry['checkpoint'],entry['sealed_checkpoint_sha256'])
        result['identities'].append(checkpoint)
        rng=random.Random(SEED);state_order=list(range(32));rng.shuffle(state_order)
        query_order=list(range(len(train)));rng.shuffle(query_order)
        episodes=[dict(episode=i+1,state_index=state_order[i%32],query_index=query_order[i%len(train)],transform=SEGMENTS[i%4]) for i in range(600)]
        result['tasks'][task]=dict(source=entry['source'],checkpoint=checkpoint,proxy=selected['proxy'],source_query=selected['query'],
            train_query=train,history=history,history_fallback_to_critic_train=fallback,episodes=episodes,target=chosen,
            target_roles=roles,target_exclusions=excluded,target_counts=counts,checkpoint_training_membership='UNKNOWN')
    return result
