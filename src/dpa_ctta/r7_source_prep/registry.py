"""Metadata-first source registry and same-byte verified CPU decoding.

No directory discovery, target loading, or checkpoint loading on import/audit.
"""
import copy,hashlib,io,json,os,re,stat
from pathlib import Path
from ..r7_shared.context import json_digest
from ..r7_shared.source import split,SourceData,Record
from ..r1.plan import registration_digest

SOURCE='RIM_ONE_r3'
TARGET_DIGEST='8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf'

def digest(data):return hashlib.sha256(data).hexdigest()
def sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA256 required')
    return value

def ordinary(path):
    p=Path(path).absolute()
    if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('source symlink path')
    s=p.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('ordinary unlinked source file required')
    return p,s

def verified(path,expected,max_bytes,counts=None):
    sha(expected);p,before=ordinary(path)
    if before.st_size>max_bytes:raise ValueError('input byte cap')
    if counts is not None:counts.update(read_attempts=1)
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        with os.fdopen(fd,'rb') as f:
            first=os.fstat(f.fileno());data=f.read(max_bytes+1);last=os.fstat(f.fileno())
        if counts is not None:counts.update(raw_reads=1,raw_bytes_read=len(data))
        after=p.stat()
        def key(s):return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink
        if any(key(s)!=key(before) for s in (first,last,after)):raise ValueError('source changed while reading')
        if len(data)>max_bytes or digest(data)!=expected:raise ValueError('source file digest mismatch')
        if counts is not None:counts.update(verified_reads=1,verified_bytes=len(data))
        return data
    except BaseException:raise

def asset_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=SOURCE:raise ValueError('source-only relative path required')
    return Path(root).absolute()/p

def validate_manifest(manifest,target):
    if manifest.get('schema')!='R7_SOURCE_MANIFEST_V1' or manifest.get('source')!=SOURCE:raise ValueError('source manifest schema/domain')
    if manifest.get('target_registration_digest')!=TARGET_DIGEST or registration_digest(target)!=TARGET_DIGEST:raise ValueError('target metadata binding')
    grouping=manifest['grouping']
    if grouping.get('kind') not in ('PATIENT','EYE','CONTENT') or not grouping.get('evidence'):raise ValueError('documented grouping required')
    if grouping['kind']!='PATIENT' and grouping.get('dependency_risk')!='PATIENT_DEPENDENCE_UNKNOWN':raise ValueError('nonpatient dependency risk must be explicit')
    rows=manifest['records'];seen=set();images={};paths=set();by_group={}
    forbidden={r['image_sha256'] for r in target['target']}
    patients={r.get('patient_id') for r in target['target']} - {None,'UNKNOWN'}
    for row in rows:
        sid=row['sample_id'];g=row['group_id'];sha(row['image_sha256']);sha(row['mask_sha256'])
        if not isinstance(g,str) or not g or sid in seen or row['domain']!=SOURCE:raise ValueError('duplicate/malformed source record')
        if row['image_sha256'] in forbidden or row.get('patient_id') in patients:raise ValueError('source/target overlap')
        if grouping['kind']=='CONTENT' and g!=row['image_sha256']:raise ValueError('content group must use registered image content, not filename')
        if grouping['kind']=='PATIENT' and row.get('patient_id')!=g:raise ValueError('patient grouping evidence missing')
        if grouping['kind']=='EYE' and row.get('eye_id')!=g:raise ValueError('eye grouping evidence missing')
        if row['image_sha256'] in images and images[row['image_sha256']]!=g:raise ValueError('same image crosses groups')
        images[row['image_sha256']]=g;seen.add(sid);by_group.setdefault(g,[]).append(row)
        for key in ('image','mask'):
            relative=row[key+'_relative'];asset_path('/',relative)
            if (key,relative) in paths:raise ValueError('duplicate source asset path')
            paths.add((key,relative))
        if len(row['image_size'])!=2 or any(type(x)!=int or x<1 for x in row['image_size']):raise ValueError('image geometry')
    checkpoint=manifest['checkpoint'];sha(checkpoint['sha256'])
    if checkpoint['sha256']!=target['checkpoint']['sha256'] or checkpoint['bytes']!=target['checkpoint']['bytes']:raise ValueError('registered source checkpoint mismatch')
    if not checkpoint.get('provenance') or 'pretraining_exposure' not in checkpoint:raise ValueError('checkpoint provenance/exposure audit required')
    return by_group

def freeze(manifest,target,existing=None):
    groups=validate_manifest(manifest,target);folds=split(list(groups),existing)
    return dict(schema='R7_SOURCE_SPLIT_V1',manifest_payload_sha256=json_digest(manifest),folds=folds,representatives={g:min(rows,key=lambda r:r['sample_id'])['sample_id'] for g,rows in groups.items()},representative_rule='smallest registered sample_id within an evidence-backed group; not a new patient',grouping=copy.deepcopy(manifest['grouping']))

def audit(manifest,frozen,target):
    groups=validate_manifest(manifest,target)
    if frozen.get('schema')!='R7_SOURCE_SPLIT_V1' or frozen.get('manifest_payload_sha256')!=json_digest(manifest):raise ValueError('split not bound to manifest')
    expected=freeze(manifest,target,frozen['folds'])
    if frozen!=expected:raise ValueError('split/representatives/grouping changed')
    return dict(groups=len(groups),records=len(manifest['records']),fold_groups={k:len(v) for k,v in frozen['folds'].items()},group_overlap=0,registered_image_target_overlap=0,patient_independence=manifest['grouping']['kind']=='PATIENT',target_patient_overlap='NOT_IDENTIFIABLE' if manifest['grouping']['kind']!='PATIENT' else 'CHECKED_WHERE_REGISTERED',decoded_RGB=0,decoded_masks=0)

class Reader:
    """Only instantiate after scope/code/science/metadata/resource authorization."""
    def __init__(self,manifest,frozen,target,root,counts,max_file_bytes):
        audit(manifest,frozen,target)
        self.manifest=copy.deepcopy(manifest);self.frozen=copy.deepcopy(frozen);self.root=Path(root);self.counts=counts;self.max_file_bytes=max_file_bytes;self.identities={}
    def read(self,relative,expected):
        p=asset_path(self.root,relative);data=verified(p,expected,self.max_file_bytes,self.counts);self.identities[str(p)]=(expected,len(data));return data
    def decode(self,row,kind):
        from ..source_io import read_pixels,read_mask
        raw=self.read(row[kind+'_relative'],row[kind+'_sha256'])
        reader=read_pixels if kind=='image' else read_mask
        self.counts.update(**{kind+'_decode_attempts':1})
        result=reader(io.BytesIO(raw),'fundus',row['image_size'])
        self.counts.update(**{('RGB_decodes' if kind=='image' else 'mask_decodes'):1})
        return result
    def data(self):
        rows={r['sample_id']:r for r in self.manifest['records']};records=[]
        for fold,groups in self.frozen['folds'].items():
            for group in groups:
                row=rows[self.frozen['representatives'][group]]
                records.append(Record(group,fold,self.decode(row,'image'),self.decode(row,'mask')))
        return SourceData(records,self.frozen['folds'])
    def after_check(self):
        # Each used input is re-read read-only. A failure is kept separately from
        # the first training failure by the runner; it never triggers a retry.
        for p,(expected,size) in self.identities.items():verified(p,expected,size,self.counts)
