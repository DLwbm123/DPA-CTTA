"""Read registered metadata only; do not open image/mask pixels or infer patient IDs."""
import csv,json,time
from pathlib import Path
root=Path('/remote-home/wangbomin/root-migrated/CTTA/data');panels=Path('/tmp/n102/private')
entries=[];targets=set()
for p in sorted(root.glob('*/*.csv')):
    with p.open() as f:
        d=csv.DictReader(f);rows=list(d);fields=d.fieldnames
    entries.append(dict(task=p.parent.name,manifest=p.name,rows=len(rows),fields=fields))
    if p.parent.name=='Fundus' and not p.name.startswith('RIM_ONE_r3'):
        for row in rows:
            q=Path(row['image']);q=q if q.is_absolute() else p.parent/q
            targets.add(str(q.resolve()))
panel=json.loads((panels/'screen_o0.json').read_text())
ids={str(Path(v['image_path']).resolve()) for v in panel}
assert len(targets)==1951 and len(ids)==192 and ids<=targets
result=dict(checked_at=time.time(),scope='existing mounted Fundus/Polyp CSV metadata and registered 192-content panel; no global-server inventory or pixel reads',manifests=entries,fundus_target_manifest_unique_paths=len(targets),registered_panel_unique_paths=len(ids),panel_paths_subset=True,duplicate_detection='path-level only; no new hashes, no claim of content or patient independence',patient_eye_keys_present=False,source_membership_record_in_inspected_model_tree=False,historical_exposure_evidence='reports/r85-r86-loss-aggregation/DATA_AUDIT.json; reports/r102-c1-c2-falsification/PROTOCOL.md',target_exposure='all 1951 historically exposed per retained audit; 192 is a subset',patient_eye_linkage='UNKNOWN',source_pretraining_overlap='UNKNOWN',independent_confirmation='NOT_ADMITTED',polyp='present but exposure/identity/compatible study contract not established; not admitted or used',development_mechanism='192 registered contents permitted by current explicit request',new_downloads=0,image_reads=0,label_pixel_reads=0)
print(json.dumps(result,indent=2))
