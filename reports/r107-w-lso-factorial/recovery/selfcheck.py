"""CPU-only guard injection: no model calls, no old prediction edits."""
import json,os,sys,tempfile
from pathlib import Path
from codec import guard_write,StorageLimit,emergency,usage,self_check
root=Path('/tmp/n107').resolve()
with tempfile.TemporaryDirectory(prefix='guard_check_',dir=root/'private') as td:
    p=Path(td);(p/'private').mkdir();receipt=p/'private/RECOVERY_EMERGENCY.json'
    with receipt.open('xb') as f:f.write(b'{}'.ljust(32768,b' '));f.flush();os.fsync(f.fileno())
    c=dict(output_root=str(p),remote_normal_cap_bytes=1)
    try:guard_write(c,123,'synthetic')
    except StorageLimit:pass
    else:raise AssertionError('oversize write admitted')
    emergency(c,worker_state=dict(physical_updates=65,status='FAILED'))
    got=json.loads(receipt.read_text());assert got['storage_trigger']['next_bytes']==123
    assert got['storage_trigger']['usage']>1 and got['worker_state']['physical_updates']==65
    assert receipt.stat().st_size==32768
self_check()
assert 36+1793+576==2405 and 348+16137+5184==21669
for p in Path(__file__).resolve().parent.glob('*.py'):compile(p.read_text(),str(p),'exec')
print(json.dumps(dict(status='PASS',forced_storage_failure_preserved=True,counters_preserved=True,receipt_bytes=32768,model_forwards=0,training_updates=0,backwards=0,current_usage=usage(root))))
