"""Fixed postprocessing transfer audit on sealed R106 comparisons, zero model calls."""
import csv,gzip,io,json,os,runpy,time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from codec import put,unpack
ARMS=('E_C2','E_VV','C_C2','C_VV');STAGES=('raw','largest8','fill4','intersect')
SEEDS=(20260907,17011,29009)

def stages(mask):
    largest=np.zeros_like(mask)
    for h in (0,1):
        labels,n=ndi.label(mask[h],np.ones((3,3),bool))
        if n:
            counts=np.bincount(labels.ravel());counts[0]=0;largest[h]=labels==counts.argmax()
    filled=np.stack([ndi.binary_fill_holes(v,structure=ndi.generate_binary_structure(2,1)) for v in largest]);final=filled.copy();final[1]&=final[0]
    return (mask,largest,filled,final)

def geometry(m):
    hs=[]
    for h in (0,1):
        rr=np.flatnonzero(m[h].any(1));hs.append(int(rr[-1]-rr[0]+1) if len(rr) else 0)
    return np.array((hs[1]/hs[0] if hs[0] else np.nan,*hs),float)

def self_check():
    m=np.zeros((2,512,512),bool);m[0,10:30,10:30]=1;m[0,15:18,15:18]=0;m[0,90,90]=1;m[1,8:20,12:16]=1
    s=stages(m);assert not s[1][0,90,90] and s[2][0,16,16] and not (s[3][1]&~s[3][0]).any()
    assert geometry(s[2])[1]==geometry(s[1])[1]

def run(c,write):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';self_check();start=time.time();root=Path(c['output_root']);oldroot=Path('/tmp/n106')
    assert json.loads((oldroot/'public/CONTROLLER_EXIT.json').read_text())['status']=='COMPLETE'
    assert json.loads((oldroot/'public/SEALED.json').read_text())['labels_read_before_seal']==0
    module=runpy.run_path('/tmp/n90/m0.py');post=module['post'];from dpa_ctta.r7_target_screen.runner import TargetReader
    panels=[json.loads((root/'scorer'/f'screen_o{o}.json').read_text()) for o in (0,1)];idx={r['image_sha256']:i for i,r in enumerate(panels[0])};ds=sorted({r['domain'] for r in panels[0]});domains=np.array([ds.index(r['domain']) for r in panels[0]])
    with np.load(oldroot/'scorer/values.private.npz',allow_pickle=False) as f:
        reference=f['values'];assert np.array_equal(domains,f['domains'])
    reader=TargetReader(c['target_root'],256*1024**2,'mask');target=[]
    for row in panels[0]:target.append(geometry(reader.read(row).numpy()[0].astype(bool))[0])
    target=np.array(target);data=np.full((192,6,4,4,4),np.nan)
    for ai,a in enumerate(ARMS):
        for seed in SEEDS:
            for o in (0,1):
                if time.time()>c['deadline']-120:raise TimeoutError('post audit deadline')
                j=SEEDS.index(seed)*2+o
                if ai<2:stream=gzip.open(oldroot/'jobs'/f'{a}_s{seed}_o{o}'/'K.bits.gz','rb')
                else:stream=open(c['reference_predictions'][f'{"C2" if ai==2 else "VV"}_{seed}_{o}'],'rb')
                with stream:
                    for row in panels[o]:
                        i=idx[row['image_sha256']]
                        if ai==3:assert len(stream.read(65536))==65536  # retained PRE, not a new readout
                        m=unpack(stream.read(65536));ss=stages(m);assert np.array_equal(ss[-1],post(m))
                        for si,x in enumerate(ss):
                            g=geometry(x);data[i,j,ai,si]=[abs(g[0]-target[i]),*g]
                    assert not stream.read(1)
                write('public/POST_AUDIT_PROGRESS.json',dict(arm=a,seed=seed,order=o,completed_stream=ai*6+j+1))
    reader.after_check()
    for si,vi in ((0,0),(3,1)):
        assert np.allclose(data[:,:,:,si,0],reference[:,:,:,vi,0,6],atol=1e-12,rtol=0,equal_nan=True),'retained raw/post metric mismatch'
    def mean(x):return float(x.mean()) if np.isfinite(x).all() else None
    def macro(x):return np.mean([x[domains==d].mean(axis=0) for d in range(4)],axis=0)
    rows=[]
    for ai,a in enumerate(ARMS):
        for si,stage in enumerate(STAGES):
            for d in range(4):
                for j in range(6):
                    x=data[domains==d,j,ai,si]
                    rows.append(dict(arm=a,stage=stage,domain=f'D{d}',seed=SEEDS[j//2],order=j%2,n=len(x),n_NA=int((~np.isfinite(x[:,0])).sum()),vcdr_abs_error=mean(x[:,0]),pred_vcdr=mean(x[:,1]),OD_height=mean(x[:,2]),OC_height=mean(x[:,3]),empty_OD=int((x[:,2]==0).sum()),empty_OC=int((x[:,3]==0).sum())))
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);put(c,root/'public/POST_AUDIT_CELLS.csv',buf.getvalue().encode())
    totals=[];history=[];transfers=[]
    for ai,a in enumerate(ARMS):
        for si,stage in enumerate(STAGES):totals.append(dict(arm=a,stage=stage,error=mean(macro(data[:,:,ai,si,0])),n_NA=int((~np.isfinite(data[:,:,ai,si,0])).sum())))
    for name,ci,ei in (('C2',2,0),('VV',3,1)):
        diff=data[:,:,ci,:,0]-data[:,:,ei,:,0]
        for si,stage in enumerate(STAGES):history.append(dict(method=name,stage=stage,continuous_minus_episodic=mean(macro(diff[:,:,si]))))
        for si in (1,2,3):
            x=diff[:,:,si]-diff[:,:,si-1];per=x.mean(1);weights=np.array([1/(4*int((domains==d).sum())) for d in domains]);contribution=per*weights;den=float(np.abs(contribution).sum());top=np.argsort(np.abs(contribution))[-5:]
            transfers.append(dict(method=name,operation=STAGES[si],history_difference_shift=mean(macro(x)),domain_shifts=[mean(x[domains==d]) for d in range(4)],content_mean=mean(per),content_median=float(np.median(per)),positive_contents=int((per>0).sum()),negative_contents=int((per<0).sum()),unchanged_contents=int((per==0).sum()),top5_absolute_contribution_fraction=float(np.abs(contribution[top]).sum()/den) if den else 0.,top5_signed_contribution=float(contribution[top].sum()),interpretation='descriptive fixed operator transfer, no causal source attribution or selection'))
    result=dict(status='COMPLETE',model_calls=0,streams=24,contents=192,mask_reads=4608,retained_raw_post_metric_match=True,stage_order=list(STAGES),totals=totals,history=history,transfers=transfers,CPU_body_seconds=time.time()-start,changed_postprocessing=False)
    write('public/POST_AUDIT.json',result);return dict(status='COMPLETE',model_calls=0,streams=24)
