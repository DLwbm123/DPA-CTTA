"""Aggregate completed private scores/traces; no new inference or label reads."""
import csv,json,sys
from pathlib import Path
import numpy as np


def main(root):
    read=lambda p:json.loads(p.read_text())
    def write(name,records):
        with (root/'public'/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    assert read(root/'RUN_STATE.json')['status']=='COMPLETE'
    rows=sorted(read(root/'scorer/SCREEN_o0.json'),key=lambda r:r['image_sha256'])
    scores=np.load(root/'scorer/scores.private.npy');assert scores.shape==(88,2,4,2)
    ds=np.array([r['domain'] for r in rows]);cy=np.array([r['cycle'] for r in rows]);names=sorted(set(ds))
    records=[]
    for cycle in (-1,0,1):
        pick=np.ones(88,dtype=bool) if cycle==-1 else cy==cycle
        for d in names:
            for ch,label in enumerate(('OD','OC')):
                means=scores[pick&(ds==d),:,:,ch].mean((0,1))
                records.append(dict(cycle='all' if cycle==-1 else cycle,domain=d,channel=label,
                    native_Dice=float(means[0]),soft_Dice=float(means[1]),always_Dice=float(means[2]),dsr_Dice=float(means[3]),
                    dsr_native_pp=float(means[3]-means[0]),dsr_soft_pp=float(means[3]-means[1]),dsr_always_pp=float(means[3]-means[2])))
    write('DOMAIN_CHANNEL_MEANS.csv',records)
    gates=[];areas=[]
    for order in (0,1):
        panel=read(root/f'scorer/SCREEN_o{order}.json')
        for arm in ('NATIVE','SOFT','ALWAYS','DSR'):
            trace=[json.loads(x) for x in (root/f'private/{arm}_o{order}.jsonl').read_text().splitlines()]
            masks=np.unpackbits(np.frombuffer((root/f'predictions/{arm}_o{order}.bits').read_bytes(),dtype=np.uint8)).reshape(88,2,512,512)
            for cycle in (0,1):
                for d in names:
                    index=[i for i,r in enumerate(panel) if r['cycle']==cycle and r['domain']==d]
                    ts=[trace[i] for i in index];post=[t for t in ts if not t['warmup']]
                    gates.append(dict(arm=arm,order=order,cycle=cycle,domain=d,n_contents=len(index),
                        active_steps=sum(t['hard_active'] for t in ts),postwarm_steps=len(post) if arm!='NATIVE' else 0,
                        mean_discrepancy=float(np.mean([t['bernoulli_discrepancy'] for t in ts])),
                        mean_reference=float(np.mean([t['discrepancy_reference'] for t in ts])),
                        positive_fraction=float(np.mean([t['positive_fraction'] for t in ts])),
                        negative_fraction=float(np.mean([t['negative_fraction'] for t in ts]))))
                    for ch,label in enumerate(('OD','OC')):
                        areas.append(dict(arm=arm,order=order,cycle=cycle,domain=d,channel=label,
                            mean_foreground_pixels=float(masks[index,ch].sum((1,2)).mean())))
    write('GATE_BY_DOMAIN_CYCLE.csv',gates);write('PREDICTED_AREA.csv',areas)


if __name__=='__main__':main(Path(sys.argv[1]))
