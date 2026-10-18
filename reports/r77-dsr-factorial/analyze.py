"""Summarize anonymous saved tables only; no images, labels or model execution."""
import csv,json,statistics,time
from collections import defaultdict
from pathlib import Path

root=Path(__file__).parent
start=time.perf_counter()
def read(name):return list(csv.DictReader((root/name).open()))
def write(name,rows):
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=read('DOMAIN_CHANNEL_ERRORS.csv');cells=read('ALL_CELLS.csv');mechanism=read('MECHANISM.csv')
index={(x['arm'],x['domain'],x['channel']):x for x in rows}
arms=('NATIVE_BN','NATIVE_DEC','DSR_BN','DSR_DEC')
summary=[];deltas=[];worst=[]
for arm in arms:
    rr=[x for x in rows if x['arm']==arm]
    od=statistics.mean(float(x['Dice_percent']) for x in rr if x['channel']=='OD')
    oc=statistics.mean(float(x['Dice_percent']) for x in rr if x['channel']=='OC')
    summary.append(dict(arm=arm,OD_Dice_percent=od,OC_Dice_percent=oc,macro_Dice_percent=(od+oc)/2))
    for x in rr:
        ref=index['NATIVE_BN',x['domain'],x['channel']]
        deltas.append(dict(arm=arm,domain=x['domain'],channel=x['channel'],
            delta_vs_NATIVE_BN_pp=float(x['Dice_percent'])-float(ref['Dice_percent']),
            Dice_percent=float(x['Dice_percent']),foreground_percent=100*float(x['foreground_fraction']),
            false_positive_pixels_percent_of_image=100*float(x['false_positive_pixel_fraction']),
            false_negative_pixels_percent_of_image=100*float(x['false_negative_pixel_fraction'])))
for name in dict.fromkeys(x['contrast'] for x in cells):
    rr=[x for x in cells if x['contrast']==name]
    negative=sum(float(x['delta_pp'])<0 for x in rr);w=min(rr,key=lambda x:float(x['delta_pp']))
    buckets=defaultdict(list)
    for x in rr:buckets[x['domain'],x['order'],x['channel']].append(float(x['delta_pp']))
    avg,key=min((statistics.mean(v),k) for k,v in buckets.items())
    worst.append(dict(contrast=name,negative_cells=negative,total_cells=len(rr),
        worst_seedmean_pp=avg,seedmean_domain=key[0],seedmean_order=int(key[1]),seedmean_channel=key[2],
        worst_individual_pp=float(w['delta_pp']),individual_domain=w['domain'],individual_seed=int(w['seed']),
        individual_order=int(w['order']),individual_channel=w['channel']))
mechan=[]
for arm in arms:
    mm=[x for x in mechanism if x['arm']==arm]
    count=sum(int(x['hard_activations']) for x in mm)
    mechan.append(dict(arm=arm,hard_activations=count,formal_arrivals=1152,eligible_after_warmup=1032,
        hard_fraction_of_eligible=count/1032 if arm.startswith('DSR') else None,
        mean_BN_gradient=statistics.mean(float(x['mean_BN_gradient']) for x in mm),
        mean_decoder_gradient=statistics.mean(float(x['mean_decoder_gradient']) for x in mm),
        measured_step_seconds=sum(float(x['seconds']) for x in mm)))
write('ARM_CHANNEL_SCORES.csv',summary);write('DOMAIN_CHANNEL_DELTAS.csv',deltas)
write('WORST_CELLS.csv',worst);write('MECHANISM_SUMMARY.csv',mechan)
(root/'ANALYSIS_RECEIPT.json').write_text(json.dumps(dict(input='anonymous saved aggregate tables',
    new_model_forwards=0,new_backwards=0,new_target_image_reads=0,new_label_reads=0,
    local_summary_body_seconds=time.perf_counter()-start,
    time_scope='summary body only; interpreter/import/SSH/report writing not separately metered'),indent=2))
print(json.dumps(dict(scores=summary,mechanism=mechan),indent=2))
