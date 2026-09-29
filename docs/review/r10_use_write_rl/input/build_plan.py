"""Build the finite planning matrix. This file does not import or run the experiment."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
SEEDS=[20260924,20260925,20260926,20260927,20260928]
METHODS=['SUP_STATIC','SUP_SEQ','SUP_RET','GR_CUR','GR_SEQ','GR_RET','GR_RET_EMA']
SUP=METHODS[:3]; GR=METHODS[3:]
BASE=['N_SOURCE_EVAL','C0','B_CARRIER_FULL','B_CARRIER_RESET','B_CARRIER_STATIC']
EXT=['VPTTA_NATIVE','C_CTTA','G_CTTA']
source=[]
for s in SEEDS:
    source.append(dict(id=f'WARM_{s}',kind='warmup',seed=s,updates=2000,
                       needs=['ASSET_BINDING']))
for s in SEEDS[:2]:
    for m in METHODS:
        source.append(dict(id=f'FIT_{m}_{s}',kind='post_training',method=m,seed=s,
                           collection_rounds=4000,optimizer_epochs=2,
                           needs=[f'WARM_{s}','FREEZE_PROTOCOL']))
for s in SEEDS[2:]:
    for f in ['SUP','GR']:
        source.append(dict(id=f'FIT_SELECTED_{f}_{s}',kind='post_training',
                           method=f'SELECTED_{f}',seed=s,collection_rounds=4000,
                           optimizer_epochs=2,needs=[f'WARM_{s}','FAMILY_SELECTION']))

target=[]
def add(stage,arm,seed,order,**extra):
    tag=str(order)
    row=dict(id=f'{stage}__{arm}__{seed if seed is not None else "det"}__{tag}',
             stage=stage,arm=arm,seed=seed,order=order,
             arrivals=19510 if order=='LONG10' else 1951,
             principal_visits=16950 if order=='LONG10' else 1695,
             needs=['SOURCE_LOCK'],**extra)
    target.append(row)
for s in SEEDS[:2]:
    for m in METHODS:
        for o in [0,1]:add('DISCOVERY',m,s,o,checkpoint='source_selected')
for s in SEEDS[:2]:
    for f in ['SUP','GR']:
        for o in [2,3,4]:add('EXTENSION',f'SELECTED_{f}',s,o,checkpoint='source_selected')
for s in SEEDS[2:]:
    for f in ['SUP','GR']:
        for o in range(5):add('CONFIRMATION',f'SELECTED_{f}',s,o,checkpoint='source_selected')
main_ids=[r['id'] for r in target]
for s in SEEDS:
    for a in ['RESET_ALL','FORCE_WRITE','CONST_HALF']:
        for o in range(5):add('ABLATION',f'SELECTED_GR_{a}',s,o,
                              checkpoint='source_selected',diagnostic=a)
for b in BASE:
    for o in range(5):add('BASELINE',b,None,o)
for b in EXT:
    for s in range(20260907,20260912):
        for o in range(5):add('BASELINE',b,s,o)
for s in SEEDS:
    for o in range(5):add('BASELINE','WARM_STATIC',s,o)
for o in ['MIXED','LONG10']:
    for f in ['SUP','GR']:
        for s in SEEDS:add('STRESS',f'SELECTED_{f}',s,o,checkpoint='source_selected')
    for b in BASE:add('STRESS',b,None,o)
    for b in EXT:
        for s in range(20260907,20260912):add('STRESS',b,s,o)
    for s in SEEDS:add('STRESS','WARM_STATIC',s,o)
core=target.copy()
final=[]
for r in core:
    if r['id'] in main_ids:
        row=dict(r,id='FINAL4000__'+r['id'],stage='FINAL4000',checkpoint='round_4000',
                 alias_if_identical=r['id'])
        final.append(row)

counts=dict(source_jobs=len(source),warmups=sum(j['kind']=='warmup' for j in source),
            post_training_jobs=sum(j['kind']=='post_training' for j in source),
            main_target_slots=len(main_ids),core_target_slots=len(core),
            sensitivity_slots_max=len(final),all_target_slots_max=len(core)+len(final),
            core_arrivals=sum(x['arrivals'] for x in core),
            all_arrivals_max=sum(x['arrivals'] for x in core+final),
            core_principal_visits=sum(x['principal_visits'] for x in core),
            all_principal_visits_max=sum(x['principal_visits'] for x in core+final),
            warmup_optimizer_steps=5*2000,post_collection_rounds=20*4000,
            post_optimizer_steps=20*4000*2,
            post_initial_candidate_visits=20*4000*4*4)
spec=dict(
 schema='R10_USE_WRITE_RL_V1',status='PLAN_AND_REFERENCE_MATH_NOT_IMPLEMENTED_NOT_LAUNCHED',
 base_repository='DLwbm123/DPA-CTTA',base_code_sha='13bd6a8cdf9c30a0a5ed7fa464c67e72301ac8c6',
 final_runtime_sha=None,namespace='r10_use_write_rl',branch='experiment/r10-use-write-rl-v1',
 carrier=dict(source='SCREEN24_FROZEN_DEPLOYMENT',job='B_FULL_20260924',rank=64,
              amplitude=.3,observer='global',all_weights_frozen=True,
              selection='fixed identity; not chosen by target score',
              depends_on_R9_completion=False,gradient_scale='verified source-fit B64 scale; floor 1e-3'),
 policy=dict(input_dim=193,use_mlp=[193,128,64,9],write_mlp=[193,64,32,1],
             independent_networks=True,activation='SiLU',mu_bound=5.,
             raw_gaussian_std=.35,gain='sigmoid(a[0])',
             residual_coordinates=list(range(8)),residual_scale=.5,
             write='sigmoid(a[9])',initial_gain=.8,initial_write=.5,
             deployment='raw_action=mu; no sample, verifier, reference, reward or gradient'),
 memory=dict(fields={'m':64,'q':32,'h':1},initial='all zero',
             difference='h*d-q',m_update='(1-w)*m+w*u',q_update='(1-w)*q+w*d',
             h_update='(1-w)*h+w',audit_counter_not_policy_input=True),
 training=dict(policy_seeds=SEEDS,discovery_seeds=SEEDS[:2],confirmation_seeds=SEEDS[2:],
               methods=METHODS,supervised_family=SUP,rl_family=GR,warmup_steps=2000,
               warmup_batch_visits=8,warmup_lr_peak=3e-4,warmup_lr_final=3e-5,warmup_lr_warmup=100,
               post_rounds=4000,group_size=4,horizon=4,optimizer_epochs=2,
               optimizer='AdamW',post_lr_peak=3e-5,post_lr_final=3e-6,post_lr_warmup_rounds=100,
               betas=[.9,.999],eps=1e-8,weight_decay=1e-4,grad_clip_norm=1.,
               checkpoints=[500,1000,2000,4000],sigma_fixed=True,
               candidate_group='same prefix and exogenous source sequence',
               temporal_modes=['abrupt_16_16','gradual_32','recurrence_8_8_8_8','iid_style_32'],
               source_style_mode_hidden_from_actor=True),
 objective=dict(task='macro hard Dice for RL; macro soft Dice for post-training SUP',
                warmup='existing seg_loss',retention_lambda=.05,retention_alpha=20.,
                retention='exp(-20*mean(relu(D_anchor_probe-D_candidate_probe)))',
                retention_anchor='same sampling policy and pre-window memory; clone readouts, no commits',
                reference_policy='frozen per-seed warmup policy',ppo_old='exact collection snapshot',
                ppo_clip=.2,policy_kl_coefficient=.005,kl='analytic raw Gaussian forward KL; mean active dimensions',
                advantage_std_floor=.005,advantage_epsilon=1e-8,advantage_clip=5.,
                reward_ema_beta=.99,reward_ema_initial=.01,
                all_equal_group='zero advantage; no resampling; common KL still evaluated'),
 validation=dict(episodes=64,visits=32,per_mode=16,action='deterministic mean',
                 selection='.5*mean(mode_hard_Dice)+.5*min(mode_hard_Dice)',
                 checkpoint_tie='within 1e-8 -> earlier',
                 family_selection='mean of first-two-seed selected validation score',
                 family_tie_order={'SUP':SUP,'GR':GR},
                 sampled_mean_gap='16 source-cal episodes, 4 samples, selected checkpoint only; diagnostic, no selection'),
 d0=dict(prefix_contexts_per_seed=64,seeds=SEEDS[:2],candidate_use_actions=4,
         write_interventions=[0,1],future_horizon=4,probe_count=2,
         threshold_future_difference=.002,scientific_early_stop=False),
 target=dict(principal_orders=[0,1],secondary_orders=[2,3],recurrence_order=4,
             primary='domain equal, OD/OC equal, principal orders equal, policy seeds equal',
             score_release='sealed_internal_per_trajectory__release_at_round_end',
             max_temporary_probability_bytes=19510*2*512*512*4,
             endpoint_sensitivity='all 70 main source slots at round 4000; reuse identical artifacts',
             clinical_claims=False),
 resources=dict(gpu_worker_hours=256,disk_bytes=64*1024**3,model_forwards=12000000,
                backward_calls=2000000,optimizer_steps=1000000,vjp_calls=0,max_gpu_workers=1,
                profile_subcaps=dict(gpu_worker_hours=8,model_forwards=250000,
                                     backward_calls=20000,optimizer_steps=20000),
                recovery=dict(max_jobs=3,max_extra_attempts_per_job=1,
                              reserve='per-resource top-three additional full attempts'),
                excludes_R9_budget=True,do_not_preempt_R9=True),
 counts=counts,source_jobs=source,target_core_slots=core,target_final4000_slots_max=final)
(ROOT/'R10_SPEC_AND_MATRIX.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(counts,ensure_ascii=False,indent=2))
