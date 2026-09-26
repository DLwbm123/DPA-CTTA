"""Immutable planning inputs and finite dependencies; no runtime side effects."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT/'docs/review/r9_current_first/input/R9_SPEC_AND_MATRIX.json'
SPEC_BYTES = SPEC_PATH.read_bytes()
SPEC = json.loads(SPEC_BYTES)
SPEC_SHA = hashlib.sha256(SPEC_BYTES).hexdigest()
CHECKPOINTS = (4000, 8000, 12000, 16000)
RECIPES = ('LEGACY', 'SELF', 'SELF_TASK')
CAPS = dict(gpu_seconds=512*3600, disk_bytes=64*1024**3, model_forwards=32000000,
            backward_calls=3000000, optimizer_steps=3000000, vjp_calls=4096)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def parse_recipe(recipe, selected=None):
    if recipe in ('MLP_LEGACY','MLP_SELF'):
        return 'MLP',recipe[4:],'STATIC'
    route,body=recipe[0],recipe[2:]
    if route not in ('A','B'):raise ValueError('R9 route')
    mode=body.rsplit('_',1)[-1]; kind=body[:-(len(mode)+1)]
    if kind=='SELECTED':kind=selected[route] if selected else None
    if kind not in RECIPES or mode not in ('FULL','STATIC'):raise ValueError('R9 recipe not resolved')
    return route,kind,mode


def graph():
    nodes=[]
    first=[j['id'] for j in SPEC['source_tasks'] if j['seed'] in (20260924,20260925)]
    for j in SPEC['source_tasks']:
        nodes.append(dict(id=j['id'],kind='source',resource='gpu',job=j,
                          needs=['assets'] if j['id'] in first else ['assets','recipe_selection']))
    nodes += [dict(id='assets',kind='bind_assets',resource='cpu',needs=[]),
              dict(id='recipe_selection',kind='select_recipes',resource='cpu',needs=first),
              dict(id='gradient_selection',kind='select_lr',resource='gpu',needs=[j['id'] for j in SPEC['source_tasks']]),
              dict(id='source_lock',kind='lock',resource='cpu',needs=['gradient_selection','recipe_selection']+[j['id'] for j in SPEC['source_tasks']])]
    for j in SPEC['target_core_slots']+SPEC['target_final16k_slots_max']:
        nodes += [dict(id=j['id']+'__online',kind='online',resource='gpu',job=j,needs=['source_lock']),
                  dict(id=j['id']+'__score',kind='score',resource='cpu',job=j,needs=['source_lock',j['id']+'__online'])]
    ids={n['id'] for n in nodes}
    if len(ids)!=len(nodes) or any(set(n['needs'])-ids for n in nodes):raise ValueError('invalid R9 DAG')
    return dict(schema='R9_DAG_V1',spec_sha256=SPEC_SHA,execution_authorized=False,
                counts=SPEC['counts'],nodes=nodes)


def disabled_config():
    return dict(schema='R9_LAUNCH_V1',execution_authorized=False,spec_sha256=SPEC_SHA,
                code_sha=None,source_inventory=None,bindings=None,gpu_assignments=None,
                lr_source_policy=None,score_release_policy='sealed_internal_score_release_at_end',
                output_root=None,profile=None,admission=None,authorization=None,
                caps=CAPS,estimated_time_is_soft=True,max_gpu_workers=3,cpu_score_workers=1)
