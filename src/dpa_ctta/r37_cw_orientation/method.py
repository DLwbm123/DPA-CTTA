import copy
from ..r36_incremental_modules.method import Host, candidates as previous_candidates


def candidates():
    c, w = copy.deepcopy(previous_candidates()[:2])
    cw = dict(w, id='CW', host='C', final_lr_multiplier=1.)
    lso = dict(cw, id='CW_LSO', module='orientation', module_strength=.1)
    return [c, w, cw, lso]
