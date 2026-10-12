"""CPU-qualified shared-span configuration; no experiment queue or registration."""
from .method import Host, candidates as original_candidates


def candidates():
    c, w, lso, joint = original_candidates()
    return [c, w, lso, dict(joint, id='CW_LSO_JS_SHARED', joint_basis='shared')]
