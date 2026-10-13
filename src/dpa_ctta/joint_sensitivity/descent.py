"""Opt-in CPU configuration; no formal experiment registration or launch."""
from .method import Host
from .shared_span import candidates as shared_candidates


def candidates():
    controls = shared_candidates()
    return controls + [dict(controls[-1], id='CW_LSO_JS_SHARED_DESCENT', joint_descent_constraint=True)]
