from ..joint_sensitivity.shared_span import Host, candidates as shared_candidates
from ..joint_sensitivity.method import candidates as original_candidates


def candidates():
    return original_candidates()+[shared_candidates()[-1]]
