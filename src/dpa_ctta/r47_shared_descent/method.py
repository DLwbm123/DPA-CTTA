from ..joint_sensitivity.descent import Host as QualifiedHost, candidates as qualified_candidates
from ..joint_sensitivity.profile import check_joint_update


def candidates():
    return qualified_candidates()


class Host(QualifiedHost):
    def step(self, pixels):
        logits, detail = super().step(pixels)
        if self.strength and self.basis == 'shared':
            check_joint_update(self.config, detail['diagnostics']['joint'])
        return logits, detail
