"""Frozen joint-sensitivity efficacy matrix on the existing guarded runtime."""
from pathlib import Path
from ..joint_sensitivity.method import Host
from ..r37_cw_orientation import run as reporting


def report(c, state):
    reporting.report(c, state)
    path = Path(c['output_root'])/'public/REPORT.md'
    text = path.read_text().replace(
        'RL, extra views, output fusion',
        'RL or output fusion; only the registered two acquisition probes are added')
    path.write_text(text)


def main():
    reporting.retained.ID = 'R45_JOINT_SENSITIVITY'
    reporting.retained.Host = Host
    reporting.retained.report = report
    reporting.retained.public = reporting.public
    reporting.retained.main()


if __name__ == '__main__':
    main()
