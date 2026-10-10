from ..r39_conflict_trust.method import Host
from ..r37_cw_orientation import run as reporting


def main():
    reporting.retained.ID = 'R40_PROFILED_VRAM'
    reporting.retained.Host = Host
    reporting.retained.report = reporting.report
    reporting.retained.public = reporting.public
    reporting.retained.main()


if __name__ == '__main__': main()
