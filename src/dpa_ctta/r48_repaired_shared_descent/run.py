"""Reuse the five-arm guarded runtime with a fresh R48 identity."""
from ..r47_shared_descent.run import Host, report, stage_success, reporting


def main():
    reporting.retained.ID = 'R48_REPAIRED_SHARED_DESCENT'
    reporting.retained.Host = Host
    reporting.retained.report = report
    reporting.retained.public = reporting.public
    reporting.retained.main()


if __name__ == '__main__':
    main()
