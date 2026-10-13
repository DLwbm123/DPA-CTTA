"""Retain the guarded five-condition runtime; no automatic follow-up."""
import csv
from pathlib import Path
from .method import Host
from ..r46_shared_span.run import report as shared_report, stage_success as shared_success
from ..r37_cw_orientation import run as reporting


def stage_success(state, pairs):
    return shared_success(state, pairs, 'CW_LSO_JS_SHARED')


def report(c, state):
    shared_report(c, state)
    root = Path(c['output_root']); out = root/'public'
    jobs = reporting.read(root/'FINAL_JOBS.json') if (root/'FINAL_JOBS.json').exists() else []
    done = [j for j in jobs if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
    rows = reporting.retained.scoring.load_rows(root, 'FINAL', done, final=True)
    pair, intervals, negatives = reporting.bootstrap_pair(rows, c['primary'], 'CW_LSO', c['seeds'], role='SEALED_REVIEW')
    for name, extra in [('PAIRED_SUMMARY.csv', [pair]), ('CONTENT_BOOTSTRAP_CI.csv', intervals), ('ALL_NEGATIVE_CELLS.csv', negatives)]:
        with (out/name).open() as f: existing = list(csv.DictReader(f))
        reporting.csvout(out/name, existing+extra)


def main():
    reporting.retained.ID = 'R47_SHARED_DESCENT'
    reporting.retained.Host = Host
    reporting.retained.report = report
    reporting.retained.public = reporting.public
    reporting.retained.main()


if __name__ == '__main__': main()
