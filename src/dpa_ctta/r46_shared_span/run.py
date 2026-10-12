"""One frozen five-condition shared-span matrix on the existing runtime."""
import csv
import json
import re
from pathlib import Path
from .method import Host
from ..r45_joint_sensitivity.run import report as previous_report
from ..r37_cw_orientation import run as reporting


def stage_success(state, pairs):
    required = {'C': .3, 'W': .3, 'CW_LSO': .1, 'CW_LSO_JS': .1}
    return (state['status'] == 'COMPLETE' and len(pairs) == 4
            and {p['baseline'] for p in pairs} == set(required)
            and all(reporting.passes(p, required[p['baseline']]) for p in pairs))


def report(c, state):
    previous_report(c, state)
    root = Path(c['output_root']); out = root/'public'
    jobs = reporting.read(root/'FINAL_JOBS.json') if (root/'FINAL_JOBS.json').exists() else []
    done = [j for j in jobs if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
    rows = reporting.retained.scoring.load_rows(root, 'FINAL', done, final=True)
    pairs, intervals, negatives = [], [], []
    for role in ('SEARCH', 'SEALED_REVIEW'):
        a, b, d = reporting.bootstrap_pair(rows, c['primary'], 'CW_LSO_JS', c['seeds'], role=role)
        pairs.append(a); intervals.extend(b); negatives.extend(d)
    for name, extra in [('PAIRED_SUMMARY.csv', pairs), ('CONTENT_BOOTSTRAP_CI.csv', intervals), ('ALL_NEGATIVE_CELLS.csv', negatives)]:
        with (out/name).open() as f: existing = list(csv.DictReader(f))
        reporting.csvout(out/name, existing+extra)
    with (out/'PAIRED_SUMMARY.csv').open() as f: paired = list(csv.DictReader(f))
    tests = [p for p in paired if p['candidate'] == c['primary'] and p['role'] == 'SEARCH']
    success = stage_success(state, tests)
    reporting.save(out/'FINAL_DECISION.json', dict(primary=c['primary'], strong_development_signal=success,
                   comparisons=reporting.public(tests), review_descriptive_only=True,
                   independent_generalization=False, automatic_retuning=False))
    path = out/'REPORT.md'
    text = path.read_text().replace('Exactly four frozen conditions', 'Exactly five frozen conditions')
    text = re.sub(r'Frozen primary passed: (True|False)', f'Frozen primary passed: {success}', text)
    path.write_text(text)


def main():
    reporting.retained.ID = 'R46_SHARED_SPAN'
    reporting.retained.Host = Host
    reporting.retained.report = report
    reporting.retained.public = reporting.public
    reporting.retained.main()


if __name__ == '__main__': main()
