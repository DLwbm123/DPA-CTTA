"""Reuse the frozen input guard, qualification, retirement and scoring pipeline."""
from ..r63_rca import run as base
from .score import features, candidate_score


def worker(config):
    return base.worker(config, features=features, candidate_score=candidate_score)


if __name__ == '__main__':
    raise SystemExit(base.main(worker=worker))
