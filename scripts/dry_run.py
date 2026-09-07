#!/usr/bin/env python3
"""Emit a deterministic synthetic scientific payload; performs no training."""

import argparse
import json
from pathlib import Path

from dpa_ctta.audit import scientific_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    text = json.dumps(scientific_payload(), indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
