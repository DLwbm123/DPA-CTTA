#!/usr/bin/env python3
"""M0 command boundary. Real-data execution remains closed pending review."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data-root")
    parser.add_argument("--authorize-real-data-training", action="store_true")
    args = parser.parse_args()
    if args.real_data_root and not args.authorize_real_data_training:
        parser.error("real data requires --authorize-real-data-training")
    if args.real_data_root:
        raise SystemExit("BLOCKED_PENDING_CODE_REVIEW_PASS")
    print(
        json.dumps(
            {
                "mode": "dry-run",
                "training_executed": False,
                "real_data_accessed": False,
                "next_gate": "CODE_REVIEW_PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
