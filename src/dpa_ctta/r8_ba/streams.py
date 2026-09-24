"""Bind R8 standard and stress arrival sequences without reading pixels."""
import hashlib
import json

from ..r1.plan import registration_digest
from ..r3.plan import stream
from ..r7_shared.plan import dry_run

MIXED_KEY = "R8_MIXED_V1|20260924|"


def _sequence_sha(rows):
    return hashlib.sha256(json.dumps([row["group_id"] for row in rows],
                                     separators=(",", ":")).encode()).hexdigest()


def stress(rows, kind):
    if (len(rows) != 1951 or len({row["group_id"] for row in rows}) != 1951 or
            sum(row["subset"] == "remaining_dev" for row in rows) != 1695):
        raise ValueError("R8 stress source stream identity")
    if kind == "MIXED":
        return sorted(rows, key=lambda row: (hashlib.sha256(
            (MIXED_KEY + row["group_id"]).encode()).digest(), row["group_id"]))
    if kind == "LONG10":
        return rows * 10
    raise ValueError("R8 stress stream")


def bind(registration):
    """The existing R7 metadata verifier fixes the five 1951-item orders."""
    audit = dry_run(registration)
    standard = [stream(registration, order) for order in range(5)]
    mixed, long10 = stress(standard[0], "MIXED"), stress(standard[0], "LONG10")
    if ([row["sequence_sha256"] for row in audit["stream"]] !=
            [_sequence_sha(rows) for rows in standard]):
        raise ValueError("R8 standard order digest mismatch")
    return dict(schema="R8_TARGET_STREAM_BINDING_V1",
                registration_digest=registration_digest(registration),
                standard=[dict(order=i, arrivals=1951, scored=1695,
                               sequence_sha256=_sequence_sha(rows)) for i, rows in enumerate(standard)],
                stress=dict(MIXED=dict(arrivals=1951, scored=1695, seed=20260924,
                                       rule=MIXED_KEY, sequence_sha256=_sequence_sha(mixed)),
                            LONG10=dict(arrivals=19510, scored=16950,
                                        rule="order0 repeated 10 times without reset",
                                        sequence_sha256=_sequence_sha(long10))))
