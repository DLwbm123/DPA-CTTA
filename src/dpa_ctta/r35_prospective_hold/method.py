"""Pause future learning while advancing consumed arrivals and paired RNG."""
from ..r30_state_history.method import readonly, set_native_rng

HORIZONS = (64, 256)
PIVOTS = tuple(range(64, 1952, 128))
PAUSE = 256


def hold_step(h, x, rng_pair):
    set_native_rng(h, rng_pair[0])
    z = readonly(h, x)
    h.visits += 1
    set_native_rng(h, rng_pair[1])
    return z
