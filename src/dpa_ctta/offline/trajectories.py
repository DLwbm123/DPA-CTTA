"""Pure trajectory utilities for later source-only experiments."""


def trajectory_increments(states):
    if states.ndim < 2 or states.shape[0] < 2:
        raise ValueError("states must contain at least two time steps")
    return states[1:] - states[:-1]
