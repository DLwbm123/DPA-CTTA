"""Primitive-only RNG snapshots for trusted weights-only checkpoint loading."""
import random

import numpy as np
import torch


def capture():
    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return dict(torch_cpu=torch.random.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                numpy=(name, keys.tolist(), position, has_gauss, cached_gauss),
                python=random.getstate())


def restore(state):
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("R8 CUDA RNG unavailable during restore")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    random.setstate(state["python"])
