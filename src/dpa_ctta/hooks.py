"""Exception-safe feature tracing and directional logits JVP utilities."""

from contextlib import contextmanager
import torch


@contextmanager
def capture_output(model, path):
    modules = dict(model.named_modules())
    if path not in modules:
        raise ValueError(f"module not found: {path}")
    captured = []
    handle = modules[path].register_forward_hook(lambda _m, _i, output: captured.append(output))
    try:
        yield captured
    finally:
        handle.remove()


def enumerate_4d_outputs(model, forward):
    """Run one caller-supplied forward and return executed 4-D Tensor outputs."""
    records, handles = [], []

    def make_hook(name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                records.append(
                    {
                        "order": len(records),
                        "path": name,
                        "shape": list(output.shape),
                        "output_type": "Tensor",
                    }
                )
        return hook

    try:
        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(make_hook(name)))
        forward()
    finally:
        for handle in handles:
            handle.remove()
    return records


def directional_logits_jvp(model, path, image, logits_fn):
    """Exact autograd JVP for a deterministic direction at one module output."""
    modules = dict(model.named_modules())
    if path not in modules:
        raise ValueError(f"module not found: {path}")
    alpha = [None]

    def inject(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError("injection output must be a 4-D Tensor")
        # The direction is fixed: the JVP variable is the scalar perturbation only.
        direction = torch.tanh(output.detach()) + 0.125
        return output + alpha[0] * direction

    handle = modules[path].register_forward_hook(inject)

    def forward(scale):
        alpha[0] = scale
        return logits_fn(model, image)

    try:
        _, jvp = torch.autograd.functional.jvp(
            forward,
            (torch.zeros((), dtype=image.dtype, device=image.device),),
            (torch.ones((), dtype=image.dtype, device=image.device),),
            strict=True,
        )
    finally:
        handle.remove()
    return jvp
