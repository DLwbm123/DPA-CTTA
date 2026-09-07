"""Source-supervised oracle-state optimization for the later M0 gate."""

import torch

from ..losses import worst_class_segmentation_loss


def _logits(output):
    return output[0] if isinstance(output, tuple) else output


def optimize_oracle_state(model, image, source_label, initial_state=None, steps=20, learning_rate=0.1):
    if source_label is None or image.shape[0] != source_label.shape[0]:
        raise ValueError("an explicit aligned source_label is required")
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    latent_dim = model.latent_dim
    start = image.new_zeros(image.shape[0], latent_dim) if initial_state is None else initial_state
    state = torch.nn.Parameter(start.detach().clone())
    optimizer = torch.optim.SGD([state], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = worst_class_segmentation_loss(_logits(model(image, state)), source_label)
        loss.backward()
        optimizer.step()
    return state.detach()
