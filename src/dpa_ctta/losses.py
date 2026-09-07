"""Offline-only mathematical loss functions with no data loading side effects."""

import torch
import torch.nn.functional as F


def worst_class_segmentation_loss(logits, source_label, eps=1e-6):
    if logits.shape != source_label.shape or logits.ndim != 4:
        raise ValueError("logits and source_label must share [batch, class, height, width]")
    probability = logits.sigmoid()
    intersection = (probability * source_label).sum((0, 2, 3))
    denominator = probability.sum((0, 2, 3)) + source_label.sum((0, 2, 3))
    dice_loss = 1 - (2 * intersection + eps) / (denominator + eps)
    bce = F.binary_cross_entropy_with_logits(logits, source_label, reduction="none").mean((0, 2, 3))
    return (dice_loss + bce).max()


def state_matching_loss(predicted_state, oracle_state):
    if predicted_state.shape != oracle_state.shape:
        raise ValueError("state shapes must match")
    return F.mse_loss(predicted_state, oracle_state)


def trajectory_increment_loss(predicted_states, oracle_states):
    if predicted_states.shape != oracle_states.shape or predicted_states.ndim < 2:
        raise ValueError("trajectory tensors must share [time, ..., latent_dim]")
    if predicted_states.shape[0] < 2:
        raise ValueError("at least two trajectory states are required")
    return F.mse_loss(predicted_states[1:] - predicted_states[:-1], oracle_states[1:] - oracle_states[:-1])


def _class_statistics(feature, mask, eps):
    mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear", align_corners=False)
    mass = mask.sum((-2, -1)).clamp_min(eps)
    mean = torch.einsum("bchw,bkhw->bkc", feature, mask) / mass[:, :, None]
    second = torch.einsum("bchw,bkhw->bkc", feature.square(), mask) / mass[:, :, None]
    return mean, (second - mean.square()).clamp_min(0)


def classwise_dense_feature_loss(synthetic_features, reference_features, synthetic_mask, reference_mask=None, eps=1e-6):
    if len(synthetic_features) != len(reference_features) or not synthetic_features:
        raise ValueError("synthetic and reference feature lists must be nonempty and aligned")
    reference_mask = synthetic_mask if reference_mask is None else reference_mask
    losses = []
    for synthetic, reference in zip(synthetic_features, reference_features):
        if synthetic.shape[1] != reference.shape[1]:
            raise ValueError("feature channels must match")
        synthetic_stats = _class_statistics(synthetic, synthetic_mask, eps)
        reference_stats = _class_statistics(reference, reference_mask, eps)
        losses.extend(F.mse_loss(a, b) for a, b in zip(synthetic_stats, reference_stats))
    return torch.stack(losses).mean()


def distillation_loss(
    task_loss,
    predicted_state,
    oracle_state,
    predicted_trajectory,
    oracle_trajectory,
    synthetic_features,
    reference_features,
    synthetic_mask,
    reference_mask=None,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    return (
        task_loss
        + alpha * state_matching_loss(predicted_state, oracle_state)
        + beta * trajectory_increment_loss(predicted_trajectory, oracle_trajectory)
        + gamma
        * classwise_dense_feature_loss(
            synthetic_features, reference_features, synthetic_mask, reference_mask
        )
    )
