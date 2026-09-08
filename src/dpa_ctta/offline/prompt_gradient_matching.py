"""Prompt-space supervised gradient matching, not full-network paper reproduction."""
import torch


def cosine_objective(synthetic,real):
    a,b=synthetic.flatten(),real.detach().flatten()
    return 1-torch.dot(a,b)/((a.square().sum()+1e-24).sqrt()*(b.square().sum()+1e-24).sqrt())
