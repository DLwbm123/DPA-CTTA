"""Source-val-only low-dimensional reachability probe; no target data."""
import torch

from ..r7_shared.numerics import COUNTS, finite, project, seg_loss
from ..r7_shared.source import simulate
from .protocol import CAPACITY_OPTIMIZER, PROTOCOL_SHA256
from .schedule import anchors


SPACES = {"A16": 16, "A32": 32, "B32": 32, "B64": 64}
ANCHORS = tuple(range(1, 33)) + tuple(range(64, 96))


def capacity_one(segmenter, data, val_oracles, basis, space, anchor):
    """Compare four interventions on one fixed source-val support/query split."""
    val_oracles.validate(data)
    if (val_oracles.fold != "val" or val_oracles.amplitude != segmenter.amplitude or
            space not in SPACES or anchor not in ANCHORS or
            basis.shape != (1024, SPACES[space]) or
            torch.linalg.matrix_rank(basis.double()) != SPACES[space]):
        raise ValueError("R8 capacity source/amplitude/space/anchor")
    support_names = val_oracles.support_pairs[anchor]
    query_names = val_oracles.query_pairs[anchor]
    style = anchors("val")[anchor]
    support = [(simulate(data.get(name, "val").image, style,
                         f"R8_ORACLE_SUPPORT|val|{anchor}|{name}"), data.get(name, "val").label)
               for name in support_names]
    query = [(simulate(data.get(name, "val").image, style,
                       f"R8_ORACLE_QUERY|val|{anchor}|{name}"), data.get(name, "val").label)
             for name in query_names]
    ambient = val_oracles.values[:, anchor].detach().float()
    projected, projection_audit = project(basis, ambient)
    basis = basis.detach().double()
    z = torch.zeros(SPACES[space], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=CAPACITY_OPTIMIZER["lr"],
                                 betas=tuple(CAPACITY_OPTIMIZER["betas"]),
                                 eps=CAPACITY_OPTIMIZER["eps"],
                                 weight_decay=CAPACITY_OPTIMIZER["weight_decay"])
    for _ in range(CAPACITY_OPTIMIZER["steps"]):
        optimizer.zero_grad(set_to_none=True)
        v = (basis @ z).float()
        loss = sum(seg_loss(segmenter(image, v), label) for image, label in support) / 2
        loss = loss + 1e-3 * v.square().mean()
        finite(loss)
        loss.backward()
        COUNTS["capacity_backward_calls"] += 1
        finite(z.grad)
        optimizer.step()
        COUNTS["capacity_Adam"] += 1
        finite(z)

    @torch.no_grad()
    def query_dice(v):
        values = []
        for image, label in query:
            probability = segmenter(image, v).sigmoid()
            channel = (2 * (probability * label).sum((0, 2, 3)) + 1e-6) / (
                probability.sum((0, 2, 3)) + label.sum((0, 2, 3)) + 1e-6)
            values.append(channel.mean().item())
        return sum(values) / len(values)

    direct = (basis @ z.detach()).float()
    scores = {name: query_dice(v) for name, v in (
        ("zero", torch.zeros_like(ambient)), ("ambient_oracle", ambient),
        ("oracle_projection", (basis @ projected).float()), ("direct_latent", direct))}
    return dict(schema="R8_SOURCE_VAL_CAPACITY_ONE_V1", protocol_sha256=PROTOCOL_SHA256,
                amplitude=segmenter.amplitude, space=space, anchor=anchor,
                support_groups=list(support_names), query_groups=list(query_names),
                optimizer_lr=CAPACITY_OPTIMIZER["lr"], optimizer_steps=CAPACITY_OPTIMIZER["steps"],
                projection=projection_audit, direct_support_objective=float(loss.detach()),
                query_soft_Dice=scores)
