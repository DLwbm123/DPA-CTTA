"""Data-free LAST transfer checks; no third-party code imports or execution."""
import json
import math
import time


def run():
    start = time.perf_counter()
    n = 0
    for z in [-8, -2, -0.1, 0, 0.1, 2, 8]:
        for mean in [-3, 0, 1]:
            for std in [0, 0.01, 1, 100]:
                assert ((z - mean) / (std + 1e-5) >= 0) == (z >= mean)
                n += 1
    # B=1, same initialization and same stream: equations differ before any model update.
    mean, paper_std, archive_std = 0.0, 1.0, 1.0
    for _ in range(40):
        z = 2.0
        paper_std = math.sqrt(0.9 * paper_std**2 + 0.1 * (z - mean)**2)
        archive_std *= 0.9
        mean = 0.9 * mean + 0.1 * z
    assert abs(paper_std - archive_std) > 0.1
    assert math.isclose(archive_std, 0.9**40)
    # Spatial centering can expand foreground without receiving correctness information.
    dense_logits = [-4.0, -4.0, -4.0, 1.0]
    spatial_mean = sum(dense_logits) / len(dense_logits)
    shifted = [-4.0, -4.0, -1.0, 1.0]
    shifted_mean = sum(shifted) / len(shifted)
    raw = sum(z >= 0 for z in shifted)
    centered = sum(z >= shifted_mean for z in shifted)
    assert raw == 1 and centered == 2
    # Positive temperature can change entropy gradient direction for a shared parameter vector.
    def entropy_gradient(z, std):
        a = z / std
        p = 1 / (1 + math.exp(-a))
        return -a * p * (1 - p) / std
    ratios = [entropy_gradient(1, s) / entropy_gradient(3, s) for s in [1, 2]]
    assert not math.isclose(*ratios)
    # Common shifts are not an invariance of independent Bernoulli decisions.
    assert [z >= 0 for z in [-1, -2]] != [z + 1.5 >= 0 for z in [-1, -2]]
    return {
        "status": "PASS", "scope": "data_free_formula_qualification_only",
        "mask_identity_cases": n,
        "b1_constant_stream": {"arrivals": 40, "mean": mean,
                               "paper_std": paper_std, "archive_std": archive_std},
        "spatial_counterexample": {"raw_positive": raw, "centered_positive": centered,
                                    "first_example_mean": spatial_mean},
        "entropy_gradient_ratios_std1_std2": ratios,
        "gpu_runs": 0, "model_forwards": 0, "image_label_weight_reads": 0,
        "body_seconds": time.perf_counter() - start,
        "limitations": "Algebra/examples only; no Dice, efficacy, or runtime admission evidence."
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
