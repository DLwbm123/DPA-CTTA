# Real-model injection contract

Discovery ran on CPU with synthetic `torch.linspace` inputs, no checkpoint, and the public `DLwbm123/CTTA` checkout pinned at `dbff0d985c6c95345d9fb78f5b1daef57b392564`. The script enumerated `named_modules()`, captured every executed 4-D Tensor output during one full forward, retained decoder/tail candidates, selected three distinct spatial scales, and evaluated an exact autograd directional JVP from each module output to final logits.

For the JVP, the feature-dependent direction is deliberately treated as a fixed tangent; only its scalar perturbation is differentiated. This isolates the module-output-to-logits Jacobian being tested.

## Fundus ResUNet34

Input: `[1, 3, 512, 512]`; frozen descriptor feature: `res.conv1`, Tensor `[1, 64, 256, 256]`.

| Module path | Output | Channels | Directional logits JVP norm |
|---|---:|---:|---:|
| `up1` | Tensor `[1, 256, 32, 32]` | 256 | 1.5237574577 |
| `up3` | Tensor `[1, 256, 128, 128]` | 256 | 19.5884227753 |
| `seg_head` | Tensor `[1, 2, 512, 512]` | 2 | 53.9338340759 |

Zero-state logits were bitwise identical to the unwrapped full-model logits. One shared nonzero 16-D state changed the final logits. All JVPs were finite and nonzero.

## Polyp PraNet

Input: `[1, 3, 352, 352]`; frozen descriptor feature: `resnet.conv1.0`, Tensor `[1, 32, 176, 176]`.

| Module path | Output | Channels | Directional logits JVP norm |
|---|---:|---:|---:|
| `ra4_conv5` | Tensor `[1, 1, 11, 11]` | 1 | 260.2180786133 |
| `ra3_conv4` | Tensor `[1, 1, 22, 22]` | 1 | 331.4739074707 |
| `ra2_conv4` | Tensor `[1, 1, 44, 44]` | 1 | 19.2921428680 |

Zero-state logits were bitwise identical to the unwrapped full-model logits. One shared nonzero 16-D state changed the final logits. All JVPs were finite and nonzero.

## Reproduction

```bash
export DPA_CTTA_BASE_ROOT=/path/to/pinned/DLwbm123-CTTA-checkout
CUDA_VISIBLE_DEVICES="" python scripts/discover_injection_points.py
```

The public pin omits the `ctta-repro-suite/third_party` directory expected by its own loader. The integration bridge therefore creates a temporary symlink view of `VPTTA/OPTIC` and `VPTTA/POLYP` from the same commit, calls `ctta_suite.models.build_model/model_logits`, then removes the view. No upstream file is modified or copied.
