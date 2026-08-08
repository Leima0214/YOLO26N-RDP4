# Japan4-cleanV3 MSHC static audit

Date: 2026-08-08
GPU: NVIDIA GeForce RTX 4090
Scope: fixed non-augmented train batch plus synthetic loss/backward; Test was not read; no training epoch was started.

## Verdict

`yolo26-MSHC.yaml` is statically eligible for a pretrained 30E experiment. The current candidate must be named `M1: P3+P4 MSHC`; it is not the article's stated `M2: P4x2 MSHC` placement.

## Structure

- Layer 4: B0 P3 `C3k2` replaced by `MSHCBlock`, input 64, output 128, hidden 64.
- Layer 6: B0 P4 `C3k2` replaced by `MSHCBlock`, input/output 128, hidden 64.
- Each block executes parallel depthwise 3x3, 5x5, 7x7, horizontal 1x7 and vertical 7x1 branches.
- The five outputs are concatenated, fused by 1x1 convolution, globally channel-gated, and added to `proj(F)`.
- P3/P4 branch output shapes and all 44 MSHC parameter gradients were finite and non-zero.
- Caveat: P4 still applies a random 1x1 `proj` even though input and output channels match, so the residual baseline is not an identity mapping at initialization.

## Pretrained transfer

| Component | Loaded / total parameters | Coverage |
|---|---:|---:|
| Unchanged Backbone | 1,252,352 / 1,252,352 | 100% |
| P3 MSHC | 0 / 81,088 | 0%, expected random |
| P4 MSHC | 0 / 89,280 | 0%, expected random |
| Neck | 897,152 / 897,152 | 100% |
| Detect regression | 143,640 / 143,640 | 100% |
| Detect classification | 9,856 / 99,096 | 9.946% |
| Whole target | 2,303,000 / 2,562,608 | 89.869% |

No MSHC parameter was accidentally loaded. All unchanged matched items passed layer-index, module-type, name and shape checks. Trainer reconstruction preserved the full initialized state bit-exactly (`5089b76...` before and after).

## Static comparison

| Metric | B0 | MSHC M1 | Delta |
|---|---:|---:|---:|
| Params | 2,505,360 | 2,562,608 | +2.285% |
| GFLOPs | 5.7782 | 6.2607 | +8.350% |
| Fused FP32 B1 latency | 5.113 ms | 5.936 ms | +16.089% |
| AMP batch-1 peak allocation | 174.75 MiB | 255.88 MiB | +46.425% |

Forward, real detection loss, backward, AMP, finite gradients, Detect P3/P4/P5 shapes (`64x80x80`, `128x40x40`, `256x20x20`) and ONNX export all passed. The MSHC ONNX file is 9,991,839 bytes.

The memory increase is much larger than the parameter increase because five P3/P4 branch activations are concatenated before fusion.

## Initial pretrained interface

On the same eight non-augmented train images, MSHC versus initialized B0 feature cosine was `-0.100` at Backbone P3, `0.072` at P4 and `0.437` at P5. After the inherited PAN, Detect-input cosine recovered to `0.811`, `0.746`, and `0.617`.

Initial real-batch loss was finite but higher:

| Loss | B0 | MSHC M1 |
|---|---:|---:|
| box | 2.5978 | 2.9868 |
| cls | 12.3890 | 15.0485 |
| dfl | 0.03265 | 0.03762 |

Initial one-to-one classification confidence remained well calibrated and close to B0 (`mean 0.000493` versus `0.000483`), so this is an adaptation burden rather than the confidence-collapse failure observed in the stopped official StarNet-S1 run.

## Decision

Static Go for `M1 pretrained 30E`, subject to a fresh-start training entry that loads only the audited name-and-shape/semantic matches and re-verifies the initialized state after Trainer reconstruction. Keep `M2 P4x2` frozen as a later position ablation; do not combine or silently substitute it for M1.
