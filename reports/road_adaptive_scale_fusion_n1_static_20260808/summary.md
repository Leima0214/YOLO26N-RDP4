# RoadAdaptiveScaleFusion-N1 static audit (RTX 4090)

Scope: Japan4-cleanV3 train-only fixed batch; no Val, no Test, and no training was started.

## Structure

- YOLO26 layers 0-22 are unchanged.
- Layer 23 receives the original Detect inputs `[16, 19, 22]` (P3/P4/P5).
- It downsamples P3 once, keeps P4 as the identity main path, and projects/upsamples P5 once.
- A local DWConv3x3 + Conv1x1 predicts three per-pixel softmax scale weights.
- Output is `F4 + gamma * (Fmix - F4)`, with `gamma=0.01`.
- Detect moves from layer 23 to 24 and reads `[16, 23, 22]`; only its P4 input is adapted.

## Initialization and equivalence

| Item | Result |
|---|---:|
| Total legal pretrained parameter coverage | 94.6416% |
| Backbone 0-10 coverage | 100% |
| Neck 11-22 coverage | 100% |
| Detect regression coverage | 100% |
| Detect classification coverage | 9.9459% (80-class incompatible tensors remain random) |
| New adapter parameters | 47,556, all random by design |
| Trainer reconstruction | exact SHA-256 match |
| Detect P3 cosine vs B0 | 1.0000001 |
| Detect P4 cosine vs B0 | 0.9999991 |
| Detect P5 cosine vs B0 | 0.9999999 |
| Detect P4 relative L2 | 0.006703 |

Fixed non-augmented train batch losses:

| Model | box | cls | dfl | total |
|---|---:|---:|---:|---:|
| B0 | 2.597813 | 12.389026 | 0.032646 | 77.884079 |
| N1 | 2.612833 | 11.987196 | 0.033254 | 77.521942 |

## Cost and engineering checks

| Model | Params | GFLOPs | fused FP32 B1 median | AMP peak VRAM |
|---|---:|---:|---:|---:|
| B0 | 2,505,360 | 5.7782 | 5.1063 ms | 174.75 MiB |
| N1 | 2,552,916 | 5.8550 | 5.9376 ms | 209.25 MiB |
| Delta | +1.90% | +1.33% | +16.28% | +19.74% |

- Forward, true detection loss, backward, AMP, finite gradients, and ONNX opset 17: PASS.
- `gamma`, all adapter parameter groups, and all three P3/P4/P5 input tensors have finite non-zero gradients.
- Initial scale means are `[0.32567, 0.33894, 0.33539]`; the random gate begins near balanced but is spatially non-constant.
- ONNX output size: 9,963,988 bytes.

## Verdict

Static GO: N1 satisfies the pretraining-preservation and near-B0 initialization gate. It is not an AP result and no 30E run was started. The measured PyTorch latency/VRAM cost is materially larger than its Params/GFLOPs increase, so any future training promotion should require a meaningful accuracy gain rather than a marginal one.

Remote JSON SHA-256: `b34b4b605a54aab30483eb885e53fd29654e73e5e6cc30ea6ff8bc4ab794126a`

Remote ONNX SHA-256: `9fb712f6affbbe592e6f8b51a9582ea22e350c0fe1e069db2b910f6178b7823b`
