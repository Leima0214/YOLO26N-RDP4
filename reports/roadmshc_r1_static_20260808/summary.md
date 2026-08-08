# RoadMSHC-R1 static audit

## Decision

**GO for a fresh pretrained 30E experiment.** No training was started during this audit, and Test was not read.

R1 preserves the complete pretrained B0 path and adds one random P4 adapter through an identity shortcut with learnable `gamma=0.01`. Its initialization is effectively B0-equivalent and avoids the feature-distribution break observed in M1.

## Initialization and engineering

- B0 P3, P4, P5/SPPF/C2PSA, Neck, and Detect regression coverage: 100% in every group.
- RoadMSHC adapter pretrained coverage: 0% by design; 72,641 new random parameters.
- Total semantic pretrained parameter coverage: 2,416,120 / 2,578,001 = 93.72%.
- Trainer reconstruction hash: unchanged (`18bf6a9...848e`).
- Forward, real detection loss, backward, AMP, finite gradients, ONNX: PASS.
- All adapter groups have finite nonzero gradients: gamma, reduce, 3x3, 5x5, 7x7, horizontal, vertical, fuse, gate.

## B0-equivalent initialization

| Probe | B0 | M1 | R1 |
|---|---:|---:|---:|
| Fixed-batch total loss | 77.8841 | 87.7381 | 78.3264 |
| box loss | 2.59781 | 2.98684 | 2.60871 |
| cls loss | 12.38903 | 15.04847 | 12.23584 |
| dfl loss | 0.032646 | 0.037623 | 0.032671 |

R1-vs-B0 feature cosine: backbone P3/P4/P5 = 1.000000 / 1.000000 / 1.000000; Detect P3/P4/P5 = 1.000000 / 1.000000 / 1.000000 (rounded to six decimals). M1-vs-B0 values were -0.09997 / 0.07181 / 0.43690 and 0.81080 / 0.74571 / 0.61730.

## Cost on RTX 4090

- Params: 2,578,001 (`+72,641`, `+2.90%` vs B0).
- GFLOPs: 5.9739 (`+3.39%` vs B0).
- Fused FP32 batch-1 latency: median 6.001 ms (recorded, not used as a hard gate).
- Static audit peak allocated VRAM: 207.61 MiB.
- ONNX: PASS, 10,061,364 bytes.

## Japan4 train-GT P4 spans (stride 16)

Values are P25/P50/P75/P90 feature cells, with augmentation disabled.

| Class | Long side | Short side |
|---|---|---|
| D00 | 4.07 / 6.13 / 9.13 / 12.80 | 2.27 / 3.53 / 5.53 / 8.00 |
| D10 | 6.00 / 8.53 / 12.40 / 17.87 | 1.27 / 1.67 / 2.27 / 3.07 |
| D20 | 9.47 / 14.53 / 21.80 / 31.73 | 5.80 / 9.60 / 13.73 / 16.53 |
| D40 | 2.60 / 3.53 / 5.00 / 7.40 | 1.53 / 2.11 / 2.97 / 4.33 |
| All | 5.07 / 8.53 / 14.47 / 22.53 | 1.93 / 3.60 / 8.00 / 13.60 |

The current 7-cell strip is a defensible first test: it is close to the all-object and D10 median long side and covers the upper range of D40. The wide D20 distribution argues against claiming that one larger kernel is universally optimal; kernel changes remain deferred until R1 establishes an AP signal.
