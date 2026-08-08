# Japan4 parent initialization decision

Date: 2026-08-08

## Decision

Neither current candidate may start formal 30E because both replacement backbones have zero legal pretrained coverage. No Test split was read and no training was started.

| Current candidate | Own-backbone pretrained coverage | YOLO tail coverage | YOLO Neck coverage | YOLO Head coverage | Random parameter fraction | 30E |
|---|---:|---:|---:|---:|---:|---|
| `yolo26-StarNet.yaml` | 0.00% | 100.00% | 98.62% | 63.24% | 67.65% | Blocked |
| `yolo26-MobileMamba-Backbone.yaml` | 0.00% | 100.00% | 100.00% | 63.24% | 44.41% | Blocked |

`tail` means the unchanged SPPF and C2PSA modules. Head coverage is below 100% primarily because the official 80-class classification outputs are intentionally not loaded into the four-class head.

## Exact incompatibilities

### Current StarNet

- Resolved local stage channels are 32/64/128/256 after a 16-channel stem; resolved block depths are `[1,1,4,2]`.
- It adds a learnable residual scale and omits the official DropPath graph.
- Released official variants are S1 `[2,2,8,3]`, S2 `[1,2,6,2]`, S3 `[2,2,8,4]`, and S4 `[3,3,12,5]`, with different base dimensions.
- The difference is structural, not a key-prefix issue. A conversion map would be invalid.

### Current MobileMamba

- Local `nano` is a four-stage detector-side convolution/router adaptation with dimensions `[24,48,96,160,224]` and four adapted blocks.
- Official variants are T2/T4/S6/B1/B2/B4 and use the WTE-Mamba/selective-scan/wavelet implementation with three stages and different dimensions.
- The difference is structural and semantic. No official checkpoint tensor may be assigned to the local backbone merely because a shape happens to match.

## Wrapper correction feasibility

### StarNet-S1: feasible and recommended

The author's S1 checkpoint was downloaded and strict-loaded into the author's model with every key matched. At 640 input, official stage outputs are:

- stage 0: `[1,24,160,160]` (stride 4)
- stage 1: `[1,48,80,80]` (stride 8)
- stage 2: `[1,96,40,40]` (stride 16)
- stage 3: `[1,192,20,20]` (stride 32)

Therefore an official-equivalent YOLO wrapper can keep the complete S1 stem/stages pretrained, expose stages 1/2/3 as P3/P4/P5, and use new 1x1 adapters after them. This must be a separately named candidate, not silently substituted under the existing blogger YAML.

Checkpoint evidence:

- 342 state items, strict load passed
- official stem/stage trainable parameters: 2,675,784
- SHA256: `4d00fd7acab420dfe93d443222ebf5e89fac5caa4c6d43ebf423fcd8957ed23b`

### Official MobileMamba: not a clean P3/P4/P5 replacement

The official patch embedding reaches stride 16 before its stages. Its detection implementation exposes stride-16/32/64 features. A YOLO P3/P4/P5 wrapper would have to expose an intermediate patch tensor and discard or newly fuse the last official stage, or change an official stride. That is a material architecture redesign, not a checkpoint conversion.

## Recommended next action

1. Implement and separately name `StarNet-S1-YOLO26` using the exact official S1 stem/stages and post-backbone adapters.
2. Verify 100% coverage of all retained official S1 stem/stage parameters, semantic YOLO tail/Neck/Head inheritance, Trainer reconstruction, forward/loss/backward/AMP and P3/P4/P5 shapes.
3. Only if that audit passes, run StarNet-S1 30E.
4. Keep the current MobileMamba candidate frozen. Do not run it from random backbone initialization. Revisit it only as an explicitly approved P4/P5/P6 or multi-scale redesign experiment.
