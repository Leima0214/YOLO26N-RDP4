# Japan4-cleanV3 S1/G1/GS1 experiment design — 2026-08-06

## Frozen protocol

- Classes: `D00`, `D10`, `D20`, `D40`.
- Split selection: train and Val only; Test is forbidden.
- Training: image size 640, batch 32, workers 8, device 0, seed 42, deterministic AMP, `optimizer=auto`.
- Matched schedule: 30E candidates use total `epochs=30`; formal candidates use a fresh official initialization and total `epochs=100`.
- A 30E `last.pt` must never be resumed into a formal 100E run.

The unified entry also locks the hyperparameters recorded by the matched B0-S30 and S1-S30 `args.yaml`: non-cosine LR, `lr0=0.01`, `lrf=0.01`, momentum 0.937, weight decay 0.0005, three warm-up epochs, mosaic 1.0, no mixup/copy-paste, close-mosaic 10, confidence 0.001, IoU 0.7, max detections 300, and a five-epoch checkpoint interval.

## S1 is frozen

S1 uses `StripDetect` and `StripAwareResidual` from `ultralytics/nn/modules/head.py`. Its committed model YAML is `ultralytics/cfg/models/26/yolo26n-japan4-s1-strip-regression.yaml`; the verified YAML blob is `762640f899bfb28e0a83ca8dc047b59dc0e09579`.

The implementation remains unchanged:

- P3/P4 box-regression inputs receive 3x3, 1x7/1x5, and 7x1/5x1 depthwise branches.
- A global softmax gate combines the three branches.
- Zero-initialized `gamma` makes the step-0 mapping exactly the identity.
- Classification, P5, strides, `reg_max`, top-k, scoring, output dimensions, and both assignment paths remain unchanged.
- One-to-many and one-to-one regression adapters are symmetric.

The 4090 static check reported 728/728 transferred tensors, zero step-0 output error, 2,513,184 parameters (+0.312%), 5.831 GFLOPs (+0.922%), and fused batch-1 latency +4.53%. A later sequential benchmark was biased by GPU clock drift; a 300-pair alternating benchmark resolved S1 at +4.23%, G1 at +0.02%, and GS1 at +4.47%. The unified verifier now uses this paired method and enforces the predeclared 5% gate.

## G1: Shape-aware Gaussian Region Guidance

G1 is implemented by `RegionGuidedDetect` and `RegionGuidedE2ELoss`.

- Training-only 1x1 heads read the shared P3/P4 features and produce one logit map per level.
- Each normalized GT xywh box is projected into the relevant feature coordinates.
- The target uses anisotropic standard deviations `max(w/6, 1)` and `max(h/6, 1)`; multiple GT maps are combined by pixelwise maximum.
- Empty-GT images produce all-zero targets.
- Stable soft-target BCE is used. Background pixels receive weight 0.1 rather than a hard full-weight negative, limiting damage from incomplete crack annotations.
- Region logits do not modulate features, classification scores, regression, top-k, or final ranking.
- In evaluation they are not called; `fuse()` removes the auxiliary heads before deployment/export.

The fixed P3/P4 weights are 54.5. They were selected once from a deterministic seed-42 real 640/batch-32 train batch containing 115 GT boxes: detection loss sum 362.80646, unscaled region losses 0.124804/0.101112, detection shared-feature gradient norm 6.77328, and region gradient norm 0.00031076 at the provisional weight 0.05. The provisional ratio was only 0.00004588; linear scaling to 54.5 gives approximately 0.05001, the smallest accepted 5% boundary. No Val data or AP was read. The same fixed values are committed in G1 and GS1.

## GS1: organic S1 + G1 composition

`StripRegionGuidedDetect` applies G1 supervision to the same shared P3/P4 tensors consumed by S1, while the S1 regression path itself is unchanged. Thus:

- G1 trains shared foreground representation.
- S1 models directional localization.
- There is no region-score multiplication, feature mask, dynamic S1 gate, ranking change, or third loss.
- After fusion, GS1 is structurally the S1 deployment model.

## Static evidence before remote GPU preflight

The focused eight-test suite passes. CPU static verification at image size 128 provides implementation evidence only; the formal 4090 verification at 640 remains authoritative.

| Candidate | inherited tensors | step-0 max error | Params delta | GFLOPs delta | required active gradient |
|---|---:|---:|---:|---:|---|
| S1 | 728/728 | 0 | +0.312% | +0.922% | S1 gamma |
| G1 | 712/712 | 0 | +0.008% training model | +0.021% training model | P3/P4 region heads |
| GS1 | 732/732 | 0 | +0.320% training model | +1.033% training model | region heads and S1 gamma |

Deployment verification is performed after `fuse()`: G1 region heads must be absent, G1 must match fused B0 exactly, and GS1 must match fused S1 exactly.

## Experiment order and gates

1. Run S1 1E smoke and fresh formal S1-100E.
2. After S1 is evaluated and archived, calibrate G1 once, then run G1 1E and fresh G1-30E.
3. Run GS1 1E/30E only if G1 passes or is AP-neutral with a clear D10 benefit.
4. Run G1-100E only if G1-30E passes; run GS1-100E only if GS1-30E beats the best single module sufficiently.

The exact numerical gates and commands are in `docs/experiments/japan4_s1_g1_gs1_long_run_20260806.md`.

## Known artifact limitation

The current server contains the matched B0-S30 and S1-S30 checkpoints but not the Japan4-cleanV3 B0-V3 100E checkpoint. The ledger records B0-V3 best epoch 82, Val AP 0.24142, AP50 0.52184, AP75 0.19609, AP-small 0.09924, AR100 0.50709, and checkpoint SHA256 `76fd04c527d212e3f7e0b747c33ff65c463e535bf555e46376b1bbf8850be656`. Those values may be used as a frozen headline reference, but missing per-class metrics must not be invented; unified re-evaluation waits until that exact checkpoint is recovered.
