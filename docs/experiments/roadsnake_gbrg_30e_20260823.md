# RoadSnake-GBRG 30E experiment record (2026-08-23)

## Frozen protocol

- Dataset: Japan4-cleanV3, Val only; Test was not read.
- Model: RoadSnake-R1 with the P3-only GBRG training auxiliary branch.
- Training: fresh start from `yolo26n.pt`, 30 epochs, image size 640, batch 32, workers 8, seed 42, deterministic, optimizer `auto`, AMP enabled.
- No changes were made to the RoadSnake-R1 P4 adapter, inference graph, detection loss, learning-rate recipe, or augmentation recipe.

## Completion and artifacts

- Run: `yolo26n-japan4-roadsnake-gbrg_cleanv3_30e_seed42_20260823`
- Remote run path: `/root/YOLO26N-RDP4/runs/paper1_japan4_clean/yolo26n-japan4-roadsnake-gbrg_cleanv3_30e_seed42_20260823`
- Training completed 30/30 epochs and emitted `GBRG_DONE`.
- Native training-log best: epoch 30, P 0.55357, R 0.49777, AP50 0.50673, AP50-95 0.24115.
- Trainer initialization audit passed: 635 compatible state items were preserved; RoadSnake and GBRG-head hashes were unchanged through Trainer reconstruction; initial RoadSnake gamma was 0.
- Final controller audit at epoch 30: lambda 0.160808, raw gradient ratio 0.165354, weighted gradient ratio 0.026590, region loss 0.148303. The weighted auxiliary gradient remained close to the 3% design target and below the 8% safety limit.

## Unified COCO Val evaluation of best.pt

| Metric | Value |
|---|---:|
| Precision | 0.55187 |
| Recall | 0.49832 |
| AP50 | 0.50417 |
| AP50-95 | 0.24135 |
| AP75 | 0.18762 |
| AP-small | 0.09949 |
| AP-medium | 0.19805 |
| AP-large | 0.27415 |
| AR100 | 0.56618 |

| Class | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| D00 | 0.22924 | 0.46891 | 0.18966 | 0.57383 |
| D10 | 0.16098 | 0.40517 | 0.08699 | 0.51985 |
| D20 | 0.34679 | 0.66043 | 0.33107 | 0.62194 |
| D40 | 0.22838 | 0.48218 | 0.14274 | 0.54911 |

Efficiency recorded by the same evaluator: 2,395,819 parameters, 5.255 GFLOPs, PyTorch batch-1 latency 7.005 ms, checkpoint size 5.189 MB.

## Decision

- Versus B0-S30 AP 0.23570: +0.00565 AP.
- Versus the frozen R1-S30 reference AP approximately 0.24039: about +0.00096 AP.
- This is a weak positive signal over R1, but it does not meet the preregistered +0.002 AP strong-promotion condition.
- A fresh 100E run was subsequently authorized explicitly by the user. It must not resume this 30E checkpoint and must not change the GBRG mechanism or protocol.
