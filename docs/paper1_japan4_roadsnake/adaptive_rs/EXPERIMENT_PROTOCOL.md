# Frozen experiment protocol

All formal comparisons use Japan4-cleanV3 (`D00`, `D10`, `D20`, `D40`) and the exact B0/R1 training recipe:

```text
pretrained = yolo26n.pt
fresh start = true
imgsz = 640
batch = 32
workers = 8
seed = 42
deterministic = true
optimizer = auto
AMP = true
epochs = 1, 30, or 100 as an experiment fixed from launch
```

Learning rate, momentum, weight decay, warmup, augmentation, close-mosaic, loss gains, assignment, confidence, IoU, max detections, O2O/O2M structure, and inference scoring come from `configs/experiments/japan4_cleanv3_adaptive_roadsnake_matched.yaml` and must not be opportunistically changed.

Selection and diagnosis are Val-only. The Test split is sealed. `scripts/evaluate_adaptive_roadsnake.py` hard-codes `--splits val` because the legacy evaluator defaults to both Val and Test.

Each formal run is a fresh run from the original pretrained parent. A 1E smoke is engineering evidence only and cannot be resumed into 30E/100E. A 30E run cannot be resumed into 100E. No automatic rescue sweep or stacked module is authorized.

Required output includes P/R, AP50, AP50-95, AP75, AP-small/medium/large, AR100, D00/D10/D20/D40 AP and AP75, best epoch, late-epoch trend, parameters, GFLOPs, measured latency, VRAM, scale diagnostics, checkpoint hashes, and physical-pruning equivalence.
