# Physical pruning protocol

SA-RS and MG-SA-RS are training-time residual experiments whose native deployment checkpoint is defined by setting the learned scalar gamma to zero and physically removing the adaptive adapter.

The valid comparison is:

```text
trained full model
-> copy with gamma=0
-> physically remove road_snake and restore native Detect
```

It is invalid to compare a full model with nonzero gamma directly against the pruned model. The learned gamma is recorded before pruning.

`scripts/prune_adaptive_roadsnake.py` performs the following checks without overwriting the source checkpoint:

1. Full checkpoint produces finite outputs.
2. Gamma-zero reference and physically-pruned native Detect are bit-exact before serialization.
3. Both `model` and `ema`, when present, are pruned.
4. The saved checkpoint reloads through `YOLO()` with exact native `Detect` class.
5. Reloaded output remains bit-exact to the gamma-zero reference.
6. No `road_snake`, scale, metric, or Sobel state remains.
7. Parameter count and GFLOPs equal native B0.
8. The pruned native ONNX exports and passes `onnx.checker`.
9. Source/output/ONNX SHA256, latency, and VRAM are recorded.

The full and pruned checkpoints must each be evaluated separately with the Val-only wrapper. The pruned Val result controls the 100E promotion decision.
