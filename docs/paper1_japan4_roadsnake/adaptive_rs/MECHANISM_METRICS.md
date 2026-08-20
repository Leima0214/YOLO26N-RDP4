# Mechanism metrics

The diagnostic interface is opt-in, detached, read-only, and excluded from the loss. It exposes:

```text
s_h, s_v
raw horizontal/vertical offsets
cumulative horizontal/vertical offsets
horizontal and vertical sampling coordinates
residual feature
scalar gamma
MG proxy cues and base/metric logits (MG only)
```

For each scale field report mean, standard deviation, P10/P25/P50/P75/P90, minimum, maximum, `s<0.45` ratio, and `s>2.40` ratio. Aggregate globally and by D00/D10/D20/D40 and COCO small/medium/large bins.

Required relationships are:

- Spearman(GT width, horizontal scale).
- Spearman(GT height, vertical scale).
- Spearman(GT area, mean horizontal/vertical scale).

These are mechanism diagnostics, not promotion substitutes. A favorable correlation cannot rescue a detection AP failure, and global correlation alone cannot prove useful within-object behavior.

Static validation must additionally confirm scale exactly 1 at initialization; R1/SA geometry equality at scale 1; forced 0.5/1/2 spans of 32/64/128 input pixels at P4 stride 16; unchanged orthogonal curve; per-image batch isolation; finite nonzero scale, offset, residual, gamma, and shared-P4 gradients; and finite CUDA AMP loss/gradients. MG also requires finite detached cues, a zero metric branch that exactly reduces to SA, and spatially varying scale under a controlled nonzero metric probe.
