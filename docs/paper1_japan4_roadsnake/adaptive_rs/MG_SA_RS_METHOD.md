# MG-SA-RS method

Status: **LOCKED / WAIT FOR SA-RS RESULT**.

MG-SA-RS asks only whether explicit local structure and edge proxy cues help the SA-RS scale predictor choose `s_h` and `s_v`. It is not MGFFBConcat, does not change the neck, and does not claim to reproduce a named paper's exact metric formulation.

From the reduced P4 feature `F`, the implementation computes:

```text
G  = channel_mean(F)
D  = abs(G - AvgPool3x3(G))
Ex = abs(Sobel_x(G))
Ey = abs(Sobel_y(G))
c_norm = tanh(c / (mean_spatial(c) + eps))
metric_cues = concat(D_norm, Ex_norm, Ey_norm)
```

Normalization is per image and per cue. No GT, labels, predicted boxes, BatchNorm, or dataset statistics are used. The cue tensor is detached so it conditions scale prediction without adding a separate cue-to-P4 gradient path.

The SA base scale predictor is retained and a zero-initialized metric residual is added:

```text
z = scale_head(F) + scale_metric(metric_cues)
s = exp(log(2.5) * tanh(z))
```

When `scale_metric` is zero, MG-SA-RS is exactly SA-RS. At initialization both predictors are zero, scale is 1, gamma is 0, and the detector output is exactly B0. No metric gate, attention, auxiliary loss, manual edge-to-direction assignment, or branch weighting is allowed.

MG formal training unlocks only when SA-RS 100E, after physical pruning and the unified Val-only evaluator, reaches AP50-95 >= 0.2499. MG must not be used to rescue a failed SA experiment.
