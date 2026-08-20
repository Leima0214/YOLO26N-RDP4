# SA-RS method

RoadSnake-R1 learns how a five-point horizontal or vertical sampler bends through cumulative orthogonal offsets, but its main-axis positions remain fixed at `[-2,-1,0,1,2]`. At P4 stride 16 this fixes the end-to-end main-axis span at 64 input pixels. The Japan4 geometry audit showed this span is relatively large for small damage and relatively short for large damage.

SA-RS preserves every R1 operation and predicts two dense scale fields from the same reduced P4 feature:

```text
z_h, z_v = Conv3x3(F_reduced)
s_h = exp(log(2.5) * tanh(z_h))
s_v = exp(log(2.5) * tanh(z_v))
```

The scale-head weights and biases are zero initialized, hence both scales start at exactly 1 and are bounded approximately to `[0.4, 2.5]`.

Only the main axis changes:

```text
horizontal: grid_x = base_x + positions * s_h
            grid_y = base_y + cumulative_offset_h * max_offset

vertical:   grid_y = base_y + positions * s_v
            grid_x = base_x + cumulative_offset_v * max_offset
```

Thus, scale determines how far to look while the frozen R1 offset mechanism determines how to bend. `K=5`, `max_offset=1`, fixed point weights, normalization, local branch, fuse branch, scalar gamma, P4-only placement, and Detect semantics are unchanged. Gamma remains zero initialized, so SA-RS step-0 output is bit-identical to B0 even though its adaptive branch is present.

This first experiment forbids scale supervision, regularization, class-specific rules, orientation prediction, additional losses, attention, new neck paths, P3/P5 insertion, kernel/range sweeps, SCHM, distillation, and continuation from an R1 checkpoint. Formal training must start fresh from the same original `yolo26n.pt` used by B0/R1.
