# RoadLite-26 Stage-1 static audit

Protocol: nc=4, input=640x640. GFLOPs = 2 x Conv/Linear MACs; B0 differs from the recorded THOP value by <0.1%.

| Model | Params | Params vs B0 | GFLOPs | GFLOPs vs B0 | Backbone P3/P4/P5 | Detect P3/P4/P5 | Transfer params |
|---|---:|---:|---:|---:|---|---|---:|
| B0 | 2,505,360 | +0.0% | 5.6238 | +0.0% | [128, 128, 256] | [64, 128, 256] | 96.4% |
| A1 | 1,315,328 | -47.5% | 4.2117 | -25.1% | [128, 96, 160] | [64, 96, 160] | 12.5% |
| A2 | 980,016 | -60.9% | 3.7505 | -33.3% | [128, 80, 128] | [64, 80, 128] | 16.8% |
| A3 | 1,168,596 | -53.4% | 4.4623 | -20.7% | [144, 96, 128] | [72, 96, 128] | 4.2% |

## PyTorch CUDA latency

| Model | B1 median | B1 p95 | B32 throughput | B32 peak VRAM |
|---|---:|---:|---:|---:|
| B0 | 9.792 ms | 11.784 ms | 3211.7 img/s | 489.2 MiB |
| A1 | 10.180 ms | 11.379 ms | 3077.6 img/s | 487.0 MiB |
| A2 | 10.181 ms | 10.491 ms | 3100.8 img/s | 486.3 MiB |
| A3 | 10.366 ms | 12.569 ms | 3023.2 img/s | 486.7 MiB |

## Complete channel mapping

| i | module | B0 | A1 | A2 | A3 |
|---:|---|---:|---:|---:|---:|
| 0 | Conv | 16 | 16 | 16 | 16 |
| 1 | Conv | 32 | 32 | 32 | 32 |
| 2 | C3k2 | 64 | 64 | 64 | 64 |
| 3 | Conv | 64 | 64 | 64 | 64 |
| 4 | C3k2 | 128 | 128 | 128 | 144 |
| 5 | Conv | 128 | 96 | 80 | 96 |
| 6 | C3k2 | 128 | 96 | 80 | 96 |
| 7 | Conv | 256 | 160 | 128 | 128 |
| 8 | C3k2 | 256 | 160 | 128 | 128 |
| 9 | SPPF | 256 | 160 | 128 | 128 |
| 10 | C2PSA | 256 | 160 | 128 | 128 |
| 11 | Upsample | 256 | 160 | 128 | 128 |
| 12 | Concat | 384 | 256 | 208 | 224 |
| 13 | C3k2 | 128 | 96 | 80 | 96 |
| 14 | Upsample | 128 | 96 | 80 | 96 |
| 15 | Concat | 256 | 224 | 208 | 240 |
| 16 | C3k2 | 64 | 64 | 64 | 72 |
| 17 | Conv | 64 | 64 | 64 | 72 |
| 18 | Concat | 192 | 160 | 144 | 168 |
| 19 | C3k2 | 128 | 96 | 80 | 96 |
| 20 | Conv | 128 | 96 | 80 | 96 |
| 21 | Concat | 384 | 256 | 208 | 224 |
| 22 | C3k2 | 256 | 160 | 128 | 128 |

## B0 per-layer audit

| i | from | module | params | MACs | activation MiB | output |
|---:|---|---|---:|---:|---:|---|
| 0 | -1 | ultralytics.nn.modules.conv.Conv | 464 | 44.237M | 6.250 | `[1, 16, 320, 320]` |
| 1 | -1 | ultralytics.nn.modules.conv.Conv | 4,672 | 117.965M | 3.125 | `[1, 32, 160, 160]` |
| 2 | -1 | ultralytics.nn.modules.block.C3k2 | 6,640 | 163.840M | 6.250 | `[1, 64, 160, 160]` |
| 3 | -1 | ultralytics.nn.modules.conv.Conv | 36,992 | 235.930M | 1.562 | `[1, 64, 80, 80]` |
| 4 | -1 | ultralytics.nn.modules.block.C3k2 | 26,080 | 163.840M | 3.125 | `[1, 128, 80, 80]` |
| 5 | -1 | ultralytics.nn.modules.conv.Conv | 147,712 | 235.930M | 0.781 | `[1, 128, 40, 40]` |
| 6 | -1 | ultralytics.nn.modules.block.C3k2 | 87,040 | 137.626M | 0.781 | `[1, 128, 40, 40]` |
| 7 | -1 | ultralytics.nn.modules.conv.Conv | 295,424 | 117.965M | 0.391 | `[1, 256, 20, 20]` |
| 8 | -1 | ultralytics.nn.modules.block.C3k2 | 346,112 | 137.626M | 0.391 | `[1, 256, 20, 20]` |
| 9 | -1 | ultralytics.nn.modules.block.SPPF | 164,608 | 65.536M | 0.391 | `[1, 256, 20, 20]` |
| 10 | -1 | ultralytics.nn.modules.block.C2PSA | 249,728 | 98.765M | 0.391 | `[1, 256, 20, 20]` |
| 11 | -1 | torch.nn.modules.upsampling.Upsample | 0 | 0.000M | 1.562 | `[1, 256, 40, 40]` |
| 12 | [-1, 6] | ultralytics.nn.modules.conv.Concat | 0 | 0.000M | 2.344 | `[1, 384, 40, 40]` |
| 13 | -1 | ultralytics.nn.modules.block.C3k2 | 119,808 | 190.054M | 0.781 | `[1, 128, 40, 40]` |
| 14 | -1 | torch.nn.modules.upsampling.Upsample | 0 | 0.000M | 3.125 | `[1, 128, 80, 80]` |
| 15 | [-1, 4] | ultralytics.nn.modules.conv.Concat | 0 | 0.000M | 6.250 | `[1, 256, 80, 80]` |
| 16 | -1 | ultralytics.nn.modules.block.C3k2 | 34,304 | 216.269M | 1.562 | `[1, 64, 80, 80]` |
| 17 | -1 | ultralytics.nn.modules.conv.Conv | 36,992 | 58.982M | 0.391 | `[1, 64, 40, 40]` |
| 18 | [-1, 13] | ultralytics.nn.modules.conv.Concat | 0 | 0.000M | 1.172 | `[1, 192, 40, 40]` |
| 19 | -1 | ultralytics.nn.modules.block.C3k2 | 95,232 | 150.733M | 0.781 | `[1, 128, 40, 40]` |
| 20 | -1 | ultralytics.nn.modules.conv.Conv | 147,712 | 58.982M | 0.195 | `[1, 128, 20, 20]` |
| 21 | [-1, 10] | ultralytics.nn.modules.conv.Concat | 0 | 0.000M | 0.586 | `[1, 384, 20, 20]` |
| 22 | -1 | ultralytics.nn.modules.block.C3k2 | 463,104 | 183.962M | 0.391 | `[1, 256, 20, 20]` |
| 23 | [16, 19, 22] | ultralytics.nn.modules.head.Detect | 242,736 | 433.664M | 5.988 | `[[1, 300, 6], {'one2many': {'boxes': [1, 4, 8400], 'scores': [1, 4, 8400], 'feats': [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]}, 'one2one': {'boxes': [1, 4, 8400], 'scores': [1, 4, 8400], 'feats': [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]}}]` |
