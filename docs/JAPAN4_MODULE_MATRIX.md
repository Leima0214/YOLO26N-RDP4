# RDD2022-Japan4-Clean-v1 module matrix

All experiments retain the YOLO26 P3/P4/P5 Detect head, O2O/O2M branches,
NMS-free O2O inference, `reg_max=1`, pretrained `yolo26n.pt`, and the matched
30-epoch training protocol in `train.py`.

| Experiment | YAML | Change |
|---|---|---|
| B0 | `ultralytics/cfg/models/26/yolo26.yaml` | Native YOLO26n |
| SOM | `yolo26n-japan4-som.yaml` | Star-operation residual at backbone P3/P4 |
| MAF | `yolo26n-japan4-maf.yaml` | Identity-initialized attentive/deformable fusion at four neck joins |
| WTC | `yolo26n-japan4-wtc.yaml` | Haar wavelet convolution refinement at neck P3/P4/P5 |
| SOM+MAF | `yolo26n-japan4-som-maf.yaml` | Backbone interaction plus aligned fusion |
| SOM+WTC | `yolo26n-japan4-som-wtc.yaml` | Backbone interaction plus frequency refinement |
| MAF+WTC | `yolo26n-japan4-maf-wtc.yaml` | Aligned fusion plus frequency refinement |
| SOM+MAF+WTC | `yolo26n-japan4-som-maf-wtc.yaml` | Full road-damage candidate |

Only change the top-level `MODEL` selector in `train.py`. Do not change the
dataset, seed, image size, optimizer, augmentation, or validation settings
between rows.

Run order after B0:

1. WTC, because it has the smallest parameter increase and the strongest
   direct small-damage motivation.
2. MAF, because it targets cross-scale/background interference.
3. SOM, because it is the most expensive single module.
4. Run pairwise and full combinations only after their relevant single modules
   have completed; keep negative combinations as ablation evidence rather than
   rescue-tuning them.

Primary promotion gate: O2O mAP50-95 at least B0 + 0.005, no material D00/D10
regression, and no single-epoch-only spike. Always record O2M, per-class AP,
parameters, GFLOPs, latency, and peak VRAM as secondary evidence.
