# StarNet and MobileMamba checkpoint compatibility audit (2026-08-08)

## Query and fallback

- Query: official StarNet and MobileMamba ImageNet checkpoint architectures and exact compatibility with the local YOLO26 implementations.
- The configured research API was unavailable because no API key was installed. The audit therefore used the two authors' official repositories and release/config source directly.

## StarNet

- Official source: https://github.com/ma-xu/Rewrite-the-Stars
- Official checkpoint release: https://github.com/ma-xu/Rewrite-the-Stars/releases/tag/checkpoints_v1
- Released pretrained variants are S1-S4. Their base dimensions/depths are S1 24/[2,2,8,3], S2 32/[1,2,6,2], S3 32/[2,2,8,4], and S4 32/[3,3,12,5].
- The local YOLO `n` variant resolves to a different stage layout and includes a learnable residual scale absent from the official StarBlock.
- Verdict: no published official StarNet checkpoint is a strict module-name, semantic, and shape match. Do not load official StarNet backbone weights.

## MobileMamba

- Official source: https://github.com/lewandofskee/MobileMamba
- Official variants are T2, T4, S6, B1, B2, and B4. They use three MobileMamba stages with official dimensions such as T2 [144,272,368], T4 [176,368,448], S6 [192,384,448], and B-series [200,376,448].
- The local `MobileMambaBackboneYOLO(nano)` is a four-stage convolutional/router implementation with dimensions [24,48,96,160,224], rather than the official WTE-Mamba/selective-scan/wavelet block graph.
- Verdict: no published official MobileMamba checkpoint is a strict module-name, semantic, and shape match. Do not load official MobileMamba backbone weights.

## Frozen initialization decision

- B0: exact-name-and-shape YOLO26 checkpoint loading.
- StarNet: random new backbone; explicit source layers 9-23 to target layers 9-23 mapping for retained SPPF/C2PSA, Neck, and Detect.
- MobileMamba: random new backbone/adapters; explicit source layers 9-23 to target layers 4-18 mapping for retained SPPF/C2PSA, Neck, and Detect.
- Every mapped tensor must pass both a declared semantic-layer mapping and exact shape equality. The four-class classifier outputs are intentionally not forced from the 80-class checkpoint.
- Formal 30E is blocked while backbone coverage remains zero.

## Can the wrappers be corrected to official-equivalent backbones?

### StarNet: yes, as a new S1 candidate

- The released S1 checkpoint was downloaded from the author's release and verified as a real checkpoint (342 state items; SHA256 `4d00fd7acab420dfe93d443222ebf5e89fac5caa4c6d43ebf423fcd8957ed23b`).
- An official-equivalent detection wrapper can retain the exact S1 stem and all four stages, return stage-1/stage-2/stage-3 as stride-8/16/32 features, and place random 1x1 adapters after them for YOLO channel compatibility.
- This would legally load all official stem/stage parameters. It is not the current blogger `yolo26-StarNet.yaml`; it must be named and audited as an official StarNet-S1 backbone candidate.

### MobileMamba: not without a material detection-scale redesign

- The smallest official pretrained family member is T2, not `nano`.
- The official patch embedding reaches stride 16 before the three stages. The official detection implementation exposes indices `(1,2,3)`, corresponding to stride-16/32/64 features rather than YOLO's stride-8/16/32 inputs.
- Producing P3/P4/P5 would require exposing an intermediate patch-embedding tensor and either discarding the final pretrained stage or adding a new fusion path. Changing a patch stride would also stop being an official-equivalent graph.
- Therefore the current MobileMamba candidate cannot be corrected by a key conversion alone. It should remain blocked unless a separately named P4/P5/P6 detector or an explicitly redesigned multi-scale wrapper is approved.
