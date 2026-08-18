# SCHM static design audit — 2026-08-18

## Verdict

**Static GO.** SCHM passed pretrained inheritance, step-0 equivalence, real-batch forward/loss/backward, AMP, finite-gradient, batch-isolation, candidate legality, explicit index mapping, Trainer-rebuild, ONNX, and checkpoint round-trip checks. Test remained sealed; calibration and static batches came from the Japan4-cleanV3 Train split only.

This is an engineering and mechanism gate, not evidence that SCHM improves AP. The formal 100E matched experiment is required for that claim.

## Frozen definition

SCHM adds one training-only localization term to native YOLO26 end-to-end training:

```text
L_total = L_native_O2M + L_native_O2O + lambda_schm * L_SCHM
```

For each GT, the native post-conflict O2M positive set proposes one candidate index. Selection uses detached box IoU. A candidate enters `L_SCHM` only when it is a different index from the native O2O match and has positive detached IoU gain. The prediction optimized at that index is the **O2O prediction**, and the target is the **real GT**. O2M logits, scores, boxes, features, and quality values are never used as targets. No classification, scoring, matcher, inference, or postprocess rule is changed.

The loss reuses native O2O localization primitives and normalizes by total GT count, not by the sum of harvest weights. Consequently the auxiliary signal naturally decays when the O2O candidate catches up.

## Gradient calibration

Calibration report: `reports/schm_family_static_20260818/schm_lambda_calibration.json` on the remote audit host.

- 10 fixed Train batches, batch 32, workers 0, seed 42.
- At `lambda_schm=1`, gradient ratio to native O2O box regression: median `0.015413`, p90 `0.030803`.
- Because lambda 1 was already conservative, it was not amplified to force the nominal 5–10% band.
- Frozen value: **`lambda_schm=1.0`**.
- Across calibration batches, initial harvest ratio was approximately `0.449–0.632`; illegal harvest count was always zero.

## Static evidence

| Check | Result |
|---|---:|
| B0 state items inherited | 708 / 708 |
| Shared missing or mismatched tensors | 0 |
| Shared tensors changed after load | 0 |
| Tensors changed by Trainer reconstruction | 0 |
| Step-0 inference max abs error vs B0 | 0.0 |
| Fixed-shape batch-isolation raw O2O error | 0.0 |
| Full-precision gradients finite | 366 / 366 |
| AMP gradients finite | 366 / 366 |
| SCHM gradient into O2M selection prediction | 0.0 |
| SCHM gradient into harvested O2O prediction, L1 | 0.153637 |
| ONNX checker | PASS |
| Candidate params | 2,505,360, identical to B0 |

Real-batch audit contained 16 GTs: 7 legal new-index harvests, 0 conflicts, 0 missing O2M references, 0 missing O2O references, and 0 illegal harvests. The same-index-positive-gain share was 0.30, so the new-index-only hypothesis remains meaningfully testable rather than being dominated by same-index reweighting.

## Explicit index proof

For a 640 input, both native O2M and O2O use this exact flatten layout:

| Scale | Shape | Number of positions | Global offset |
|---|---:|---:|---:|
| P3 | 80 × 80 | 6,400 | 0 |
| P4 | 40 × 40 | 1,600 | 6,400 |
| P5 | 20 × 20 | 400 | 8,000 |

Within each scale the order is row-major `(y, x)`. The audit checked equality of level shapes, anchors, and strides for O2M/O2O and performed 100 deterministic random `global_index -> (scale,y,x) -> global_index` round trips; all passed.

## Full 1E engineering smoke

Fresh smoke run:

```text
/root/YOLO26N-RDP4/runs/smoke_schm_family/
yolo26n-japan4-schm_cleanv3_100e_seed42_20260818_SMOKE1E_20260818_143229
```

- Exit code 0; 198/198 Train batches and full Val completed.
- Peak observed training memory: approximately 4.71 GB.
- `best.pt` and `last.pt` were both saved (5,385,278 bytes each).
- Framework reloaded `best.pt` and completed Val, proving checkpoint round trip.
- `best.pt` SHA256: `e21f722ab8cc00616f1a332be617e423351a9895fe2667dab09f34c934ee89eb`.
- Epoch audit: 7,800 harvests, harvest ratio `0.38047`, same-index ratio `0.30720`, conflict count 0, illegal count 0, missing O2M rate `0.00112`, missing O2O rate `0.000439`.
- Mean IoU gain `0.01198`, mean weight `0.04421`, SCHM/native-O2O-box loss ratio `0.00911`, measured gradient ratio `0.01255`.

The default Ultralytics results-curve plotter emitted a non-fatal index error because it assumes three loss columns while this trainer records four. Training, weights, JSONL statistics, checkpoint reload, and Val all completed successfully; this cosmetic plot incompatibility is not a model gate.

## Implementation surface

- `ultralytics/utils/schm_loss.py`
- `ultralytics/models/yolo/detect/schm_train.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/cfg/models/26/yolo26n-japan4-schm.yaml`
- `configs/experiments/japan4_cleanv3_schm_100e.yaml`
- `scripts/calibrate_schm_lambda.py`
- `scripts/verify_japan4_schm_family.py`
- `scripts/train_schm_family.py`
- `scripts/run_schm_cleanv3_100e.sh`

Validated source commit: `566e80d89007dd10a0e705c7d3c1b1cd525187e8`.
