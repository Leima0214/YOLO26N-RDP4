# Paper 1 external SVRDD protocol

## Claim boundary

SVRDD is a seven-class external-dataset validation of architectural transfer. It is not zero-shot because each candidate is trained on the SVRDD training split. Absolute AP is not compared across SVRDD and Japan4; only matched within-SVRDD deltas support attribution.

## Frozen data contract

- Root: `/SVRDD/SVRRDD_YOLO_READY`
- Classes in exact ID order: `LC, TC, AC, P, MC, LP, TP`
- Expected images: train 6000, val 1000, test 1000
- Selection and diagnosis: Val only
- Test: sealed until candidate, evaluator, and any deployment transformation are frozen
- COCO annotations: `annotations/instances_val.json` and `annotations/instances_test.json`

The dataset audit must pass label bounds, normalized box validity, corrupt-image checks, unique stems, and zero cross-split stem overlap. Count deviations require an explicit dataset-version decision.

## Matched training

All 100E candidates use image size 640, batch 32, seed 42, workers 8, deterministic mode, optimizer `auto`, and the same Japan4-derived augmentation and learning-rate protocol. Baseline, RoadSnake-R1, and RoadSnake-GBRG differ only in the declared model YAML. The Trainer reconstruction callback must report `head_nc: 7` and preserve every name-and-shape-compatible pretrained tensor.

R10 is exactly ten fresh epochs from a selected `best.pt`: `resume=False`, with a new optimizer, schedule, and warmup. It is not an extension of the original optimizer state.

## Evaluation and reporting

Use the generic O2M + score-box-vote evaluator with confidence floor 0.001, NMS IoU 0.70, `max_det=300`, and COCO `maxDets=100`. Report overall AP/AP50/AP75, scale AP, AR100, and the same metrics for every class. Missing classes, category-name mismatches, duplicate COCO stems, or `head.nc != len(names)` are hard failures.

Any T1 checkpoint must be produced with `scripts/prune_roadsnake_t1.py` and retain its equivalence report and SHA256. Training logs, initialization audits, exit codes, evaluator JSON, and checkpoint hashes are mandatory evidence.
