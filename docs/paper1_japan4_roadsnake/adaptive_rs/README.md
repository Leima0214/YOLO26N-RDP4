# Adaptive RoadSnake experiment package

This directory freezes two executable, single-variable experiments built from the verified RoadSnake-R1 parent.

- **SA-RS (Scale-Adaptive RoadSnake):** changes only the longitudinal sampling span of the existing P4 RoadSnake.
- **MG-SA-RS (Metric-Guided SA-RS):** a strict SA-RS extension that conditions the scale logits on detached local deviation and Sobel proxy cues.

The original `ultralytics/nn/roadsnake.py`, B0/R1 topology, P4 placement, curved offsets, loss, matcher, O2O/O2M semantics, inference scoring, dataset protocol, and optimizer recipe remain unchanged. Both candidates start with scale 1 and scalar residual gate gamma 0, so the initial deployed output is exactly B0.

## Status

- SA-RS: engineering candidate; static audit must pass before any 1E smoke.
- MG-SA-RS: **LOCKED / WAIT FOR SA-RS RESULT**. Formal training is mechanically blocked until a Val-only SA-RS 100E physically-pruned result reaches AP50-95 >= 0.2499.
- Test: sealed throughout selection and diagnosis.

## Static audits

```bash
python scripts/audit_adaptive_roadsnake_static.py --variant sa --device 0
python scripts/audit_adaptive_roadsnake_static.py --variant mgsa --device 0
```

## SA-RS commands (prepared, not automatically executed)

```bash
python scripts/train_sa_rs.py --data configs/japan4_clean_v3_remote.yaml --epochs 1 --imgsz 640 --batch 32 --device 0 --workers 8 --seed 42 --name sa_rs_cleanv3_e1_seed42
python scripts/train_sa_rs.py --data configs/japan4_clean_v3_remote.yaml --epochs 30 --imgsz 640 --batch 32 --device 0 --workers 8 --seed 42 --name sa_rs_cleanv3_e30_seed42
python scripts/train_sa_rs.py --data configs/japan4_clean_v3_remote.yaml --epochs 100 --imgsz 640 --batch 32 --device 0 --workers 8 --seed 42 --name sa_rs_cleanv3_e100_seed42
```

Val-only evaluation and deployment pruning:

```bash
python scripts/evaluate_adaptive_roadsnake.py --checkpoint SA_FULL=runs/paper1_japan4_clean/sa_rs_cleanv3_e100_seed42/weights/best.pt --data configs/japan4_clean_v3_remote.yaml --output reports/sa_rs_e100_full_val
python scripts/prune_adaptive_roadsnake.py --variant sa --source runs/paper1_japan4_clean/sa_rs_cleanv3_e100_seed42/weights/best.pt --output runs/paper1_japan4_clean/sa_rs_cleanv3_e100_seed42/weights/best_pruned_native.pt --report reports/sa_rs_e100_pruning.json --onnx reports/sa_rs_e100_pruned_native.onnx --device 0
python scripts/evaluate_adaptive_roadsnake.py --checkpoint SA_PRUNED=runs/paper1_japan4_clean/sa_rs_cleanv3_e100_seed42/weights/best_pruned_native.pt --data configs/japan4_clean_v3_remote.yaml --output reports/sa_rs_e100_pruned_val
python scripts/diagnose_adaptive_roadsnake_scale.py --checkpoint runs/paper1_japan4_clean/sa_rs_cleanv3_e100_seed42/weights/best.pt --data configs/japan4_clean_v3_remote.yaml --split val --device 0 --output reports/sa_rs_e100_scale_diagnosis.json
```

## MG-SA-RS commands (documentation only; currently locked)

Every MG training command requires the SA pruned Val-only `metrics.json` as an administrative unlock artifact.

```bash
python scripts/train_mg_sa_rs.py --sa-pruned-metrics reports/sa_rs_e100_pruned_val/metrics.json --data configs/japan4_clean_v3_remote.yaml --epochs 1 --imgsz 640 --batch 32 --device 0 --workers 8 --seed 42 --name mg_sa_rs_cleanv3_e1_seed42
python scripts/train_mg_sa_rs.py --sa-pruned-metrics reports/sa_rs_e100_pruned_val/metrics.json --data configs/japan4_clean_v3_remote.yaml --epochs 30 --imgsz 640 --batch 32 --device 0 --workers 8 --seed 42 --name mg_sa_rs_cleanv3_e30_seed42
python scripts/train_mg_sa_rs.py --sa-pruned-metrics reports/sa_rs_e100_pruned_val/metrics.json --data configs/japan4_clean_v3_remote.yaml --epochs 100 --imgsz 640 --batch 32 --device 0 --workers 8 --seed 42 --name mg_sa_rs_cleanv3_e100_seed42
```

If and only if MG later unlocks and finishes, its Val-only deployment closure is:

```bash
python scripts/evaluate_adaptive_roadsnake.py --checkpoint MG_FULL=runs/paper1_japan4_clean/mg_sa_rs_cleanv3_e100_seed42/weights/best.pt --data configs/japan4_clean_v3_remote.yaml --output reports/mg_sa_rs_e100_full_val
python scripts/prune_adaptive_roadsnake.py --variant mgsa --source runs/paper1_japan4_clean/mg_sa_rs_cleanv3_e100_seed42/weights/best.pt --output runs/paper1_japan4_clean/mg_sa_rs_cleanv3_e100_seed42/weights/best_pruned_native.pt --report reports/mg_sa_rs_e100_pruning.json --onnx reports/mg_sa_rs_e100_pruned_native.onnx --device 0
python scripts/evaluate_adaptive_roadsnake.py --checkpoint MG_PRUNED=runs/paper1_japan4_clean/mg_sa_rs_cleanv3_e100_seed42/weights/best_pruned_native.pt --data configs/japan4_clean_v3_remote.yaml --output reports/mg_sa_rs_e100_pruned_val
python scripts/diagnose_adaptive_roadsnake_scale.py --checkpoint runs/paper1_japan4_clean/mg_sa_rs_cleanv3_e100_seed42/weights/best.pt --data configs/japan4_clean_v3_remote.yaml --split val --device 0 --output reports/mg_sa_rs_e100_scale_diagnosis.json
```

Do not run those commands unless the unlock gate is genuinely met. See the remaining documents for the frozen mechanism, protocol, gates, diagnostics, and pruning definition.
