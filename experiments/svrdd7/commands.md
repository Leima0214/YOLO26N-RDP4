# Commands

Run from the repository root. Keep Test sealed until the Val decision is frozen.

```bash
python scripts/audit_svrdd7_dataset.py --root /SVRDD/SVRRDD_YOLO_READY --strict-counts --output experiments/svrdd7/dataset_audit.json
python scripts/prepare_svrdd7.py --root /SVRDD/SVRRDD_YOLO_READY --split all

python scripts/train_svrdd7_baseline.py --name svrdd7_yolo26n_baseline_100e_seed42
python scripts/train_svrdd7_roadsnake_r1.py --name svrdd7_roadsnake_r1_100e_seed42
python scripts/train_svrdd7_roadsnake_gbrg.py --name svrdd7_roadsnake_gbrg_100e_seed42

python scripts/eval_o2m_box_vote_generic.py --checkpoint runs/paper1_svrdd7/svrdd7_roadsnake_gbrg_100e_seed42/weights/best.pt --data configs/svrdd7_remote.yaml --split val --output reports/svrdd7_gbrg_val_box_vote

python scripts/train_svrdd7_r10_native.py --parent runs/paper1_svrdd7/svrdd7_roadsnake_gbrg_100e_seed42/weights/best.pt --name svrdd7_gbrg_r10_native_10e_seed42

python scripts/prune_roadsnake_t1.py --source runs/paper1_svrdd7/svrdd7_roadsnake_r1_100e_seed42/weights/best.pt --output reports/svrdd7_r1_t1/best_t1.pt --report reports/svrdd7_r1_t1/prune_report.json --device 0
```

Only after the model choice and thresholds are frozen:

```bash
python scripts/eval_o2m_box_vote_generic.py --checkpoint PATH/TO/FROZEN/best.pt --data configs/svrdd7_remote.yaml --split test --output reports/svrdd7_frozen_test
```
