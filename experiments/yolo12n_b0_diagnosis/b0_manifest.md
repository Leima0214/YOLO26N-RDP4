# B0 manifest

```json
{
  "git": {
    "commit": "eb6e250d49204631e84f180a4f78f64a968c6a28",
    "branch": "codex/rgt-gbrg-r10v-t1",
    "status_porcelain": "M configs/svrdd7_remote.yaml\n M scripts/eval_o2m_box_vote_generic.py\n M ultralytics/nn/tasks.py\n?? archive/\n?? configs/experiments/svrdd7_hmr_snake_h3.yaml\n?? configs/experiments/svrdd7_morphology_m1.yaml\n?? configs/experiments/svrdd7_rs_mid_v1.yaml\n?? configs/svrdd7_rs_mid_remote.yaml\n?? docs/experiments/hmr_snake_h3_v1.md\n?? docs/experiments/rs_mid_v1_svrdd7.md\n?? docs/experiments/rs_mid_v2_channel_aggregation_proposal.md\n?? docs/experiments/rs_v1_performance_v1.md\n?? docs/experiments/svrdd7_morphology_m1.md\n?? experiments/svrdd7/B0_100E_seed42_frozen.md\n?? experiments/svrdd7/RS_30E_SCREEN.md\n?? experiments/svrdd7/RS_VARIANTS.md\n?? experiments/yolo12n_b0_diagnosis/\n?? experiments/yolo12n_b0_diagnosis_audit_v1/\n?? experiments/yolo12n_b0_diagnosis_audit_v2/\n?? experiments/yolo12n_b0_diagnosis_smoke/\n?? runtime_meta/\n?? scripts/compare_rs_mid_eval_parity.py\n?? scripts/diagnose_svrdd7_b0_rsv1_oracle_ranking.py\n?? scripts/diagnose_svrdd7_b0_rsv1_quality_survival.py\n?? scripts/diagnose_svrdd7_d2_failure_decomposition.py\n?? scripts/diagnose_svrdd7_d3_early_detail_survival.py\n?? scripts/diagnose_svrdd7_rsv1_n3.py\n?? scripts/diagnose_yolo12n_b0_route.py\n?? scripts/eval_svrdd7_hmr_snake_h3.py\n?? scripts/eval_svrdd7_morphology_m1.py\n?? scripts/eval_svrdd7_rs_mid.py\n?? scripts/eval_svrdd7_rs_p4_study.py\n?? scripts/eval_svrdd7_yolo_official.py\n?? scripts/rs_mid_bootstrap.py\n?? scripts/rs_mid_experiment.py\n?? scripts/rs_mid_o2m.py\n?? scripts/run_svrdd7_b0_100e_20260904.sh\n?? scripts/run_svrdd7_gbrg_100e_eval_20260904.sh\n?? scripts/run_svrdd7_gcf_p4_v0_30e_20260907.sh\n?? scripts/run_svrdd7_hmr_snake_h3_30e_20260908.sh\n?? scripts/run_svrdd7_hmr_snake_h3_smoke_20260907.sh\n?? scripts/run_svrdd7_morphology_m1_30e_20260908.sh\n?? scripts/run_svrdd7_o2m_schedule_30e_20260907.sh\n?? scripts/run_svrdd7_rs_gbrg_paired30_20260904.sh\n?? scripts/run_svrdd7_rs_p4_30e_pair_20260907.sh\n?? scripts/run_svrdd7_rs_v1_30e_20260904.sh\n?? scripts/run_svrdd7_rs_v2_30e_20260904.sh\n?? scripts/run_svrdd7_rs_v3_30e_20260904.sh\n?? scripts/run_svrdd7_rs_v4_base_v1_30e_20260904.sh\n?? scripts/run_svrdd7_single_components_30e_20260904.sh\n?? scripts/run_svrdd7_straight_rs_30e_20260908.sh\n?? scripts/smoke_svrdd7_deploy.py\n?? scripts/smoke_svrdd7_runtime.py\n?? scripts/svrdd7_rs_variants_common.py\n?? scripts/test_svrdd7_factorial_30e.py\n?? scripts/test_svrdd7_rs_30e.py\n?? scripts/test_svrdd7_rs_variants.py\n?? scripts/train_svrdd7_factorial_30e.py\n?? scripts/train_svrdd7_gcf_p4_v0.py\n?? scripts/train_svrdd7_hmr_snake_h3.py\n?? scripts/train_svrdd7_morphology_m1.py\n?? scripts/train_svrdd7_o2m_schedule.py\n?? scripts/train_svrdd7_rs_30e.py\n?? scripts/train_svrdd7_rs_detail_v1.py\n?? scripts/train_svrdd7_rs_gbrg_30e_legacy.py\n?? scripts/train_svrdd7_rs_mid_baseline.py\n?? scripts/train_svrdd7_rs_mid_detail_v1.py\n?? scripts/train_svrdd7_rs_mid_v1.py\n?? scripts/train_svrdd7_rs_mid_v2.py\n?? scripts/train_svrdd7_rs_v1.py\n?? scripts/train_svrdd7_rs_v1_30e.py\n?? scripts/train_svrdd7_rs_v1_performance_v1.py\n?? scripts/train_svrdd7_rs_v2.py\n?? scripts/train_svrdd7_rs_v2_30e.py\n?? scripts/train_svrdd7_rs_v3.py\n?? scripts/train_svrdd7_rs_v3_30e.py\n?? scripts/train_svrdd7_rs_v4.py\n?? scripts/train_svrdd7_rs_v4_30e.py\n?? scripts/train_svrdd7_yolo11.py\n?? scripts/train_svrdd7_yolo_official.py\n?? scripts/verify_rs_mid_v1.py\n?? scripts/verify_rs_v1_performance_v1.py\n?? sources/\n?? tests/test_gcf_p4.py\n?? tests/test_hmr_snake_h3.py\n?? tests/test_morphology_m1.py\n?? tests/test_rs_mid_v1.py\n?? tests/test_straight_rsv1.py\n?? ultralytics/cfg/models/11/yolo11-svrdd7.yaml\n?? ultralytics/cfg/models/26/yolo26-rs-detail-v1.yaml\n?? ultralytics/cfg/models/26/yolo26-rs-mid-detail-v1.yaml\n?? ultralytics/cfg/models/26/yolo26-rs-mid-v1.yaml\n?? ultralytics/cfg/models/26/yolo26-rs-mid-v2.yaml\n?? ultralytics/nn/gcf_p4.py\n?? ultralytics/nn/hmr_snake.py\n?? ultralytics/nn/morphology_m1.py\n?? ultralytics/nn/roadsnake_factorial.py\n?? ultralytics/nn/roadsnake_mid.py\n?? ultralytics/nn/roadsnake_rs_only_variants.py\n?? ultralytics/nn/roadsnake_v1_performance.py\n?? ultralytics/nn/roadsnake_variants.py\n?? ultralytics/nn/rs_detail_companion.py\n?? yolo12n.pt"
  },
  "artifacts": {
    "best_pt": "/root/YOLO26N-RDP4/runs/paper1_svrdd7_official/YOLO12N_OFFICIAL_SVRDD7_100E_seed42/weights/best.pt",
    "best_sha256": "76a35d4bd5b7c9ed499c30b9b272d2c206ea10f81273ce4b5ccd4b040d38690b",
    "last_pt": "/root/YOLO26N-RDP4/runs/paper1_svrdd7_official/YOLO12N_OFFICIAL_SVRDD7_100E_seed42/weights/last.pt",
    "last_sha256": "b856a8f9e90ba3597ab613b3553bb32b810948c11b6bb7029ae360dfb995f76a",
    "results_csv": "/root/YOLO26N-RDP4/runs/paper1_svrdd7_official/YOLO12N_OFFICIAL_SVRDD7_100E_seed42/results.csv",
    "args_yaml": "/root/YOLO26N-RDP4/runs/paper1_svrdd7_official/YOLO12N_OFFICIAL_SVRDD7_100E_seed42/args.yaml",
    "data_yaml": "/root/YOLO26N-RDP4/runtime_meta/YOLO12N_OFFICIAL_SVRDD7_100E_seed42/data_snapshot.yaml"
  },
  "training": {
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "seed": 42,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 10
  },
  "software": {
    "ultralytics": "8.4.2",
    "python": "3.11.15 | packaged by conda-forge | (main, Jun 11 2026, 03:34:02) [GCC 14.3.0]",
    "torch": "2.5.1+cu124"
  },
  "model": {
    "parameters": 2569413,
    "GFLOPs": 6.486272,
    "detect_type": "Detect",
    "end2end": false,
    "strides": [
      8.0,
      16.0,
      32.0
    ]
  },
  "selection": {
    "best_epoch": 75,
    "native_AP50": 0.66517,
    "native_AP50_95": 0.3916
  },
  "canonical_val": {
    "AP": 0.39384085077843045,
    "AP50": 0.662674771977008,
    "AP75": 0.4022365534568944,
    "AP_small": 0.16845470787676053,
    "AP_medium": 0.34671495420441334,
    "AP_large": 0.6082789252731532,
    "AR100": 0.5921029107724026
  },
  "protocol": {
    "split": "val",
    "images": 1000,
    "imgsz": 640,
    "batch": 32,
    "score_floor": 0.001,
    "error_conf": 0.25,
    "nms_iou": 0.7,
    "max_det": 300,
    "test_accessed": false,
    "class_names": [
      "LC",
      "TC",
      "AC",
      "P",
      "MC",
      "LP",
      "TP"
    ],
    "class_note": "D00/D10/D20/D40 are absent from SVRDD7; no mapping was invented."
  }
}
```
