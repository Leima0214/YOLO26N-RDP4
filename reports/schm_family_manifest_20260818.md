# Japan4 SCHM-family experiment manifest — 2026-08-18

## Protocol boundary

All new formal runs are fresh starts from the same original YOLO26n pretrained initialization as B0. They use Japan4-cleanV3, 100 epochs, image size 640, batch 32, workers 8, seed 42, deterministic mode, AMP, `optimizer=auto`, and the frozen B0 augmentation/close-mosaic recipe. Only Val may be read. Test is sealed. No RoadSnake/T1 checkpoint is a training initializer for SCHM or RS-SCHM.

## Lineage and causal variables

| Model | Definition and unique variable | Initialization | Train-only components | Inference components | Expected pruning | Code/record |
|---|---|---|---|---|---|---|
| B0 | Native YOLO26n end-to-end detector | original `yolo26n.pt` | native O2M | native O2O | none | frozen B0 record; AP 0.24142 |
| RoadSnake-R1 | Pretrained-preserving P4 RoadSnake residual in the shared feature path | original `yolo26n.pt`, gamma 0 | none | RoadSnake active | none | `48a508c`; AP 0.24727 |
| RoadSnake-T1 | R1 training trajectory, then physical RoadSnake removal | original `yolo26n.pt`, gamma 0 | RoadSnake during training | native detector after pruning | required | `fbc60a0`, runner fix `f14183d`; AP 0.24694, AP75 0.20667, AR100 0.52715 |
| SCHM | Native detector plus new-index-only GT localization at legal O2M-proposed indices | original `yolo26n.pt` | `L_SCHM` only | native detector | auxiliary criterion absent at inference | implementation `abe221b`, validated `566e80d` |
| RS-SCHM | RoadSnake modifies only O2M P4 proposals; SCHM optimizes native O2O prediction at harvested index | original `yolo26n.pt`, gamma 0 | RoadSnake O2M proposer + `L_SCHM` | native detector | RoadSnake physically removed | implementation `abe221b`, validated `566e80d` |

The historical metrics above are frozen context, not re-evaluated by this implementation audit.

## Reproducible configurations

### SCHM

- Model: `ultralytics/cfg/models/26/yolo26n-japan4-schm.yaml`
- Experiment: `configs/experiments/japan4_cleanv3_schm_100e.yaml`
- Run wrapper: `scripts/run_schm_cleanv3_100e.sh`
- Direct runner: `python -u scripts/train_schm_family.py --variant schm --config configs/experiments/japan4_cleanv3_schm_100e.yaml`
- Frozen lambda: `1.0`
- Formal run base name: `yolo26n-japan4-schm_cleanv3_100e_seed42_20260818`

### RS-SCHM

- Model: `ultralytics/cfg/models/26/yolo26n-japan4-rs-schm.yaml`
- Experiment: `configs/experiments/japan4_cleanv3_rs_schm_100e.yaml`
- Run wrapper: `scripts/run_rs_schm_cleanv3_100e.sh`
- Direct runner: `python -u scripts/train_schm_family.py --variant rs-schm --config configs/experiments/japan4_cleanv3_rs_schm_100e.yaml`
- Frozen lambda: `1.0`
- Formal run base name: `yolo26n-japan4-rs-schm_cleanv3_100e_seed42_20260818`
- Formal post-training artifact: `best_pruned_native.pt`

Both shell wrappers use `set -euo pipefail`, check CUDA and the resolved Train/Val paths, print Git/config provenance, refuse run-directory overwrite, tee a unique log, and execute a Val-only final evaluation. RS-SCHM additionally performs pruning, equality, cost, Val, and ONNX checks.

## Audit status

| Gate | SCHM | RS-SCHM |
|---|---|---|
| Static real-batch audit | GO | GO |
| Step-0 B0 inference equivalence | exact | exact |
| O2M-selection gradient isolation | PASS | PASS |
| Explicit 100-index mapping proof | PASS | PASS |
| Trainer reconstruction | PASS | PASS |
| FP32 / AMP finite backward | PASS / PASS | PASS / PASS |
| ONNX | PASS | PASS |
| Full checkpoint round trip | PASS, fresh 1E smoke | covered by shared criterion serialization; formal pruning still required |
| Test sealed | PASS | PASS |

Branch: `codex/japan4-schm-rs-schm`.

Validated implementation/checkpoint-safety commit: `566e80d89007dd10a0e705c7d3c1b1cd525187e8`.
