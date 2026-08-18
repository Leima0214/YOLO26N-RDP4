# RS-SCHM static design audit — 2026-08-18

## Verdict

**Static GO.** RS-SCHM passed the shared SCHM gates plus the RoadSnake routing and gradient gates. Test remained sealed. This does not authorize a performance claim; RS-SCHM must be run as a fresh matched experiment only after SCHM finishes.

## Frozen definition

RS-SCHM is not shared-feature RoadSnake-R1 plus SCHM. Its training graph is deliberately asymmetric:

```text
native P3/P4/P5 ------------------------> native O2O and inference
              \
               P4 -> RoadSnake(gamma=0, K=5, e=0.25) -> O2M proposer
                                                        |
                                        detached legal candidate index
                                                        |
                                      native O2O[index] -> real GT
```

RoadSnake modifies only the training-time O2M P4 proposal path. Native O2O always sees the native feature. Native O2M loss is allowed to train RoadSnake and the shared upstream network, while SCHM selection (`q_m`, `q_o`, delta, weight, argmax, and index) is detached. In evaluation mode the RoadSnake branch is skipped.

## Pretrained and step-0 evidence

| Check | Result |
|---|---:|
| B0 shared state items inherited | 708 / 708 |
| Shared missing or mismatched tensors | 0 |
| Shared tensors changed after load | 0 |
| Shared tensors changed by Trainer reconstruction | 0 |
| Step-0 inference max abs error vs B0 | 0.0 |
| Fixed-shape batch-isolation raw O2O error | 0.0 |
| Added train-only parameters | 20,395 |
| Total train-time params | 2,525,755 (+0.814%) |
| Full-precision gradients finite | PASS |
| AMP gradients finite | PASS |
| ONNX checker | PASS |

At `gamma=0`, RS-SCHM has the same initial inference and loss as SCHM/B0, so the same Train-only calibration legitimately freezes `lambda_schm=1.0`.

## Routing and gradient proof

- Gradient of native O2M loss into RoadSnake gamma: `0.370757`, finite and non-zero.
- With gamma forced to 0.05 for an audit probe, native O2O P4 error remained exactly `0.0`.
- The same probe changed the RoadSnake O2M P4 path by mean absolute `0.018734`.
- SCHM gradient into the O2M selection prediction remained exactly zero.
- Real-batch legality: 7 new-index harvests, 0 conflicts, 0 illegal harvests, and no missing references in the audited batch.

These measurements jointly show that RoadSnake can alter and learn from the native O2M trajectory without rewriting the O2O feature path or opening a gradient through discrete candidate selection.

## Deployment contract

After formal training, the runner must:

1. save the original `best.pt`;
2. physically replace `RoadSnakeO2MDetect` with native `Detect` and remove all RoadSnake parameter keys;
3. save `best_pruned_native.pt`;
4. check elementwise equality between the pruned model and the native O2O path of the unpruned checkpoint with RoadSnake disabled;
5. run Val-only evaluation, Params/GFLOPs checks, and ONNX export on the pruned artifact.

The expected deployment model is native B0 Detect with no SCHM or RoadSnake in the inference graph. A static profiler reported 5.6267 GFLOPs versus its B0 reference 5.7782 GFLOPs despite exact output equivalence; that profiler discrepancy is not interpreted as a speed gain. The decisive deployment-cost evidence will be the formal post-pruning audit.

## Implementation surface

- `ultralytics/nn/roadsnake.py` (`RoadSnakeO2MDetect`)
- `ultralytics/utils/schm_loss.py`
- `ultralytics/models/yolo/detect/schm_train.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/cfg/models/26/yolo26n-japan4-rs-schm.yaml`
- `configs/experiments/japan4_cleanv3_rs_schm_100e.yaml`
- `scripts/prune_roadsnake_t1.py`
- `scripts/verify_japan4_schm_family.py`
- `scripts/train_schm_family.py`
- `scripts/run_rs_schm_cleanv3_100e.sh`

Validated source commit: `566e80d89007dd10a0e705c7d3c1b1cd525187e8`.
