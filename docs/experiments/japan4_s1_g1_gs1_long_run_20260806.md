# Japan4 S1/G1/GS1 long-run manual — 2026-08-06

This manual is fixed to Japan4-cleanV3 train/Val, one RTX 4090, and serial execution. It never reads Test. The original checkout was audited as dirty, so all new runs use the clean worktree `/root/YOLO26N-RDP4-20260806`; existing B0-S30 and S1-S30 artifacts remain under `/root/YOLO26N-RDP4` and are read-only.

## 1. Connect and audit without changing state

From PowerShell:

```powershell
ssh -i "C:\Users\conor\.ssh\id_rsa" -p 42084 root@xj-member.bitahub.com
```

On the server:

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4

echo "===== GIT ====="
pwd
git branch --show-current
git remote -v
git status --short
git log --oneline --decorate -10

echo "===== SYSTEM ====="
date
hostname
uname -a
df -h
free -h
nproc
nvidia-smi

echo "===== PROCESS ====="
pgrep -af "python|train" || true
tmux ls || true
```

Stop if an unknown training process owns the GPU, free disk is under 15 GB, the project path is wrong, or the data mount is missing. Do not reset, clean, delete runs, or overwrite weights.

## 2. Preserve the dirty checkout and create the clean run worktree

The audited original checkout has tracked changes. Preserve them and avoid pulling into it:

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4
git diff --stat
git diff > /root/YOLO26N-RDP4_remote_before_sync_20260806.patch
git status --short > /root/YOLO26N-RDP4_remote_before_sync_20260806.status.txt
git fetch origin codex/japan4-road-modules --prune

test ! -e /root/YOLO26N-RDP4-20260806
git worktree add -b codex/japan4-road-modules-run-20260806 \
  /root/YOLO26N-RDP4-20260806 origin/codex/japan4-road-modules

cd /root/YOLO26N-RDP4-20260806
git lfs pull
git rev-parse HEAD
git branch --show-current
git status --short
mkdir -p logs reports runtime_meta
```

`git status --short` must be empty at this point. `git rev-parse HEAD` must equal the commit SHA supplied with this manual.

## 3. Environment check

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806
which python
which /opt/conda/bin/python
/opt/conda/bin/python -V

/opt/conda/bin/python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cudnn:", torch.backends.cudnn.version())
assert torch.cuda.is_available()
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

/opt/conda/bin/python - <<'PY'
mods = ["torch", "torchvision", "yaml", "numpy", "pandas", "cv2", "matplotlib", "scipy", "psutil", "timm", "einops", "pycocotools", "onnx"]
failed = []
for name in mods:
    try:
        mod = __import__(name)
        print("[OK]", name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        failed.append(name)
        print("[FAIL]", name, type(exc).__name__, exc)
assert not failed, failed
from ultralytics import YOLO
print("[OK] local ultralytics", YOLO)
PY

pip_check_output="$(/opt/conda/bin/python -m pip check 2>&1 || true)"
printf '%s\n' "$pip_check_output"
case "$pip_check_output" in
  "No broken requirements found."|"ninja 1.11.1.1 is not supported on this platform") ;;
  *) exit 1 ;;
esac
```

The server audit found and repaired an incomplete pandas 2.3.3 install and missing pycocotools with the exact command below; it deliberately leaves Torch, CUDA, NumPy, and Ultralytics untouched:

```bash
/opt/conda/bin/python -m pip install --no-deps --force-reinstall pandas==2.3.3 pycocotools==2.0.10
```

The remaining `ninja 1.11.1.1 is not supported on this platform` report is a pre-existing optional compile-tool metadata warning; training is locked to `compile=False`. Any different `pip check` failure is a stop condition. Do not upgrade Ultralytics, Torch, Torchvision, CUDA, or NumPy. If another named import alone is missing, inspect `requirements-roadlite26.txt`, `requirements.txt`, and `pyproject.toml`, install only that package, record the command in `runtime_meta/dependency_install_20260806.txt`, and rerun the checks.

## 4. Data audit

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806
sed -n '1,200p' configs/japan4_clean_v3_remote.yaml
test -d /Japan4-V3/Japan4-cleanV3/images/train
test -d /Japan4-V3/Japan4-cleanV3/images/val
test -d /Japan4-V3/Japan4-cleanV3/labels/train
test -d /Japan4-V3/Japan4-cleanV3/labels/val

/opt/conda/bin/python -u scripts/audit_japan4_dataset.py \
  --data configs/japan4_clean_v3_remote.yaml \
  --output reports/japan4_clean_v3_audit_20260806.json \
  2>&1 | tee logs/japan4_clean_v3_audit_20260806.log
```

Proceed only if the report confirms classes 0/1/2/3, valid normalized boxes, paired images and labels, no corrupt images, no train/Val overlap, and no group leakage. Do not inspect the Test split.

## 5. Focused tests and 4090 static verification

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806

/opt/conda/bin/python -m py_compile \
  ultralytics/utils/region_loss.py \
  scripts/audit_japan4_dataset.py \
  scripts/train_japan4_candidate.py \
  scripts/verify_japan4_candidate.py \
  scripts/eval_japan4_candidate.py \
  scripts/compare_japan4_runs.py \
  scripts/diagnose_s1_gates.py \
  scripts/diagnose_g1_regions.py \
  tests/test_japan4_s1_g1_gs1.py

/opt/conda/bin/python -m pytest tests/test_japan4_s1_g1_gs1.py -q \
  2>&1 | tee logs/japan4_s1_g1_gs1_tests_20260806.log

/opt/conda/bin/python -u scripts/verify_japan4_candidate.py \
  --candidate s1 --candidate g1 --candidate gs1 \
  --weights yolo26n.pt \
  --device cuda:0 \
  --imgsz 640 \
  --onnx-dir reports/japan4_s1_g1_gs1_onnx_4090_20260806 \
  --report reports/japan4_s1_g1_gs1_static_4090_20260806.json \
  2>&1 | tee logs/japan4_s1_g1_gs1_static_4090_20260806.log
```

All candidates require exact step-0 detection equivalence, finite loss/gradients, complete shared-weight transfer, valid ONNX, and no region heads after fusion. S1 must remain below +5% parameters, GFLOPs, and latency; G1 deployment must equal B0; GS1 deployment must equal S1.

## 6. One-batch G1 gradient calibration

Run once before any G1/GS1 training. This does not train or read Val AP:

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806
test ! -e reports/g1_gradient_calibration_4090_20260806
/opt/conda/bin/python -u scripts/diagnose_g1_regions.py \
  --candidate g1 \
  --data configs/japan4_clean_v3_remote.yaml \
  --weights yolo26n.pt \
  --device 0 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --output reports/g1_gradient_calibration_4090_20260806 \
  2>&1 | tee logs/g1_gradient_calibration_4090_20260806.log

sed -n '1,240p' reports/g1_gradient_calibration_4090_20260806/g1_gradient_calibration.json
```

The accepted ratio is 0.05–0.15. The JSON recommendation targets the smallest accepted value, 0.05. If the committed P3/P4 weights differ from the recommendation, stop before training, apply the same calibrated weights to both G1 and GS1 YAMLs, commit/push them, then rebuild this worktree at that commit and rerun static verification. Never choose weights from Val AP.

## 7. S1 1E smoke

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806

/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate s1 \
  --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 1 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name smoke_S1_japan4_cleanv3_e1_img640_b32_seed42_20260806 \
  2>&1 | tee logs/smoke_S1_japan4_cleanv3_e1_img640_b32_seed42_20260806.log

test -f runs/paper1_japan4_clean/smoke_S1_japan4_cleanv3_e1_img640_b32_seed42_20260806/results.csv
test -f runs/paper1_japan4_clean/smoke_S1_japan4_cleanv3_e1_img640_b32_seed42_20260806/weights/last.pt
grep -nEi "Traceback|ERROR|Exception|RuntimeError|Killed|CUDA out of memory|NaN|Inf" \
  logs/smoke_S1_japan4_cleanv3_e1_img640_b32_seed42_20260806.log || true
```

The smoke run is engineering evidence only. It must finish training, validation, checkpoint writing, and exit with code 0.

## 8. S1 formal 100E — first formal run

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806
test ! -e runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806
test ! -e runtime_meta/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806
! pgrep -af "python.*train_japan4_candidate" >/dev/null

tmux new-session -d -s s1_100e_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && \
set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate s1 \
  --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806 \
2>&1 | tee logs/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806.log; \
status=\${PIPESTATUS[0]}; \
echo \${status} > runtime_meta/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806_wrapper_exit_code.txt; \
exit \${status}"

tmux ls
pgrep -af "python.*train_japan4_candidate"
nvidia-smi
tail -100 logs/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806.log
```

## 9. S1 monitoring and recovery

Use this at five minutes and epochs 1, 10, 30, 50, 75, and 100:

```bash
cd /root/YOLO26N-RDP4-20260806
tmux capture-pane -pt s1_100e_20260806 -S -120 || true
tail -120 logs/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806.log
nvidia-smi
pgrep -af "python.*train" || true
df -h
tail -n 10 runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/results.csv
ls -lh runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights
```

Do not stop for ordinary metric fluctuation. Stop only for persistent NaN/Inf, invalid labels, missing data, confirmed OOM without duplicate processes, killed process, near-full disk, checkpoint failure, or wrong data/model configuration.

If an infrastructure interruption leaves a valid `last.pt`, resume the same 100E schedule and run directory:

```bash
cd /root/YOLO26N-RDP4-20260806
tmux new-session -d -s s1_100e_resume_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && \
set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate s1 \
  --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 100 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806 \
  --resume-from /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/last.pt \
2>&1 | tee -a logs/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806.log; \
exit \${PIPESTATUS[0]}"
```

## 10. S1 post-processing

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806
cat runtime_meta/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/exit_code.txt
test "$(cat runtime_meta/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/exit_code.txt)" = "0"
test -f runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/best.pt
test -f runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/last.pt

/opt/conda/bin/python -u scripts/eval_japan4_candidate.py \
  --checkpoint S1-100E=/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/best.pt \
  --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --output /root/YOLO26N-RDP4-20260806/reports/eval_S1_100e_20260806 \
  --imgsz 640 --batch 32 --workers 8 --device 0 \
  2>&1 | tee logs/eval_S1_100e_20260806.log

/opt/conda/bin/python -u scripts/diagnose_s1_gates.py \
  --weights /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/best.pt \
  --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --device 0 --imgsz 640 --batch 32 --workers 8 --max-batches 25 \
  --output reports/s1_100e_gate_diagnosis_20260806.json \
  2>&1 | tee logs/s1_100e_gate_diagnosis_20260806.log
```

The current machine does not contain the exact B0-V3 100E checkpoint. Confirm that fact and retain the frozen reference hash rather than substituting an older Positive-v2 model:

```bash
find /root -type f -name best.pt -print0 2>/dev/null | xargs -0 -r sha256sum | \
  grep 76fd04c527d212e3f7e0b747c33ff65c463e535bf555e46376b1bbf8850be656 || true
```

Until the exact artifact is recovered, use only its ledger headline values; do not invent per-class B0-100E results.

## 11. G1 1E, 30E, and conditional 100E

Run only after S1-100E is finished and archived.

```bash
cd /root/YOLO26N-RDP4-20260806
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate g1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 1 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name smoke_G1_japan4_cleanv3_e1_img640_b32_seed42_20260806 \
  2>&1 | tee logs/smoke_G1_japan4_cleanv3_e1_img640_b32_seed42_20260806.log

tmux new-session -d -s g1_30e_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate g1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 30 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806 \
2>&1 | tee logs/pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806.log; \
exit \${PIPESTATUS[0]}"
```

If and only if G1-30E passes its gate, start a fresh official-weight 100E run:

```bash
cd /root/YOLO26N-RDP4-20260806
tmux new-session -d -s g1_100e_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate g1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 100 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name formal_G1_japan4_cleanv3_e100_img640_b32_seed42_20260806 \
2>&1 | tee logs/formal_G1_japan4_cleanv3_e100_img640_b32_seed42_20260806.log; \
exit \${PIPESTATUS[0]}"
```

G1 recovery uses the original total schedule:

```bash
cd /root/YOLO26N-RDP4-20260806
tmux new-session -d -s g1_30e_resume_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate g1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 30 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806 \
  --resume-from /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806/weights/last.pt \
2>&1 | tee -a logs/pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806.log; \
exit \${PIPESTATUS[0]}"
```

For a formal G1 interruption, use the same command with `epochs 100`, formal name, formal log, and `/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/formal_G1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/last.pt`.

## 12. GS1 1E, 30E, and conditional 100E

Run GS1 only if G1 passes or is AP-neutral with a clear D10 improvement.

```bash
cd /root/YOLO26N-RDP4-20260806
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate gs1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 1 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name smoke_GS1_japan4_cleanv3_e1_img640_b32_seed42_20260806 \
  2>&1 | tee logs/smoke_GS1_japan4_cleanv3_e1_img640_b32_seed42_20260806.log

tmux new-session -d -s gs1_30e_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate gs1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 30 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806 \
2>&1 | tee logs/pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806.log; \
exit \${PIPESTATUS[0]}"
```

If and only if GS1-30E beats the best single module sufficiently, run fresh 100E:

```bash
cd /root/YOLO26N-RDP4-20260806
tmux new-session -d -s gs1_100e_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate gs1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 100 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name formal_GS1_japan4_cleanv3_e100_img640_b32_seed42_20260806 \
2>&1 | tee logs/formal_GS1_japan4_cleanv3_e100_img640_b32_seed42_20260806.log; \
exit \${PIPESTATUS[0]}"
```

GS1 30E recovery:

```bash
cd /root/YOLO26N-RDP4-20260806
tmux new-session -d -s gs1_30e_resume_20260806 \
"cd /root/YOLO26N-RDP4-20260806 && set -o pipefail && \
/opt/conda/bin/python -u scripts/train_japan4_candidate.py \
  --candidate gs1 --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --epochs 30 --imgsz 640 --batch 32 --workers 8 --device 0 --seed 42 \
  --project /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean \
  --name pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806 \
  --resume-from /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806/weights/last.pt \
2>&1 | tee -a logs/pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806.log; \
exit \${PIPESTATUS[0]}"
```

For a formal GS1 interruption, use `epochs 100`, the formal name/log, and `/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/formal_GS1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/last.pt`.

## 13. Val-only evaluation, diagnostics, and comparison

After the 30E candidates that actually pass are complete, evaluate existing B0/S1 controls and the new checkpoints together:

```bash
cd /root/YOLO26N-RDP4-20260806
/opt/conda/bin/python -u scripts/eval_japan4_candidate.py \
  --checkpoint B0-S30=/root/YOLO26N-RDP4/runs/paper1_japan4_clean/b0-v3_japan4_clean_30e_seed42/weights/best.pt \
  --checkpoint S1-S30=/root/YOLO26N-RDP4/runs/paper1_japan4_clean/yolo26n-japan4-s1-strip-regression_cleanv3_30e_seed42/weights/best.pt \
  --checkpoint G1-S30=/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806/weights/best.pt \
  --checkpoint GS1-S30=/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806/weights/best.pt \
  --data /root/YOLO26N-RDP4-20260806/configs/japan4_clean_v3_remote.yaml \
  --output /root/YOLO26N-RDP4-20260806/reports/eval_30e_s1_g1_gs1_20260806 \
  --imgsz 640 --batch 32 --workers 8 --device 0 \
  2>&1 | tee logs/eval_30e_s1_g1_gs1_20260806.log

/opt/conda/bin/python -u scripts/compare_japan4_runs.py \
  --run B0-S30=/root/YOLO26N-RDP4/runs/paper1_japan4_clean/b0-v3_japan4_clean_30e_seed42 \
  --run S1-S30=/root/YOLO26N-RDP4/runs/paper1_japan4_clean/yolo26n-japan4-s1-strip-regression_cleanv3_30e_seed42 \
  --run G1-S30=/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806 \
  --run GS1-S30=/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806 \
  --eval B0-S30=/root/YOLO26N-RDP4-20260806/reports/eval_30e_s1_g1_gs1_20260806 \
  --eval S1-S30=/root/YOLO26N-RDP4-20260806/reports/eval_30e_s1_g1_gs1_20260806 \
  --eval G1-S30=/root/YOLO26N-RDP4-20260806/reports/eval_30e_s1_g1_gs1_20260806 \
  --eval GS1-S30=/root/YOLO26N-RDP4-20260806/reports/eval_30e_s1_g1_gs1_20260806 \
  --output /root/YOLO26N-RDP4-20260806/reports/compare_30e_s1_g1_gs1_20260806 \
  2>&1 | tee logs/compare_30e_s1_g1_gs1_20260806.log
```

If GS1 is not authorized, omit its checkpoint/run/eval arguments rather than creating a fake result.

G1/GS1 region response and GS1 gate diagnostics:

```bash
cd /root/YOLO26N-RDP4-20260806
/opt/conda/bin/python -u scripts/diagnose_g1_regions.py \
  --candidate g1 --data configs/japan4_clean_v3_remote.yaml \
  --weights runs/paper1_japan4_clean/pilot_G1_japan4_cleanv3_e30_img640_b32_seed42_20260806/weights/best.pt \
  --device 0 --imgsz 640 --batch 32 --workers 8 \
  --output reports/g1_30e_region_diagnosis_20260806

/opt/conda/bin/python -u scripts/diagnose_s1_gates.py \
  --weights runs/paper1_japan4_clean/pilot_GS1_japan4_cleanv3_e30_img640_b32_seed42_20260806/weights/best.pt \
  --data configs/japan4_clean_v3_remote.yaml \
  --device 0 --imgsz 640 --batch 32 --workers 8 --max-batches 25 \
  --output reports/gs1_30e_gate_diagnosis_20260806.json
```

## 14. Go/No-Go gates

| Stage | Go | No-Go |
|---|---|---|
| S1 static | zero step-0 error; finite gradients; Params/GFLOPs/latency each under +5% | any structure mismatch, nonfinite value, or threshold failure |
| S1-100E success | versus B0-V3: AP +0.005; or AP +0.003–0.005 with clear D10/D40 gain | record failure honestly; still run G1-30E, but do not promote GS1 blindly |
| G1-30E route A | AP versus B0-S30 at least +0.003 | AP below B0 by over 0.005, AR100 below by over 0.020, no D10 benefit, or degenerate maps |
| G1-30E route B | AP loss at most 0.002, D10 AP +0.010, Precision +0.010, AR100 loss at most 0.010 | failure of the joint condition |
| GS1 authorization | G1 passes, or is AP-neutral with clear D10 background benefit | clearly failed G1 |
| GS1-100E route A | AP versus B0-S30 at least +0.005 and preferably +0.002 over best single module | merely equals S1 with no extra benefit |
| GS1-100E route B | AP75 +0.010, D10 or D40 improves, D20 loss at most 0.008, AR100 loss at most 0.010 | failure of the joint condition |

Only Val decides. AP-small/AP-medium/AP-large, AR100, AP75, and every class remain mandatory context even when total AP rises.

## 15. Backup and hashes

Run after all authorized jobs finish:

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806
mkdir -p /root/japan4_20260806_archives

tar -czf /root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_full.tar.gz \
  logs reports runtime_meta runs/paper1_japan4_clean docs/experiments

tar -czf /root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_reports.tar.gz \
  reports docs/experiments

tar -czf /root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_logs_meta.tar.gz \
  logs runtime_meta

tar -czf /root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_light.tar.gz \
  --exclude='*.pt' --exclude='*.onnx' \
  logs reports runtime_meta runs/paper1_japan4_clean docs/experiments

sha256sum /root/japan4_20260806_archives/*.tar.gz | \
  tee /root/japan4_20260806_archives/SHA256SUMS.txt
ls -lh /root/japan4_20260806_archives
```

## 16. PowerShell downloads

```powershell
scp -i "C:\Users\conor\.ssh\id_rsa" -P 42084 root@xj-member.bitahub.com:/root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean/formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806/weights/best.pt .\formal_S1_japan4_cleanv3_e100_img640_b32_seed42_20260806_best.pt
scp -i "C:\Users\conor\.ssh\id_rsa" -P 42084 root@xj-member.bitahub.com:/root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_full.tar.gz .
scp -i "C:\Users\conor\.ssh\id_rsa" -P 42084 root@xj-member.bitahub.com:/root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_reports.tar.gz .
scp -i "C:\Users\conor\.ssh\id_rsa" -P 42084 root@xj-member.bitahub.com:/root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_logs_meta.tar.gz .
scp -i "C:\Users\conor\.ssh\id_rsa" -P 42084 root@xj-member.bitahub.com:/root/japan4_20260806_archives/japan4_s1_g1_gs1_20260806_light.tar.gz .
scp -i "C:\Users\conor\.ssh\id_rsa" -P 42084 root@xj-member.bitahub.com:/root/japan4_20260806_archives/SHA256SUMS.txt .
```

Download G1/GS1 best checkpoints only if those formal runs were authorized and completed.

## 17. GPU shutdown audit

```bash
set -euo pipefail
cd /root/YOLO26N-RDP4-20260806
pgrep -af "python.*train" || true
tmux ls || true
nvidia-smi
find /root/YOLO26N-RDP4-20260806/runs/paper1_japan4_clean -type f \( \
  -name "best.pt" -o -name "last.pt" -o -name "results.csv" -o -name "args.yaml" \
\) -ls
find /root/YOLO26N-RDP4-20260806/logs -type f -size +0 -ls
ls -lh /root/japan4_20260806_archives
sha256sum -c /root/japan4_20260806_archives/SHA256SUMS.txt
git status --short
```

Do not shut down until no trainer exists, all authorized runs have checkpoints/results/logs/exit codes, all archives pass SHA256, and the important files have been downloaded and locally checked.
