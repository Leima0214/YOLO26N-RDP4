#!/usr/bin/env bash
set -euo pipefail

cd /root/YOLO26N-RDP4

name="YOLO12N_TAL_SAMEGT_RANK_30E_seed42"
log="logs/${name}.log"
pid_file="logs/${name}.pid"
run_dir="runs/paper1_svrdd7_yolo12_ranking/${name}"
meta_dir="runtime_meta/${name}"

if [[ -e "${run_dir}" || -e "${meta_dir}" ]]; then
  echo "Fresh run required; existing output found for ${name}" >&2
  exit 2
fi

mkdir -p logs
nohup /opt/conda/bin/python scripts/train_svrdd7_yolo12n_same_gt_ranking.py \
  --name "${name}" \
  >"${log}" 2>&1 </dev/null &
pid=$!
echo "${pid}" >"${pid_file}"
sleep 3
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Training process exited during launch; inspect ${log}" >&2
  tail -80 "${log}" >&2
  exit 1
fi

echo "RUN_NAME=${name}"
echo "PID=${pid}"
echo "LOG=/root/YOLO26N-RDP4/${log}"
