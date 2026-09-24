#!/bin/bash
# E086 최소 회로 — 배선이 도는지 먼저 확인(짧게).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
f=/root/minc_run/smoke.log
timeout 1800 python minimal_circuit.py --seed 0 --trials 100 --block 25 --mode learn > "$f" 2>&1
rc=$?
if grep -q "MINCIRC" "$f"; then
  grep -E "시행|kc_out_|첫 구간|MINCIRC" "$f"
else
  echo "[실패 rc=$rc]"; tail -12 "$f"
fi
