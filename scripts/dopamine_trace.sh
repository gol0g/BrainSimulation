#!/bin/bash
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/datrace && cd /root/datrace
cp $R/backend/genesis/*.py . 2>/dev/null
f=/root/datrace/trace.log
timeout 2400 python dopamine_trace_probe.py --trials 200 --seed 0 --d1-inhib -400 --direct-inhib -100 > "$f" 2>&1
rc=$?
if grep -q "DATRACE" "$f"; then
  sed -n '/=== 도파민 궤적/,$p' "$f"
else
  echo "[실패 rc=$rc]"; tail -6 "$f"
fi
