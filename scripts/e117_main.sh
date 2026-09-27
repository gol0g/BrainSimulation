#!/bin/bash
# E117: KC 반응 집합별 Δg 분해(측정) — E115 가중치, 뇌 0~4.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E117.log"
RAW="$R/research/experiments/logs/E117"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e117_run && cd /root/e117_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 30 --kc-motor-eta 0.15 --tau-e 12 --env-seed 0"
for B in 0 1 2 3 4; do
  tag="kcsets b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/kcsets_b$B.log"; printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE --episodes 0 --brain-seed $B --decomp-weights "$R/research/experiments/traces/E115/w_b$B.npz" --decomp-mode kcsets > "$f" 2>&1
  if grep -q "^=> KCSETS" "$f"; then echo "=> $(grep '^=> KCSETS' "$f" | sed 's/^=> //' | tr '\n' ' ')"; else echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E117] 전체 루프 종료"
