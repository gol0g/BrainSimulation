#!/bin/bash
# E111: KC→motor 자격흔적 부호 추적 — B25/B8 × 뇌 0·1·2. 재개 가능.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E111.log"
RAW="$R/research/experiments/logs/E111"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e111_run && cd /root/e111_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 30 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --episodes 5 --steps 100 --transplant-eval --env-seed 0"
for BI in 25 8; do for B in 0 1 2; do
  tag="B$BI b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/B${BI}_b$B.log"; c="$RAW/B${BI}_b$B.csv.log"
  printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE --bias $BI --brain-seed $B --trace-kc-motor "$c" > "$f" 2>&1
  if grep -q "변조폭 변화" "$f"; then echo "=> $(grep '변조폭 변화' "$f" | tail -1) | $(python $R/scripts/analyze_e111.py "$c")"; else echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; fi
done; done
echo "[E111] 전체 루프 종료"
