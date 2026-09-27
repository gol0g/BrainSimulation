#!/bin/bash
# E110: 전체 모델 KC→motor 학습 + 보상 창(자극 끔) — 학습/무학습 × eta 0.15/0.03 × 뇌 0~4. 재개 가능(결과 줄 기준).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E110.log"
RAW="$R/research/experiments/logs/E110"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e109_run && cd /root/e109_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 30 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --episodes 5 --steps 100 --transplant-eval --env-seed 0"
for ETA in 0.15 0.03; do for B in 0 1 2 3 4; do for C in 학습 무학습; do
  tag="eta$ETA $C b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/eta${ETA}_${C}_b$B.log"
  EXTRA=""; [ "$C" = "무학습" ] && EXTRA="--no-reward"
  printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE --kc-motor-eta $ETA --brain-seed $B $EXTRA > "$f" 2>&1
  rc=$?
  if grep -q "변조폭 변화" "$f"; then grep -E "변조폭 변화|=> " "$f" | tail -1 | sed 's/^/=> /'; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done; done; done
echo "[E110] 전체 루프 종료"
