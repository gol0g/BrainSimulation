#!/bin/bash
# E139: E119 반사 0 학습 재실행(뇌 10~14, 같은 인자) + 읽기 전용 KC 계층 추적(E138 수리 분류). 재개 가능(P13).
# 요약 줄: "  e139 b10: => 사전 +0.0195 사후 -0.0838 || => KCTRACE ... || => KCTRACE2 ..."  (judge_e139.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E139.log"
RAW="$R/research/experiments/logs/E139"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E139"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e139_main_run && cd /root/e139_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for B in 10 11 12 13 14; do
  tag="e139 b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed $B \
    --save-weights $WD/w_b$B.npz --trace-kc-class $WD/tr_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCTRACE " "$f" && grep -q "^\[사후\]" "$f"; then
    echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') || $(grep '^=> KCTRACE ' "$f") || $(grep '^=> KCTRACE2' "$f")"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E139] 전체 루프 종료"
