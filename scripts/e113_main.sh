#!/bin/bash
# E113: 새 뇌 5~9 — 1단계 E110 조건 학습+저장 → 2단계 분해 7모드 + neuron 분석. 재개 가능.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E113.log"
RAW="$R/research/experiments/logs/E113"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E113"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e113_run && cd /root/e113_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 30 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
echo "########## 1단계 학습·저장 ##########"
for B in 5 6 7 8 9; do
  tag="train b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/train_b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE --episodes 5 --steps 100 --transplant-eval --brain-seed $B --save-weights "$WD/w_b$B.npz" > "$f" 2>&1
  if grep -q "변조폭 변화" "$f"; then echo "=> $(grep -E '\[사전\]|\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/' | tr '\n' ' ')"; else echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "########## 2단계 분해 ##########"
for B in 5 6 7 8 9; do for M in none all kc_only d1_only kc_shuffle kc_uniform kc_cm neuron; do
  tag="dec b$B $M"
  if grep -qF "$tag: => DECOMP" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/dec_b${B}_$M.log"; printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE --episodes 0 --brain-seed $B --decomp-weights "$WD/w_b$B.npz" --decomp-mode $M > "$f" 2>&1
  if [ "$M" = "neuron" ]; then
    if grep -q "^=> KCPRE" "$f"; then echo "=> DECOMP mode=neuron $(grep -E '^=> (NEURON|KCPRE)' "$f" | sed 's/^=> //' | tr '\n' ' ')"; else echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; fi
  else
    grep -E "^=> DECOMP" "$f" || { echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; }
  fi
done; done
echo "[E113] 전체 루프 종료"
