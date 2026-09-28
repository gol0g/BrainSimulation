#!/bin/bash
# E118: E110 조건 + 지속 전류 행동 창(act-window 3, act-current 5000, 초기 150) — 학습(저장)/무학습 × 뇌 0~4 → 학습 가중치 neuron 분석. 재개 가능.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E118.log"
RAW="$R/research/experiments/logs/E118"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E118"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e118_run && cd /root/e118_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000"
for B in 0 1 2 3 4; do for C in 학습 무학습; do
  tag="$C b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/${C}_b$B.log"; EXTRA="--save-weights $WD/w_b$B.npz"; [ "$C" = "무학습" ] && EXTRA="--no-reward"
  printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 5 --steps 100 --transplant-eval --brain-seed $B $EXTRA > "$f" 2>&1
  if grep -q "변조폭 변화" "$f"; then echo "=> $(grep -E '\[사전\]|\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/' | tr '\n' ' ')| $(grep '변조폭 변화' "$f" | tail -1 | sed 's/^=> //')"; else echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; fi
done; done
for B in 0 1 2 3 4; do
  tag="neuron b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/neuron_b$B.log"; printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE --episodes 0 --brain-seed $B --decomp-weights "$WD/w_b$B.npz" --decomp-mode neuron > "$f" 2>&1
  if grep -q "^=> KCPRE" "$f"; then echo "=> $(grep -E '^=> KCPRE' "$f" | sed 's/^=> //' | tr '\n' ' ')"; else echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E118] 전체 루프 종료"
