#!/bin/bash
# E188 본실험 — 기준 logs/E188/criteria_fixed.txt. 원시 특징 범주 규칙 반전: 뇌 10~14 × catrule 3,000(1,500 부터 반전) → 평가 학습·무학습.
# 요약 줄 "  e188 ... : => ..." 재개 가능(P13). 15줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E188.log"
RAW="$R/research/experiments/logs/E188"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E188"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e188_main_run && cd /root/e188_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E188] 원시 특징 범주 규칙 반전 · catrule 1,500 → 반전 1,500 · 뇌 10~14"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  tag="e188 train b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    f="$RAW/train_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule --rc-reverse-after 1500 --episodes 30 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^\[범주 반전\] 반전 시행" "$f" && grep -q "^\[학습\]" "$f" && [ -s "$WD/w_b$B.npz" ]; then
      echo "=> $(grep '^\[범주 반전\] 반전 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1)"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  fi
  for W in learn none; do
    tag="e188 b$B $W"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    [ -s "$WD/w_b$B.npz" ] || { echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; }
    [ "$W" = "learn" ] && XM=all || XM=none
    g="$RAW/ev_b${B}_$W.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed $B --kc-type-weights $KW --decomp-weights $WD/w_b$B.npz --decomp-mode $XM --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
    if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E188] 전체 루프 종료"
