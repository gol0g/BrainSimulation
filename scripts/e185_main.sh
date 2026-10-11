#!/bin/bash
# E185 본실험 — 기준 logs/E185/criteria_fixed.txt. 원시 특징 범주 개념의 독립 확증(뇌 16~20) + 용량-반응(500 대 1,500시행).
# 뇌마다: 학습 500·1,500(catrule, 같은 시드), 평가 학습500·학습1500·무학습(--rc-eval 4조건, 96시행, --eval-diag). 형성 전 U(특징 w 24 정적), E161 종류 가중치, 스냅숏 conn_b16~20.
# 요약 줄 "  e185 ... : => ..." 재개 가능(P13). 25줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E185.log"
RAW="$R/research/experiments/logs/E185"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E185"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e185_main_run && cd /root/e185_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E185] 원시 특징 범주 확증·용량-반응 · 특징 w 24 형성 전 · 뇌 16~20 × 500·1,500시행"
for B in 16 17 18 19 20; do
  KW="$R/research/experiments/traces/E161/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  [ -s "$SNAP/conn_b$B.npz" ] || { echo "  e185 b$B: [실패 rc=스냅숏 없음]"; continue; }
  for T in 500 1500; do
    tag="e185 train b$B $T"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    EP=$((T / 100))
    f="$RAW/train_b${B}_$T.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule --episodes $EP --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_${T}_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^\[범주 과제\] 시행" "$f" && grep -q "^\[학습\]" "$f" && [ -s "$WD/w_${T}_b$B.npz" ]; then
      echo "=> $(grep '^\[범주 과제\] 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1)"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
  for W in learn500 learn1500 none; do
    tag="e185 b$B $W"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    case $W in learn500) WF="$WD/w_500_b$B.npz"; XM=all ;; learn1500) WF="$WD/w_1500_b$B.npz"; XM=all ;; none) WF="$WD/w_1500_b$B.npz"; XM=none ;; esac
    if [ ! -s "$WF" ]; then echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; fi
    g="$RAW/ev_b${B}_$W.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed $B --kc-type-weights $KW --decomp-weights $WF --decomp-mode $XM --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
    if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E185] 전체 루프 종료"
