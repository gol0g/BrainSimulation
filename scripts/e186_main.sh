#!/bin/bash
# E186 본실험 — 기준 logs/E186/criteria_fixed.txt. 원형 효과·전형성 기울기(평가만): 뇌 16~20(E185 w_1500)·10~14(E184 w_U) × 학습·무학습,
# --rc-eval --rc-eval-sets train,new,proto,far --trials 96 --eval-diag. 요약 줄 "  e186 b16 learn: => ..." 재개 가능(P13). 20줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E186.log"
RAW="$R/research/experiments/logs/E186"; mkdir -p "$RAW"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e186_main_run && cd /root/e186_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E186] 원형 효과·전형성 기울기(평가만) · 뇌 16~20(E185) · 10~14(E184 U)"
for B in 16 17 18 19 20 10 11 12 13 14; do
  if [ $B -ge 16 ]; then KW="$R/research/experiments/traces/E161/kctype_oja_b$B.npz"; WF="$R/research/experiments/traces/E185/w_1500_b$B.npz"
  else KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"; WF="$R/research/experiments/traces/E184/w_U_b$B.npz"; fi
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  for W in learn none; do
    tag="e186 b$B $W"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    [ -s "$WF" ] || { echo "  $tag: [실패 rc=가중치 없음]"; continue; }
    [ "$W" = "learn" ] && XM=all || XM=none
    g="$RAW/ev_b${B}_$W.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed $B --kc-type-weights $KW --decomp-weights $WF --decomp-mode $XM --rc-eval --rc-eval-sets train,new,proto,far --trials 96 --eval-diag > "$g" 2>&1; rc=$?
    if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> RCEVAL $(grep -c '^RCEVAL ' "$g")줄"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E186] 전체 루프 종료"
