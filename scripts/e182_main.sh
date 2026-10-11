#!/bin/bash
# E182 본실험 — 기준 logs/E182/criteria_fixed.txt. E179 가중치(대칭 맥락 쌍조건, traces/E179/w_bc_b{B}.npz)를 학습과 같은 시간 구조로 재평가(평가만, 뇌 10~14):
# --eval-ctx-trial(시행마다 조향 처리 동안만 맥락 켬, 사이 10처리 끔·무자극, 안정화·오프셋 맥락 없음) + 기저값 −4000 + --eval-diag. 학습·무학습 × 맥락 A·B = 20 평가.
# 요약 줄 "  e182 b10 learn A: => mod -0.1000" 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E182.log"
RAW="$R/research/experiments/logs/E182"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E179"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e182_main_run && cd /root/e182_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E182] 시행 맞춤 맥락 평가 · 기저값 −4000 · E179 가중치"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  CTX="--conn-snapshot $SNAP/conn_b$B.npz --ctx-n 200 --ctx-w 4 --ctx-p 0.10 --ctx2-n 200 --ctx2-w 4 --ctx2-p 0.10 --motor-tonic -4000"
  for WC in "learn A" "learn B" "none A" "none B"; do
    set -- $WC; M=$1; C=$2
    tag="e182 b$B $M $C"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    if [ ! -s "$WD/w_bc_b$B.npz" ]; then echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; fi
    [ "$M" = "learn" ] && XM="all" || XM="none"
    [ "$C" = "B" ] && XE="--eval-ctx" || XE="--eval-ctx2"
    g="$RAW/ev_b${B}_${M}_$C.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $CTX --decomp-weights $WD/w_bc_b$B.npz --decomp-mode $XM $XE --eval-ctx-trial --eval-diag > "$g" 2>&1; rc=$?
    if grep -q "^=> DECOMP" "$g"; then echo "=> mod $(grep '^=> DECOMP' "$g" | sed -E 's/.*mod=([-+0-9.]+).*/\1/')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E182] 전체 루프 종료"
