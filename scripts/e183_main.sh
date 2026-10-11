#!/bin/bash
# E183 본실험 — 기준 logs/E183/criteria_fixed.txt. 뇌 10~14 × 팔 O(노출 eta 0.02)·Z(eta 0) rawcat 전체 절차(pre·pre2 → 노출 408 → post).
# feat_w = 보정 선택(logs/E183/pick.txt 'W=<값>'). 요약 줄 "  e183 b10 O: => RAWCAT ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E183.log"
RAW="$R/research/experiments/logs/E183"; mkdir -p "$RAW"
SNAP="$R/research/experiments/traces/E176/snap"
P=$(cat "$RAW/pick.txt" 2>/dev/null)
case "$P" in W=[0-9]*) W=${P#W=} ;; *) echo "[E183] 보정 선택 없음($P) — 본실험 안 함"; exit 1 ;; esac
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e183_main_run && cd /root/e183_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E183] 원시 특징 범주 형성 측정 · feat_w $W · 뇌 10~14 × O·Z"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  WB="$R/research/experiments/traces/E179/w_bc_b$B.npz"
  for A in O Z; do
    tag="e183 b$B $A"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    [ "$A" = "O" ] && EE=0.02 || EE=0
    g="$RAW/rc_b${B}_$A.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW --conn-snapshot $SNAP/conn_b$B.npz \
      --feat-n 200 --feat-w $W --feat-p 0.10 --feat-oja --decomp-weights $WB --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta $EE > "$g" 2>&1; rc=$?
    if grep -q "^=> RAWCAT " "$g"; then echo "=> $(grep '^=> RAWCAT ' "$g" | cut -c1-60) RCKC $(grep -c '^RCKC ' "$g")"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E183] 전체 루프 종료"
