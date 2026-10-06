#!/bin/bash
# E149 경로 검사(조건 2) — 기준 고정(logs/E149/criteria_fixed.txt 21:27:49) 뒤. 표본 밖 뇌 15, 학습 없음, kcoverlap 2조건.
# 확인: 입력 가중치 줄(차단이 food_eye→KC·it_food→KC 에 닿는가, good/bad→KC 는 그대로), KCOVERLAP 줄 파싱 가능, 반분 신뢰도, 스파이크 > 0.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E149/pathcheck"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e149_run && cd /root/e149_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
WS="$R/research/experiments/traces/E141/pathcheck/w_s0_b15.npz"
for C in base block; do
  [ "$C" = "block" ] && X="--kc-food-eye-scale 0 --kc-bilateral-scale 0" || X="--kc-food-eye-scale 1 --kc-bilateral-scale 1"
  f="$OUT/b15_$C.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --decomp-weights $WS --decomp-mode kcoverlap --trials 200 $X > "$f" 2>&1; rc=$?
  echo "[$C] $(grep '^\[E149 입력 가중치\]' "$f" | cut -c1-230)"
  echo "[$C] $(grep '^=> KCOVERLAP' "$f" | cut -c1-400)"
  [ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
done
echo "[E149 경로 검사] 종료"
