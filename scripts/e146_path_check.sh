#!/bin/bash
# E146 경로 검사(조건 2) — 기준 고정(logs/E146/criteria_fixed.txt 15:25:38) 뒤. 표본 밖 뇌 15, 학습 없음.
# 가중치 none·E141 경로 검사 배율 0(w_s0_b15, 판정 v 로 학습: 사후 −0.2103, 사전 +0.0255) × 자극 5종 + noise 재실행(결정성).
# 확인: base 가 원 사전·사후를 재현, 변형이 실제로 다른 값을 내는지(0/nan 아님), noise 가 같은 시드에서 같은 값.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E146/pathcheck"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e146_run && cd /root/e146_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --reflex-w 0 --rw-apm-scale 0"
WS="$R/research/experiments/traces/E141/pathcheck/w_s0_b15.npz"
for W in none E141; do
  [ "$W" = "none" ] && M="none" || M="all"
  for S in base int05 int07 occ noise noise; do
    f="$OUT/b15_${W}_$S.log"; [ -e "$f" ] && f="$OUT/b15_${W}_${S}_2.log"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --decomp-weights $WS --decomp-mode $M --eval-variant $S --eval-vseed 0 > "$f" 2>&1; rc=$?
    echo "[$W $S] $(grep '^\[E146 변형\]' "$f") $(grep '^=> DECOMP' "$f" | cut -c1-70)"
    [ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
    [ "$S" = "noise" ] && [ "$W" = "E141" ] || true
  done
done
echo "[E146 경로 검사] 종료"
