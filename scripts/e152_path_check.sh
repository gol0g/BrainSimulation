#!/bin/bash
# E152 경로 검사(조건 2) — 기준 고정(logs/E152/criteria_fixed.txt 18:59:45) 뒤. 표본 밖 뇌 15, 기본 상태, 학습 없음, kcoverlap3 1회(--trials 300).
# 확인: 새 모드가 돌고 KCOVERLAP3 줄이 파싱되는가, 먹이 단독 반응 > 0·반분 신뢰도, J(G,B) 가 E149 경로 검사 뇌 15(0.4225/0.4014)와 가까운가, 스파이크 > 0.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E152/pathcheck"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e152_run && cd /root/e152_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
WS="$R/research/experiments/traces/E141/pathcheck/w_s0_b15.npz"
f="$OUT/b15_base.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --decomp-weights $WS --decomp-mode kcoverlap3 --trials 300 --kc-food-eye-scale 1 --kc-bilateral-scale 1 > "$f" 2>&1; rc=$?
echo "[base] $(grep '^=> KCOVERLAP3' "$f" | cut -c1-420)"
[ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -5 "$f"; }
echo "[E152 경로 검사] 종료"
