#!/bin/bash
# E156 경로 검사(조건 2) — 기준 고정(logs/E156/criteria_fixed.txt 00:36:13) 뒤. 표본 밖 뇌 15, 반사 25 + 동결 + 형성 가중치(E153 경로 검사 kctype_b15_eta01.npz), 200시행.
# 확인: 적재 줄(학습·이식 평가 뇌), 반사 25→25, 동결 잔차·되돌림, 추적, [사전] 출력 경로.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E156/pathcheck"; WD="$R/research/experiments/traces/E156/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e156_run && cd /root/e156_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
f="$OUT/F200_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes 2 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale 0 \
  --kc-type-weights $R/research/experiments/traces/E153/pathcheck/kctype_b15_eta01.npz \
  --save-weights $WD/w_F200_b15.npz --trace-kc-class $WD/tr_F200_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
echo "[F200 rc=$rc] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 반사 $(grep '^\[반사가중치\] good_food_to_motor' "$f" | grep -oE 'w_mean \S+' | tr '\n' ' ')"
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e156 as J
s = J.stats(np.load("research/experiments/traces/E156/pathcheck/tr_F200_b15.npz")["rows"])
print("  추적: 시행 %d 동결 잔차 %.2e 되돌림 %.3f 도파민전 %.1e B/A %+.2f C/P %+.2f" % (s["n"], s["res"], s["alive"], s["pre_ratio"], s["BA"], s["CP"]))
PY
echo "[E156 경로 검사] 종료"
