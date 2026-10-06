#!/bin/bash
# E147 경로 검사(조건 2) — 기준 고정(logs/E147/criteria_fixed.txt 17:23:03) 뒤. 표본 밖 뇌 15, 반사 0·동결, 400시행(200 에서 반전), 추적.
# 확인: "[반전] 시행 200 부터" 줄, 추적의 보상-규칙 일치(앞 200: 정답 ⇔ 실행 ≠ 자극, 뒤 200: 정답 ⇔ 실행 = 자극), 동결 잔차, 반사 0→0.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E147/pathcheck"; WD="$R/research/experiments/traces/E147/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e147_run && cd /root/e147_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
f="$OUT/rev200_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --steps 100 --transplant-eval --brain-seed 15 --reverse-after 200 \
  --trace-kc-class $WD/tr_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
grep -E '^\[반전\]|^\[사전\]|^\[사후\]|^\[학습\]|^\[반사가중치\] good_food' "$f" | cut -c1-200
[ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
cd $R && python3 - <<'PY'
import numpy as np
R = np.load("research/experiments/traces/E147/pathcheck/tr_b15.npz")["rows"]
idx = np.arange(len(R)); act = R[:, 6] >= 0
rule = np.where(idx < 200, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2])
agree = np.mean((R[act, 7] == 1) == rule[act])
eda = R[:, 13:17]; res = np.abs(R[:, 21:25] - ((11 / 12) ** 20) * eda).sum() / np.abs(eda).sum()
rw = [int((R[i:i + 100, 7] == 1).sum()) for i in range(0, len(R), 100)]
print("  시행 %d | 보상-규칙 일치 %.4f(앞 200: 실행≠자극, 뒤 200: 실행=자극) | 동결 잔차 %.2e | 블록 보상 %s" % (len(R), agree, res, rw))
PY
echo "[E147 경로 검사] 종료"
