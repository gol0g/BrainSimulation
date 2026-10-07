#!/bin/bash
# E155 경로 검사(조건 2) — 기준 고정(logs/E155/criteria_fixed.txt 23:31:21) 뒤. 표본 밖 뇌 15, 형성 표현(E153 경로 검사 가중치).
# 확인: 평가 경로(형성 가중치 + 변형 + 이식 가중치 — E153 경로 검사 100시행 가중치) base·noise × none·학습, 반전 학습 짧게(400시행, 200 부터 반전) — 적재·반전 줄·규칙 일치.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E155/pathcheck"; WD="$R/research/experiments/traces/E155/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e155_run && cd /root/e155_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
KW="$R/research/experiments/traces/E153/pathcheck/kctype_b15_eta01.npz"
WP="$R/research/experiments/traces/E153/pathcheck/w_b15.npz"
for WS in "none base" "none noise" "learn base" "learn noise"; do
  set -- $WS; W=$1; S=$2
  [ "$W" = "none" ] && X="--decomp-weights $WP --decomp-mode none" || X="--decomp-weights $WP --decomp-mode all"
  g="$OUT/ev_b15_${W}_$S.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW $X --eval-variant $S --eval-vseed 0 > "$g" 2>&1
  echo "[$W $S] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$g") $(grep '^\[E146 변형\]' "$g") $(grep '^=> DECOMP' "$g" | cut -c1-60)"
done
f="$OUT/rev_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --reverse-after 200 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW \
  --save-weights $WD/w_rev_b15.npz --trace-kc-class $WD/tr_rev_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
echo "[rev rc=$rc] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") $(grep '^\[반전\]' "$f" | cut -c1-40) $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
cd $R && python3 - <<'PY'
import numpy as np
R = np.load("research/experiments/traces/E155/pathcheck/tr_rev_b15.npz")["rows"]
n = len(R); idx = np.arange(n); act = R[:, 6] >= 0
rule = np.where(idx < 200, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2])
agree = float(np.mean((R[act, 7] == 1) == rule[act]))
res = float(np.abs(R[:, 21:25] - ((11 / 12) ** 20) * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
print("  반전 추적: 시행 %d 규칙 일치(200 부터 같은 쪽) %.4f 동결 잔차 %.2e 블록 보상 %s" % (n, agree, res, [int(R[i:i + 100, 7].sum()) for i in range(0, n, 100)]))
PY
echo "[E155 경로 검사] 종료"
