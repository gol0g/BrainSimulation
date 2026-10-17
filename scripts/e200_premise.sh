#!/bin/bash
# E200 전제 점검(보고용 — 기준 정정 1, 결과 전 추가): E199 의 고정 보상 학습에서 3,000시행 평균 DA 가 양인가(H121 의 전제).
# 뇌 10 의 E199 학습 커맨드(경쟁형 규칙·η 2.4·−4900·catctx 3,000)를 그대로 + --da-stats(읽기 전용 — E200 경로 검사 (a2) 에서 가중치 불변 확인)로 다시 돌려
# (1) '[E200 도파민 통계]' 평균 DA·평균 |DA| (2) 저장 가중치 = traces/E199/w_b10.npz(배열 정확 — 재현성)를 본다. 판정에 쓰지 않는다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E200/premise"; WD="$R/research/experiments/traces/E200/premise"; mkdir -p "$OUT" "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e200_premise_run && cd /root/e200_premise_run
cp $R/backend/genesis/*.py . 2>/dev/null
B=10
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 2.4 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"
KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
echo "[E200 전제 점검] 시작 $(date '+%F %T') — E199 b$B 학습 재현 + 도파민 통계"
f="$OUT/train_e199_b$B.log"
timeout 14400 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 30 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
  --save-weights $WD/w_e199_b$B.npz --kc-motor-hard --kc-motor-comp --da-stats > "$f" 2>&1; rc=$?
echo "[전제 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1) || $(grep '^\[E200 도파민 통계\]' "$f")"
python3 - "$R/research/experiments/traces/E199/w_b$B.npz" "$WD/w_e199_b$B.npz" <<'PY'
import sys
import numpy as np
A, B = np.load(sys.argv[1]), np.load(sys.argv[2])
same = sorted(A.files) == sorted(B.files) and all(np.array_equal(A[k], B[k]) for k in A.files)
print("[전제 대조] E199 b10 재현 가중치 = traces/E199/w_b10.npz(배열 %d): %s" % (len(A.files), "같음" if same else "다름"))
PY
echo "[E200 전제 점검] 종료 $(date '+%F %T')"
