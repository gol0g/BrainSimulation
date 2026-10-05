#!/bin/bash
# E139 경로 검사 추가(V1 판단): 전체 모델 학습의 런 간 재현성. 표본 밖 뇌 15, 추적 없이 같은 학습 2회(A·B, E119 경로 검사 인자 — 판정 v).
# 비교: A vs B(런 간 잡음 바닥), A·B vs E119 path_rw0_b15(시점 간), 추적 런(E139 pathcheck) vs A·B(추적 간섭 여부).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E139/pathcheck"; WD="$R/research/experiments/traces/E139/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e139_run && cd /root/e139_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000"
for X in A B; do
  f="$OUT/notrace_${X}_b15.log"
  timeout 7200 python reflex_override_task.py $BASE $ACT --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --save-weights $WD/w_notrace_${X}_b15.npz > "$f" 2>&1
  echo "[추적 없음 $X] $(grep -E '^\[학습\]|^\[사후\]' "$f" | tr '\n' ' ' | cut -c1-200)"
done
python3 - "$WD" "$R/research/experiments/traces/E119/path_rw0_b15.npz" <<'PY'
import sys, numpy as np
wd, e119 = sys.argv[1], sys.argv[2]
W = {"추적없음A": np.load(wd + "/w_notrace_A_b15.npz"), "추적없음B": np.load(wd + "/w_notrace_B_b15.npz"),
     "추적": np.load(wd + "/w_b15.npz"), "E119": np.load(e119)}
def cmp(x, y):
    a, b = W[x], W[y]
    nd = sum(int((a[k] != b[k]).sum()) for k in a.files); mx = max(float(np.abs(a[k] - b[k]).max()) for k in a.files)
    return nd, mx
for x, y in (("추적없음A", "추적없음B"), ("추적없음A", "E119"), ("추적없음B", "E119"), ("추적", "추적없음A"), ("추적", "추적없음B"), ("추적", "E119")):
    nd, mx = cmp(x, y)
    print("    %s vs %s: 다른 시냅스 %d, 최대 |차| %.4f" % (x, y, nd, mx))
PY
echo "[E139 재현성 검사] 종료"
