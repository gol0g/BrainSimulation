#!/bin/bash
# E139 경로 검사(조건 2) — 판정 기준 고정(logs/E139/criteria_fixed.txt) 뒤. 표본 밖 뇌 15(E119 경로 검사와 같은 인자 — 판정 v).
# (1) 비간섭(수정 2026-10-03): 런 간 비결정성(재현성 검사 560~2136개) 안인가 — E119·추적 없음 A·B·1차 추적과 비교, [사후] −0.0734 재현
# (4) 수정 추적 필드: 도파민 전 변화 ≈0(V4), 보상 창 끝 흔적, 보상 창 motor 발화율(KCTRACE3)
# (2) 합 일관성·g 연속성·시행 500 기록 (KCTRACE 줄) (3) 분류 파일 = E138 경로 검사 rate_b15.npz
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E139/pathcheck"; WD="$R/research/experiments/traces/E139/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e139_run && cd /root/e139_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000"
f="$OUT/trace_b15.log"
echo "[1~3] 뇌 15 학습 + 계층 추적(판정 v — 원 경로 검사와 같음)"
timeout 7200 python reflex_override_task.py $BASE $ACT --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 \
  --save-weights $WD/w_b15.npz --trace-kc-class $WD/tr_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
grep -E '^\[E139\]|^=> KCTRACE|^\[사전\]|^\[사후\]|^\[학습\]' "$f" | cut -c1-500 || true
[ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
echo "[1 비간섭(잡음 바닥)] 저장 가중치 vs E119 path_rw0_b15·추적 없음 A·B(재현성 검사)"
python3 - "$WD" "$R/research/experiments/traces/E119/path_rw0_b15.npz" <<'PY'
import sys, numpy as np
wd, e119 = sys.argv[1], sys.argv[2]
a = np.load(wd + "/w_b15.npz")
for nm, f in (("E119", e119), ("추적없음A", wd + "/w_notrace_A_b15.npz"), ("추적없음B", wd + "/w_notrace_B_b15.npz"), ("1차 추적", wd + "/w_b15_1st.npz")):
    b = np.load(f)
    nd = sum(int((a[k] != b[k]).sum()) for k in a.files); mx = max(float(np.abs(a[k] - b[k]).max()) for k in a.files)
    print("    수정 추적 vs %s: 다른 시냅스 %d, 최대 |차| %.4f" % (nm, nd, mx))
PY
echo "[E139 경로 검사] 종료"
