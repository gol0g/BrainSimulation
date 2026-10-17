#!/bin/bash
# E200 경로 검사(조건 2, 기준 고정 뒤·본실험 전) — 표본 밖 뇌 15, catctx 40시행(기저값 −4000 — E193·E199 경로 검사와 같은 커맨드).
# (a) 규칙 끔 회귀: 저장 가중치 = traces/E193/pathcheck/w_b15.npz(배열 정확).
# (a2) E199 플래그(--kc-motor-hard --kc-motor-comp, η 0.15) + --da-stats(읽기 전용): 저장 가중치 = traces/E199/pathcheck/w_comp_b15.npz(배열 정확 — 새 코드가 E199 경로를 안 바꿈), 고정 보상의 평균 DA 보고.
# (b) (a2) + --da-baseline: '[E200 도파민 기준선]' 점검 3줄·'[E200 도파민 통계] 시행 40 … 기준선 1'·KC 별 합 300·(a2) 와 다름.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E200/pathcheck"; WD="$R/research/experiments/traces/E200/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
REF3="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
REF9="$R/research/experiments/traces/E199/pathcheck/w_comp_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e200_run && cd /root/e200_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
for f_ in "$KW15" "$SN15" "$REF3" "$REF9"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E200 경로 검사] 종료"; exit 1; }; done
echo "[E200 경로 검사] 시작 $(date '+%F %T')"
f="$OUT/a_reg_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_reg_b15.npz > "$f" 2>&1; rc=$?
echo "[(a) 규칙 끔 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | E200 줄 $(grep -c '^\[E200' "$f")"
f="$OUT/a2_e199_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_e199_b15.npz --kc-motor-hard --kc-motor-comp --da-stats > "$f" 2>&1; rc=$?
echo "[(a2) E199 플래그 + 도파민 통계 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | $(grep '^\[E199 경쟁 정규화\] 적용' "$f") | $(grep '^\[E200 도파민 통계\]' "$f") | 기준선 점검 $(grep -c '^\[E200 도파민 기준선\]' "$f")"
f="$OUT/b_base_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_base_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline > "$f" 2>&1; rc=$?
echo "[(b) 기준선 켬 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | $(grep '^\[E199 경쟁 정규화\] 적용' "$f") | $(grep '^\[E200 도파민 통계\]' "$f")"
grep '^\[E200 도파민 기준선\]' "$f" | sed 's/^/    /'
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
python3 - "$REF3" "$WD/w_reg_b15.npz" "$REF9" "$WD/w_e199_b15.npz" "$WD/w_base_b15.npz" "$SN15" <<'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis")
import kc_comp
r3, a, r9, a2, b, sn = sys.argv[1:7]
def same(x, y):
    X, Y = np.load(x), np.load(y)
    return sorted(X.files) == sorted(Y.files) and all(np.array_equal(X[k], Y[k]) for k in X.files), len(X.files)
s1, n1 = same(r3, a); s2, n2 = same(r9, a2)
print("[대조 (a)] 규칙 끔 = E193 경로 검사 가중치(배열 %d): %s" % (n1, "같음" if s1 else "다름"))
print("[대조 (a2)] E199 플래그 + --da-stats = E199 경로 검사 (b) 가중치(배열 %d): %s" % (n2, "같음" if s2 else "다름"))
B, A2, S = np.load(b), np.load(a2), np.load(sn)
pops = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")
dmax = 0.0
for kn in "lr":
    x, y = "kc_%s_to_motor_l" % kn, "kc_%s_to_motor_r" % kn
    s, both = kc_comp.kc_sums(S[x + "__pre"], B[x], S[y + "__pre"], B[y], 3000)
    dmax = max(dmax, float(np.max(np.abs(s[both] - 300.0))))
diff = any(not np.array_equal(B[p], A2[p]) for p in pops)
print("[대조 (b)] 기준선 켬: KC 별 합 최대 |S−300| %.2e · 범위 %s · (a2) 와 다름 %s → %s" % (dmax, all(float(B[p].min()) >= 0 and float(B[p].max()) <= 300 for p in pops), diff,
      "통과" if (dmax <= 0.01 and diff) else "실패"))
PY
echo "[E200 경로 검사] 종료 $(date '+%F %T')"
