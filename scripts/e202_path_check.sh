#!/bin/bash
# E202 경로 검사(조건 2, 기준 고정 뒤·본실험 전) — 표본 밖 뇌 15, catctx 40시행(기저값 −4000, η 0.15 — E200 경로 검사와 같은 커맨드).
# (a) E200 플래그(--kc-motor-hard --kc-motor-comp --da-baseline) 계측 끔: 저장 가중치 = traces/E200/pathcheck/w_base_b15.npz(배열 정확 — 새 코드가 E200 경로를 안 바꿈), E202 줄 0.
# (a2) (a) + --trace-dg: 저장 가중치 = (a)(배열 정확 — 계측은 읽기 전용), 점검 3줄·'[E202 계측] 시행 40 · …' 요약.
# (b) 계측 파일: 행 40·N 합 40·범주 합 = 총 x·값 유한, 총 x 와 (a2) 가중치 학습 Δ 최대 절대 오차 ≤ 1e-3, 범주 W·m 보고.
# (b 정정 1) 자극 쪽별 분할(xs·Ws·Ns) 합 = 합계.
# (c 정정 2) η 2.4(본실험 값) + --trace-dg 40시행: 전체 변화 x(정규화 뒤까지)의 재구성 최대 오차 ≤ 1e-3 — 진단 D1(정규화 전 Δg 만)은 최대 오차 59.
#     도파민 성분 xd 와 x 의 차(정규화 효과) 보고.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E202/pathcheck"; WD="$R/research/experiments/traces/E202/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
REF="$R/research/experiments/traces/E200/pathcheck/w_base_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e202_run && cd /root/e202_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
for f_ in "$KW15" "$SN15" "$REF"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E202 경로 검사] 종료"; exit 1; }; done
echo "[E202 경로 검사] 시작 $(date '+%F %T')"
f="$OUT/a_e200_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_a_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline > "$f" 2>&1; rc=$?
echo "[(a) E200 플래그 계측 끔 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | $(grep '^\[E200 도파민 통계\]' "$f") | E202 줄 $(grep -c '^\[E202' "$f")"
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
f="$OUT/a2_trace_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_a2_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline --trace-dg $WD/dg_pc_b15.npz > "$f" 2>&1; rc=$?
echo "[(a2) 계측 켬 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | $(grep '^\[E202 계측\] 시행 40 ' "$f")"
grep '^\[E202 계측\] 시행 [123] 범주' "$f" | sed 's/^/    /'
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
f="$OUT/c_eta24_train_b15.log"
BASE24="${BASE/--kc-motor-eta 0.15/--kc-motor-eta 2.4}"
timeout 3600 python reflex_override_task.py $BASE24 $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_c_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline --trace-dg $WD/dg_c_b15.npz > "$f" 2>&1; rc=$?
echo "[(c) η 2.4 계측 켬 rc=$rc] $(grep '^\[E202 계측\] 시행 40 ' "$f" | sed 's/ · 저장.*//')"
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
python3 - "$REF" "$WD/w_a_b15.npz" "$WD/w_a2_b15.npz" "$WD/dg_pc_b15.npz" "$SN15" <<'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import verify_e198_independent as V
ref, a, a2, dgp, sn = sys.argv[1:6]
def same(x, y):
    X, Y = np.load(x), np.load(y)
    return sorted(X.files) == sorted(Y.files) and all(np.array_equal(X[k], Y[k]) for k in X.files), len(X.files)
s1, n1 = same(ref, a); s2, n2 = same(a, a2)
print("[대조 (a)] E200 플래그 계측 끔 = E200 경로 검사 (b) 가중치(배열 %d): %s" % (n1, "같음" if s1 else "다름"))
print("[대조 (a2)] 계측 켬 = 계측 끔 가중치(배열 %d): %s" % (n2, "같음" if s2 else "다름"))
D = np.load(dgp)
CL = ("Sp", "Sm", "Cp", "Cm")
rows_ok = D["rows"].shape[0] == 40 and int(D["N"].sum()) == 40
sums_ok = all(np.allclose(sum(D["x_%s_%s" % (c, sd)] for c in CL), D["tot_x_" + sd], atol=1e-9) for sd in "lr")
fin = all(np.all(np.isfinite(D[k])) for k in D.files if D[k].dtype.kind == "f")
dl, _ = V.learned(a2, sn)
err = max(float(np.max(np.abs(np.asarray(dl[sd], float) - D["tot_x_" + sd]))) for sd in "lr")
mag = max(float(np.max(np.abs(D["tot_x_" + sd]))) for sd in "lr")
print("[대조 (b)] 계측 파일: 행 %d·N %s(합 %d)·W %s · 범주 합 = 총 x %s · 유한 %s · 재구성 최대 오차 %.2e(총 x 최대 %.3f)"
      % (D["rows"].shape[0], list(int(v) for v in D["N"]), int(D["N"].sum()), " ".join("%+.3f" % v for v in D["W"]), sums_ok, fin, err, mag))
split_ok = all(np.allclose(D["xs_%s_l_%s" % (c, sd)] + D["xs_%s_r_%s" % (c, sd)], D["x_%s_%s" % (c, sd)], atol=1e-9) for c in CL for sd in "lr") \
    and D["Ws"].shape == (4, 2) and np.allclose(D["Ws"].sum(axis=1), D["W"]) and list(D["Ns"].sum(axis=1)) == list(D["N"])
print("[대조 (b) 정정 1] 자극 쪽별 분할: 분할 합 = 합계 %s · Ns %s" % (split_ok, D["Ns"].tolist()))
ok = s1 and s2 and rows_ok and sums_ok and fin and err <= 1e-3 and mag > 0 and split_ok
print("[대조 (b)] %s" % ("통과" if ok else "실패"))
# (c) η 2.4 — 전체 변화 x 재구성(정정 2)
wc, dc = a2.replace("w_a2_b15", "w_c_b15"), dgp.replace("dg_pc_b15", "dg_c_b15")
C = np.load(dc)
dlc, _ = V.learned(wc, sn)
errc = max(float(np.max(np.abs(np.asarray(dlc[sd], float) - C["tot_x_" + sd]))) for sd in "lr")
magc = max(float(np.max(np.abs(C["tot_x_" + sd]))) for sd in "lr")
nrm = max(float(np.max(np.abs(C["tot_x_" + sd] - C["tot_xd_" + sd]))) for sd in "lr")
okc = C["rows"].shape[0] == 40 and errc <= 1e-3 and magc > 0
print("[대조 (c)] η 2.4: 행 %d · 전체 x 재구성 최대 오차 %.2e(총 x 최대 %.3f) · 정규화 효과 max|x − xd| %.3f → %s"
      % (C["rows"].shape[0], errc, magc, nrm, "통과" if okc else "실패"))
PY
echo "[E202 경로 검사] 종료 $(date '+%F %T')"
