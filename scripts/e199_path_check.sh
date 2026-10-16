#!/bin/bash
# E199 경로 검사(조건 2, 기준 고정 뒤·보정·본실험 전) — 표본 밖 뇌 15.
# (a) 규칙 끔 회귀: E193 경로 검사 (b) 커맨드(catctx 40시행, 기저값 −4000)를 그대로 → 저장 가중치가 traces/E193/pathcheck/w_b15.npz 와 배열 정확히 같음.
# (b) 규칙 켬: 같은 40시행 + --kc-motor-hard --kc-motor-comp(η 0.15) → '[E199 가산형' 줄·정규화 점검 3줄·'적용 40회 … hard=1'·KC→motor [0, 300]·KC 별 합 300·(a) 와 다름.
# (c) 규칙 켬 가중치로 이식 평가 learn × 맥락 X → RCEVAL 4·pushed 8·포화 없음(평가 경로는 E193 과 같음).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E199/pathcheck"; WD="$R/research/experiments/traces/E199/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
REF="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e199_run && cd /root/e199_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
for f_ in "$KW15" "$SN15" "$REF"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E199 경로 검사] 종료"; exit 1; }; done
echo "[E199 경로 검사] 시작 $(date '+%F %T')"
f="$OUT/a_reg_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_reg_b15.npz > "$f" 2>&1; rc=$?
echo "[(a) 규칙 끔 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | E199 줄 $(grep -c 'E199' "$f")"
f="$OUT/b_comp_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_comp_b15.npz --kc-motor-hard --kc-motor-comp > "$f" 2>&1; rc=$?
echo "[(b) 규칙 켬 rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | 가산형 줄 $(grep -c 'E199 가산형' "$f") | 정규화 점검 $(grep -c '^\[E199 경쟁 정규화\] 시행' "$f") | $(grep '^\[E199 경쟁 정규화\] 적용' "$f")"
grep '^\[E199 경쟁 정규화\] 시행' "$f" | sed 's/^/    /'
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
g="$OUT/c_ev_comp_X_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_comp_b15.npz --decomp-mode all --rc-eval --rc-eval-ctx X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(c) 평가 learn(규칙 켬 가중치) 맥락 X rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+') | motor $(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{print $3}' | tr '\n' ' ')"
python3 - "$REF" "$WD/w_reg_b15.npz" "$WD/w_comp_b15.npz" "$SN15" "$OUT/b_comp_train_b15.log" <<'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis")
import kc_comp
import judge_e199 as J
ref, reg, comp, sn, lg = sys.argv[1:6]
A, B, C, S = np.load(ref), np.load(reg), np.load(comp), np.load(sn)
same = sorted(A.files) == sorted(B.files) and all(np.array_equal(A[k], B[k]) for k in A.files)
print("[대조 (a)] 규칙 끔 가중치 = E193 경로 검사 가중치(배열 %d 개 정확): %s" % (len(A.files), "같음" if same else "다름 " + str([k for k in A.files if k not in B.files or not np.array_equal(A[k], B[k])])))
pops = J.POPS
rng_ok = all(float(C[p].min()) >= 0 and float(C[p].max()) <= 300 for p in pops)
dmax = 0.0
for kn in "lr":
    a, b = "kc_%s_to_motor_l" % kn, "kc_%s_to_motor_r" % kn
    s, both = kc_comp.kc_sums(S[a + "__pre"], C[a], S[b + "__pre"], C[b], 3000)
    dmax = max(dmax, float(np.max(np.abs(s[both] - 300.0))))
diff = any(not np.array_equal(C[p], B[p]) for p in pops)
cp = J.read_comp(lg)
good = rng_ok and dmax <= 0.01 and diff and cp["hard"] and cp["comp"] is not None and cp["comp"][0] == 40 and cp["comp"][1] <= 1e-3 and cp["comp"][2] == 1
print("[대조 (b)] 규칙 켬: 범위 %s · KC 별 합 최대 |S−300| %.2e · (a) 와 다름 %s · 규칙 줄 %s %s → %s" % (rng_ok, dmax, diff, cp["hard"], cp["comp"], "통과" if good else "실패"))
PY
echo "[E199 경로 검사] 종료 $(date '+%F %T')"
