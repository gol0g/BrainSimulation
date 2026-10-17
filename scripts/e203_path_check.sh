#!/bin/bash
# E203 경로 검사(조건 2, 기준 고정 뒤·본실험 전) — 표본 밖 뇌 15, catctx 40시행(기저값 −4000 — E200·E201 경로 검사와 같은 커맨드).
# (a) E200 플래그(--kc-motor-hard --kc-motor-comp --da-baseline, η 0.15): 저장 가중치 = traces/E200/pathcheck/w_base_b15.npz(배열 정확 — 새 코드가 E200 경로를 안 바꿈), E203 줄 0.
# (b) (a) + --kc-motor-actw traces/E196/pathcheck/sxi_b15.npz(η 0.15): '[E203 활동 가중 정규화]' 점검 3줄·'적용 40회 … 뒤 최대 ≤ 1e-3'·
#     저장 가중치 쪽별 활동 가중 |M| ≤ 1.0·KC 별 합 300·(a) 와 다름.
# (c) (b) 를 η 2.4(본실험 동작점 — E202 교훈)로: 같은 검사.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E203/pathcheck"; WD="$R/research/experiments/traces/E203/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
AW15="$R/research/experiments/traces/E196/pathcheck/sxi_b15.npz"
REF9="$R/research/experiments/traces/E200/pathcheck/w_base_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e203_run && cd /root/e203_run
cp $R/backend/genesis/*.py . 2>/dev/null
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
for f_ in "$KW15" "$SN15" "$AW15" "$REF9"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E203 경로 검사] 종료"; exit 1; }; done
echo "[E203 경로 검사] 시작 $(date '+%F %T')"
for spec in "a 0.15 0" "b 0.15 1" "c 2.4 1"; do
  set -- $spec; N=$1; ETA=$2; ON=$3
  BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta $ETA --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
  EXTRA=""; [ "$ON" = "1" ] && EXTRA="--kc-motor-actw $AW15"
  f="$OUT/${N}_train_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${N}_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline $EXTRA > "$f" 2>&1; rc=$?
  echo "[($N) η $ETA 활동 가중 $ON rc=$rc] $(grep '^\[맥락 범주 과제\] 시행' "$f") | $(grep '^\[E199 경쟁 정규화\] 적용' "$f" | cut -c1-60) | $(grep '^\[E203 활동 가중 정규화\] 적용' "$f") | E203 줄 $(grep -c '^\[E203' "$f")"
  grep '^\[E203 활동 가중 정규화\] 시행\|^\[E203 활동 가중 정규화\] 반응' "$f" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
done
python3 - "$REF9" "$WD" "$SN15" "$AW15" "$OUT" <<'PY'
import os
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis")
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import kc_comp
import judge_e203 as J3
ref, wd, sn, aw, out = sys.argv[1:6]
def same(x, y):
    X, Y = np.load(x), np.load(y)
    return sorted(X.files) == sorted(Y.files) and all(np.array_equal(X[k], Y[k]) for k in X.files), len(X.files)
s1, n1 = same(ref, os.path.join(wd, "w_a_b15.npz"))
print("[대조 (a)] E200 플래그 = E200 경로 검사 (b) 가중치(배열 %d): %s" % (n1, "같음" if s1 else "다름"))
A, S = np.load(os.path.join(wd, "w_a_b15.npz")), np.load(sn)
pops = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")
allok = s1
for N in ("b", "c"):
    B = np.load(os.path.join(wd, "w_%s_b15.npz" % N))
    dmax = 0.0
    for kn in "lr":
        x, y = "kc_%s_to_motor_l" % kn, "kc_%s_to_motor_r" % kn
        s, both = kc_comp.kc_sums(S[x + "__pre"], B[x], S[y + "__pre"], B[y], 3000)
        dmax = max(dmax, float(np.max(np.abs(s[both] - 300.0))))
    diff = any(not np.array_equal(B[p], A[p]) for p in pops)
    rng_ok = all(float(B[p].min()) >= 0 and float(B[p].max()) <= 300 for p in pops)
    am = J3.actw_offsets(os.path.join(wd, "w_%s_b15.npz" % N), sn, aw)
    am_a = J3.actw_offsets(os.path.join(wd, "w_a_b15.npz"), sn, aw)
    lg = open(os.path.join(out, "%s_train_b15.log" % N), encoding="utf-8", errors="replace").read()
    import re
    mm = re.search(r"^\[E203 활동 가중 정규화\] 적용 (\d+)회 .*뒤 최대 (\S+) · 반복 최대 (\d+)", lg, re.M)
    nchk = len(re.findall(r"^\[E203 활동 가중 정규화\] 시행 \d+ — ", lg, re.M))
    line_ok = mm is not None and int(mm.group(1)) == 40 and float(mm.group(2)) <= 1e-3 and nchk == 3
    print("    (%s) 로그: 요약 %s · 점검 %d줄 → %s" % (N, mm.group(0)[len("[E203 활동 가중 정규화] "):] if mm else "없음", nchk, "통과" if line_ok else "실패"))
    ok = dmax <= 0.01 and diff and rng_ok and max(abs(v) for v in am.values()) <= 1.0 and line_ok
    allok = allok and ok
    print("[대조 (%s)] 활동 가중 켬: 저장 가중치 M l %+.2e·r %+.2e(끔 (a) l %+.3f·r %+.3f) · KC 별 합 최대 |S−300| %.2e · 범위 %s · (a) 와 다름 %s → %s"
          % (N, am["l"], am["r"], am_a["l"], am_a["r"], dmax, rng_ok, diff, "통과" if ok else "실패"))
print("[대조] %s" % ("전부 통과" if allok else "실패 있음"))
PY
echo "[E203 경로 검사] 종료 $(date '+%F %T')"
