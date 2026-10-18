#!/bin/bash
# E204 경로 검사(조건 2, 기준 고정 뒤·본실험 전) — 표본 밖 뇌 15, 40시행, 기저값 −4000(E200~E203 경로 검사와 같은 커맨드).
# (a) 게이팅 끔·E200 플래그·η 0.15·catctx → traces/E200/pathcheck/w_base_b15.npz 와 배열 정확(새 코드가 E200 경로를 안 바꿈)
# (b) catrule(맥락 안 씀) 게이팅 끔 = 켬(배열 정확 — 억제 시냅스가 맥락 꺼짐일 때 동역학 불변)
# (c) 게이팅 켬·η 2.4(본실험 동작점)·catctx → 점검 3줄(활성 > 0)·요약 비 ≤ 0.01·구성 줄 2·KC 별 합 300·범위·같은 η 게이팅 끔(c0)과 가중치 다름
# (d) 새 세계 — 게이팅 켬·η 2.4·--rc-seed 28300 --env-seed 1 --rc-trial-seed 28515 --rc-ctx-seed 28615: 학습 로그 시드 줄, 무학습 평가(--rc-seed 28300)의 평가 세계 줄·게이팅 구성 줄
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E204/pathcheck"; WD="$R/research/experiments/traces/E204/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
REF9="$R/research/experiments/traces/E200/pathcheck/w_base_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e204_run && cd /root/e204_run
cp $R/backend/genesis/*.py . 2>/dev/null
B0="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
NEWW="--rc-seed 28300 --rc-trial-seed 28515 --rc-ctx-seed 28615"
for f_ in "$KW15" "$SN15" "$REF9"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E204 경로 검사] 종료"; exit 1; }; done
echo "[E204 경로 검사] 시작 $(date '+%F %T')"
for spec in "a catctx 0.15 0 0 0" "bo catrule 0.15 0 0 0" "bn catrule 0.15 1 0 0" "c0 catctx 2.4 0 0 0" "c catctx 2.4 1 0 0" "d catctx 2.4 1 1 1"; do
  set -- $spec; N=$1; TK=$2; ETA=$3; ON=$4; NW=$5; ENV=$6
  EXTRA=""; [ "$ON" = "1" ] && EXTRA="$GATE"; [ "$NW" = "1" ] && EXTRA="$EXTRA $NEWW"
  f="$OUT/${N}_train_b15.log"
  timeout 1800 python reflex_override_task.py $B0 --kc-motor-eta $ETA --env-seed $ENV $ACT $FEAT $CTX --rc-task $TK --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${N}_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline $EXTRA > "$f" 2>&1; rc=$?
  echo "[($N) $TK η $ETA 게이팅 $ON 새 세계 $NW rc=$rc] $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$f") | 구성 줄 $(grep -c 'E204 맥락 게이팅 KC\] 무리' "$f")"
  grep '^\[E204 맥락 게이팅 점검\]\|^\[범주 과제\] catrule\|^\[맥락 범주 과제\] catctx' "$f" | cut -c1-160 | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$f" | sed 's/^/      /'
done
g="$OUT/d_ev_none_X_b15.log"
EB="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
timeout 1800 python reflex_override_task.py $EB $ACT $FEAT $CTX $GATE --rc-seed 28300 --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_d_b15.npz --decomp-mode none --rc-eval --rc-eval-ctx X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(d 평가) none X rc=$rc] $(grep '^\[E204 평가 세계\]' "$g") | 구성 줄 $(grep -c 'E204 맥락 게이팅 KC\] 무리' "$g") | RCEVAL $(grep -c '^RCEVAL ' "$g")"
python3 - "$REF9" "$WD" "$SN15" "$OUT" <<'PY'
import os, re, sys
import numpy as np
ref, wd, sn, out = sys.argv[1:5]
P4 = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")
def same(x, y):
    X, Y = np.load(x), np.load(y)
    return sorted(X.files) == sorted(Y.files) and all(np.array_equal(X[k], Y[k]) for k in X.files), len(X.files)
def kcsum(w):
    W, S = np.load(w), np.load(sn); dev = 0.0
    for kn in "lr":
        ms, cs = [], []
        for mn in "lr":
            p = "kc_%s_to_motor_%s" % (kn, mn); pre = S[p + "__pre"].astype(np.int64)
            s_ = np.bincount(pre, weights=np.asarray(W[p], float), minlength=3000); c_ = np.bincount(pre, minlength=3000).astype(float)
            ms.append(s_ / np.maximum(c_, 1)); cs.append(c_)
        both = (cs[0] > 0) & (cs[1] > 0); dev = max(dev, float(np.max(np.abs(ms[0][both] + ms[1][both] - 300))))
    rng = all(float(W[p].min()) >= 0 and float(W[p].max()) <= 300 for p in P4)
    return dev, rng
def grab(fn, pat):
    for ln in open(os.path.join(out, fn), encoding="utf-8", errors="replace"):
        m = re.search(pat, ln)
        if m: return m
    return None
ok = True
s1, n1 = same(ref, os.path.join(wd, "w_a_b15.npz")); ok &= s1
print("[대조 (a)] 게이팅 끔 = E200 경로 검사 (b) 가중치(배열 %d): %s" % (n1, "같음" if s1 else "다름"))
s2, n2 = same(os.path.join(wd, "w_bo_b15.npz"), os.path.join(wd, "w_bn_b15.npz")); ok &= s2
print("[대조 (b)] catrule 게이팅 끔 = 켬(배열 %d): %s" % (n2, "같음" if s2 else "다름"))
gs = grab("c_train_b15.log", r"\[E204 맥락 게이팅 KC\] 시행 (\d+) · .*활성 무리 합 (\d+) · 비활성 무리 합 (\d+) · 비\(비활성/활성\) ([0-9.]+)")
chk = [ln for ln in open(os.path.join(out, "c_train_b15.log"), encoding="utf-8", errors="replace") if ln.startswith("[E204 맥락 게이팅 점검]")]
nb = sum(1 for ln in open(os.path.join(out, "c_train_b15.log"), encoding="utf-8", errors="replace") if "E204 맥락 게이팅 KC] 무리" in ln)
act_ok = all(int(re.search(r"활성 무리 (\d+)", ln).group(1)) > 0 for ln in chk)
s3, _ = same(os.path.join(wd, "w_c0_b15.npz"), os.path.join(wd, "w_c_b15.npz"))
dev, rng = kcsum(os.path.join(wd, "w_c_b15.npz"))
c_ok = bool(gs) and int(gs.group(1)) == 40 and float(gs.group(4)) <= 0.01 and len(chk) == 3 and act_ok and nb == 2 and (not s3) and dev <= 0.01 and rng
ok &= c_ok
print("[검사 (c)] η 2.4 게이팅 — 요약 %s · 점검 %d줄(활성 > 0 %s) · 구성 줄 %d · c0 와 다름 %s · KC 합 최대 |S−300| %.2e · 범위 %s → %s"
      % (gs.group(0) if gs else "없음", len(chk), act_ok, nb, not s3, dev, rng, "통과" if c_ok else "실패"))
wseed = grab("d_train_b15.log", r"세계 시드 (\d+)\).*배정 지역 난수 (\d+)"); cseed = grab("d_train_b15.log", r"catctx — .*지역 난수 (\d+)\)")
ew = grab("d_ev_none_X_b15.log", r"\[E204 평가 세계\] 세계 시드 (\d+)")
eb = sum(1 for ln in open(os.path.join(out, "d_ev_none_X_b15.log"), encoding="utf-8", errors="replace") if "E204 맥락 게이팅 KC] 무리" in ln)
d_ok = bool(wseed and cseed and ew) and (int(wseed.group(1)), int(wseed.group(2)), int(cseed.group(1)), int(ew.group(1))) == (28300, 28515, 28615, 28300) and eb >= 1
ok &= d_ok
print("[검사 (d)] 새 세계 — 학습 세계·배정·맥락 %s·%s·%s · 평가 세계 %s · 평가 구성 줄 %d → %s"
      % (wseed.group(1) if wseed else "?", wseed.group(2) if wseed else "?", cseed.group(1) if cseed else "?", ew.group(1) if ew else "?", eb, "통과" if d_ok else "실패"))
print("[E204 경로 검사] %s" % ("통과" if ok else "실패"))
PY
echo "[E204 경로 검사] 종료 $(date '+%F %T')"
