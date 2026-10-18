#!/bin/bash
# E205 경로 검사(조건 2) — 표본 밖 뇌 15, 40시행(2 × 20). 새 코드 --rc-rule-ctx(catrule 맥락 고정, 기본 none).
# (a) 회귀: catrule η 0.15·기저값 −4000·--rc-rule-ctx 기본(none) — 게이팅 0 = E204 경로 검사 (bo) 가중치, 게이팅 1 = (bn) 가중치(배열 정확)
# (b) 맥락 X 고정: catrule η 2.4·게이팅·T_G −2800·--rc-rule-ctx X — 알림 줄·맥락 점검 3줄(켠 > 0·끈 0·시행 끝 0)·게이팅 점검 3줄(활성 > 0)·요약 시행 40·비 ≤ 0.01·
#     맥락 켬 40·범주 배정 줄이 같은 시드 none(b0)과 같음·가중치는 b0 와 다름·KC 별 합 300·범위
# (c) 동작점(보고 — 맥락 X 고정 결정의 근거): 무학습 평가 motor 평균(E204 보정과 같은 측정) — 게이팅·T_G, 맥락 없음 대 맥락 X(E204 기준 m_ref 0.6094·m_G 0.5998)
# (d) 새 세계·시행열: --rc-seed 38300 --env-seed 2 --rc-trial-seed 38515(--rc-ctx-seed 38615) — 학습 로그 시드 줄(catrule·catctx), 평가 세계 줄(원형 A 5,14,15,19,29)·RCEVAL 8(A·B·C·D)·ctx_on 96
# (e) 반전·둘째 쌍 + 맥락 X 고정: --rc-reverse-after 20·--rc-phase2-after 20 — 반전·둘째 쌍 줄, 맥락 켬 40, 학습 가중치 평가(pushed 8)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E205/pathcheck"; WD="$R/research/experiments/traces/E205/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
REFO="$R/research/experiments/traces/E204/pathcheck/w_bo_b15.npz"
REFN="$R/research/experiments/traces/E204/pathcheck/w_bn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e205_run && cd /root/e205_run
cp $R/backend/genesis/*.py . 2>/dev/null
B0="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
NEWW="--rc-seed 38300 --rc-trial-seed 38515"
for f_ in "$KW15" "$SN15" "$REFO" "$REFN" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E205 경로 검사] 종료"; exit 1; }; done
echo "[E205 경로 검사] 시작 $(date '+%F %T')"
# 이름 과제 η 게이팅 기저값 맥락고정 새세계 env 추가
for spec in "ao catrule 0.15 0 -4000 none 0 0 -" "an catrule 0.15 1 -4000 none 0 0 -" "bx catrule 2.4 1 -2800 X 0 0 -" "b0 catrule 2.4 1 -2800 none 0 0 -" \
            "dv catrule 2.4 1 -2800 X 1 2 rev" "dp catrule 2.4 1 -2800 X 1 2 p2" "dl catctx 2.4 1 -2800 none 1 2 -"; do
  set -- $spec; N=$1; TK=$2; ETA=$3; ON=$4; TON=$5; RX=$6; NW=$7; ENV=$8; AD=$9
  EXTRA=""; [ "$ON" = "1" ] && EXTRA="$GATE"; [ "$NW" = "1" ] && EXTRA="$EXTRA $NEWW"; [ "$NW" = "1" ] && [ "$TK" = "catctx" ] && EXTRA="$EXTRA --rc-ctx-seed 38615"
  [ "$RX" != "none" ] && EXTRA="$EXTRA --rc-rule-ctx $RX"
  [ "$AD" = "rev" ] && EXTRA="$EXTRA --rc-reverse-after 20"; [ "$AD" = "p2" ] && EXTRA="$EXTRA --rc-phase2-after 20"
  f="$OUT/${N}_train_b15.log"
  timeout 1800 python reflex_override_task.py $B0 --kc-motor-eta $ETA --env-seed $ENV $ACT $FEAT $CW --motor-tonic $TON --rc-task $TK --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${N}_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline $EXTRA > "$f" 2>&1; rc=$?
  echo "[($N) $TK η $ETA 게이팅 $ON 기저값 $TON 맥락고정 $RX 새세계 $NW env $ENV $AD rc=$rc] $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$f" | cut -c1-120) | $(grep '^\[범주 규칙 맥락\] 시행' "$f")"
  grep '^\[범주 규칙 맥락\] catrule\|^\[맥락 범주 점검\]\|^\[E204 맥락 게이팅 점검\]\|^\[범주 과제\] catrule\|^\[범주 과제\] 시행\|^\[맥락 범주 과제\] catctx\|^\[범주 반전\]\|^\[둘째 범주 쌍\]' "$f" | cut -c1-170 | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$f" | sed 's/^/      /'
done
# 평가: (c1) 맥락 없음 (c2) 맥락 X — 무학습, 게이팅·T_G / (d) 새 세계·A·B·C·D·맥락 X 무학습 / (e) dv 가중치 학습 평가(맥락 X)
EB="$B0 --kc-motor-eta 0.15 --env-seed 0"
for spec in "c1 none $WB none - A,B" "c2 none $WB X - A,B" "dn none $WD/w_dv_b15.npz X 38300 A,B,C,D" "el all $WD/w_dv_b15.npz X 38300 A,B"; do
  set -- $spec; N=$1; XM=$2; WF=$3; EX=$4; WS=$5; CATS=$6
  EXTRA=""; [ "$EX" != "none" ] && EXTRA="--rc-eval-ctx $EX"; [ "$WS" != "-" ] && EXTRA="$EXTRA --rc-seed $WS"
  g="$OUT/ev_${N}_b15.log"
  timeout 1800 python reflex_override_task.py $EB $ACT $FEAT $CW $GATE --motor-tonic -2800 --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WF --decomp-mode $XM \
    --rc-eval --rc-eval-cats $CATS --trials 96 --eval-diag $EXTRA > "$g" 2>&1; rc=$?
  m=$(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{split($3,a,"/"); s+=a[1]+a[2]; n+=2} END {if (n) printf "%.4f", s/n; else print "nan"}')
  echo "[(ev $N) $XM 맥락 $EX 세계 $WS 범주 $CATS rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | motor 평균 $m | $(grep '^\[E204 평가 세계\]' "$g" | cut -c1-90) | 구성 줄 $(grep -c 'E204 맥락 게이팅 KC\] 무리' "$g") | $(grep -oE 'pushed=[0-9]+' "$g" | head -1)"
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
python3 - "$REFO" "$REFN" "$WD" "$SN15" "$OUT" <<'PY'
import os, re, sys
import numpy as np
refo, refn, wd, sn, out = sys.argv[1:6]
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
    return dev, all(float(W[p].min()) >= 0 and float(W[p].max()) <= 300 for p in P4)
def lines(fn, pref):
    return [ln.rstrip("\n") for ln in open(os.path.join(out, fn), encoding="utf-8", errors="replace") if ln.startswith(pref)]
def one(fn, pat):
    for ln in open(os.path.join(out, fn), encoding="utf-8", errors="replace"):
        m = re.search(pat, ln)
        if m: return m
    return None
def mot(fn):
    v = [float(x) for ln in lines(fn, "RCEVAL ") for ab in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", ln) for x in ab]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
ok = True
s1, n1 = same(refo, os.path.join(wd, "w_ao_b15.npz")); s2, n2 = same(refn, os.path.join(wd, "w_an_b15.npz")); ok &= s1 and s2
print("[대조 (a)] --rc-rule-ctx 기본 none = E204 (bo) 게이팅 0(배열 %d): %s · (bn) 게이팅 1(배열 %d): %s" % (n1, "같음" if s1 else "다름", n2, "같음" if s2 else "다름"))
ann = lines("bx_train_b15.log", "[범주 규칙 맥락] catrule")
cnt = one("bx_train_b15.log", r"\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 X 켬 (\d+)")
cchk = [tuple(int(v) for v in re.search(r"켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · 시행 끝 두 맥락 발화 (\d+)", ln).groups()) for ln in lines("bx_train_b15.log", "[맥락 범주 점검]")]
gchk = [int(re.search(r"활성 무리 (\d+)", ln).group(1)) for ln in lines("bx_train_b15.log", "[E204 맥락 게이팅 점검]")]
gs = one("bx_train_b15.log", r"\[E204 맥락 게이팅 KC\] 시행 (\d+) · .*비\(비활성/활성\) ([0-9.]+)")
cat_x, cat_0 = lines("bx_train_b15.log", "[범주 과제] 시행"), lines("b0_train_b15.log", "[범주 과제] 시행")
s3, _ = same(os.path.join(wd, "w_bx_b15.npz"), os.path.join(wd, "w_b0_b15.npz"))
dev, rng = kcsum(os.path.join(wd, "w_bx_b15.npz"))
b0_ann = lines("b0_train_b15.log", "[범주 규칙 맥락]")
b_ok = (len(ann) == 1 and bool(cnt) and cnt.groups() == ("40", "40") and len(cchk) == 3 and all(a > 0 and b == 0 and c == 0 for a, b, c in cchk)
        and len(gchk) == 3 and all(a > 0 for a in gchk) and bool(gs) and int(gs.group(1)) == 40 and float(gs.group(2)) <= 0.01
        and cat_x == cat_0 and len(cat_x) == 1 and (not s3) and dev <= 0.01 and rng and not b0_ann)
ok &= b_ok
print("[검사 (b)] 맥락 X 고정 — 알림 %d · 켬 %s · 맥락 점검 %s · 게이팅 점검 %s · 요약 %s · 범주 배정 줄 같음 %s · b0 와 가중치 다름 %s · KC 합 최대 |S−300| %.2e · 범위 %s · b0 맥락 줄 %d → %s"
      % (len(ann), cnt.groups() if cnt else None, cchk, gchk, gs.groups() if gs else None, cat_x == cat_0, not s3, dev, rng, len(b0_ann), "통과" if b_ok else "실패"))
m1, k1 = mot("ev_c1_b15.log"); m2, k2 = mot("ev_c2_b15.log")
print("[보고 (c)] 무학습 motor 평균(게이팅·T_G −2800) — 맥락 없음 %.4f(값 %d) · 맥락 X %.4f(값 %d) · E204 기준 m_ref 0.6094·m_G(X·Y) 0.5998" % (m1, k1, m2, k2))
c_ok = k1 == 16 and k2 == 16
ok &= c_ok
sv = one("dv_train_b15.log", r"세계 시드 (\d+)\).*배정 지역 난수 (\d+)"); sl = one("dl_train_b15.log", r"세계 시드 (\d+)\).*배정 지역 난수 (\d+)")
cl = one("dl_train_b15.log", r"catctx — .*지역 난수 (\d+)\)")
ew = one("ev_dn_b15.log", r"\[E204 평가 세계\] 세계 시드 (\d+) · 원형 A ([0-9,]+)")
rcd = lines("ev_dn_b15.log", "RCEVAL ")
cats_d = sorted({re.search(r"cat=([A-D])", ln).group(1) for ln in rcd})
ctxon = all("ctx=X" in ln and re.search(r"ctx_on=(\d+)", ln) and re.search(r"ctx_on=(\d+)", ln).group(1) == "96" for ln in rcd)
eb = sum(1 for ln in open(os.path.join(out, "ev_dn_b15.log"), encoding="utf-8", errors="replace") if "E204 맥락 게이팅 KC] 무리" in ln)
d_ok = (bool(sv and sl and cl and ew) and (int(sv.group(1)), int(sv.group(2)), int(sl.group(1)), int(sl.group(2)), int(cl.group(1)), int(ew.group(1))) == (38300, 38515, 38300, 38515, 38615, 38300)
        and ew.group(2).startswith("5,14,15,19,29") and len(rcd) == 8 and cats_d == ["A", "B", "C", "D"] and ctxon and eb >= 1)
ok &= d_ok
print("[검사 (d)] 새 세계 — catrule 세계·배정 %s · catctx 세계·배정·맥락 %s · 평가 세계·원형 A %s · RCEVAL %d(범주 %s, 맥락 X·ctx_on 96 %s) · 평가 구성 줄 %d → %s"
      % (sv.groups() if sv else None, (sl.groups() if sl else None, cl.group(1) if cl else None), ew.groups() if ew else None, len(rcd), ",".join(cats_d), ctxon, eb, "통과" if d_ok else "실패"))
rv = lines("dv_train_b15.log", "[범주 반전]"); p2 = lines("dp_train_b15.log", "[둘째 범주 쌍]")
cv = one("dv_train_b15.log", r"\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 X 켬 (\d+)"); cp = one("dp_train_b15.log", r"\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 X 켬 (\d+)")
pe = one("ev_el_b15.log", r"pushed=(\d+)"); rce = lines("ev_el_b15.log", "RCEVAL ")
e_ok = (len(rv) == 2 and "반전 시행 20 (시행 20 부터, 전체 40)" in rv[1] and len(p2) == 2 and "둘째 쌍 시행 20 (시행 20 부터, 전체 40)" in p2[1]
        and bool(cv and cp) and cv.groups() == ("40", "40") and cp.groups() == ("40", "40") and bool(pe) and pe.group(1) == "8" and len(rce) == 4)
ok &= e_ok
print("[검사 (e)] 반전·둘째 쌍 + 맥락 X — 반전 %s · 둘째 쌍 %s · 켬 %s·%s · 학습 평가 pushed %s·RCEVAL %d → %s"
      % (rv, p2, cv.groups() if cv else None, cp.groups() if cp else None, pe.group(1) if pe else None, len(rce), "통과" if e_ok else "실패"))
print("[E205 경로 검사] %s" % ("통과" if ok else "실패"))
PY
echo "[E205 경로 검사] 종료 $(date '+%F %T')"
