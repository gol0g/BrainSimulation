#!/bin/bash
# E206 경로 검사(조건 2) — 표본 밖 뇌 15. 새 코드 없음 — 새 것은 조합(F --feat-weights + 맥락 집단 C2 + 게이팅 + T_F)뿐.
# (f)  형성: E191 형성 커맨드 + --rc-seed 38300(측정 가중치 WB = E193 경로 검사 w_b15) → F15 = traces/E206/pathcheck/feat_formed_b15.npz — 세계 줄·제시 816·RCJ 12·겹침 J_post(AC·BD ≤ 0.10)
# (f2) 무관성: 같은 형성을 측정 가중치만 traces/E205/pathcheck/w_bx_b15.npz 로 → 저장 가중치 배열·RCJ 줄이 (f) 와 같음
# (m)  동작점 규칙(criteria_fixed.txt): 무학습 평가 motor 평균(맥락 X·Y × A·B × train·new × 두 쪽 × L/R = 32값) U·F15 × 게이팅·−2800 —
#      |m_F − m_U| ≤ 0.0170 이면 T_F −2800, 아니면 F 격자 −4900 … −2200(300 간격)에서 |m_F(T) − m_U(−2800)| 최소(같으면 −2800 에 가까운 쪽) → logs/E206/calib_tonic.txt
# (l)  catctx 40(2 × 20) F15 + C2 + 게이팅 + T_F + 세계 38300·env 2·배정 38515·맥락 38615 — F 적재·동작점·게이팅 점검·맥락 점검·가중치(범위·KC 합)·E205 경로 검사 dl(같은 시드 U)과 다름
# (r)  catrule 맥락 X + F15 40 · 반전(20 부터) · 둘째 쌍(20 부터) — 맥락 켬 40·반전·둘째 쌍 줄·F 적재·동작점
# (e)  평가(F15·T_F·세계 38300): (l) 가중치 맥락 X(A·B)·무학습 맥락 Y(A·B)·반전 가중치 맥락 X(A~D)·무학습 맥락 X(A~D) — pushed·RCEVAL 수·ctx_on 96·F 적재·동작점·평가 세계·원형
#      + 판정기 함수(judge_e191 form_ok·judge_e193 read_eval·judge_e206 read_feat·read_tonic)를 이 로그들에 한 번(형식 확인)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E206/pathcheck"; WD="$R/research/experiments/traces/E206/pathcheck"; mkdir -p "$OUT" "$WD"
CAL="$R/research/experiments/logs/E206/calib_tonic.txt"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
WB2="$R/research/experiments/traces/E205/pathcheck/w_bx_b15.npz"
WDL="$R/research/experiments/traces/E205/pathcheck/w_dl_b15.npz"
F15="$WD/feat_formed_b15.npz"; F15B="$WD/feat_formed_b15_wb2.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e206_run && cd /root/e206_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
B0="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
LRN="--kc-motor-hard --kc-motor-comp --da-baseline"
for f_ in "$KW15" "$SN15" "$WB" "$WB2" "$WDL"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E206 경로 검사] 종료"; exit 1; }; done
echo "[E206 경로 검사] 시작 $(date '+%F %T')"
# (f)·(f2) 형성 두 번(측정 가중치만 다름)
for spec in "f $WB $F15" "f2 $WB2 $F15B"; do
  set -- $spec; N=$1; W=$2; FO=$3
  g="$OUT/form_${N}_b15.log"
  if [ -s "$FO" ] && grep -q '^=> RAWCAT ' "$g" 2>/dev/null; then echo "[($N) 형성 — 이미 있음, 건너뜀]"; continue; fi
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT --feat-oja --feat-oja-eta 0 \
    --decomp-weights $W --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta 0.02 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300 --rc-save $FO > "$g" 2>&1; rc=$?
  echo "[($N) 형성 rc=$rc $(date '+%T')] $(grep '^\[원시 특징 세계\]' "$g" | cut -c1-110) | $(grep '^\[원시 특징 노출\]' "$g" | grep -oE '제시 [0-9]+') | RCJ $(grep -c '^RCJ ' "$g")"
  grep -E '^RCJ side=[lr] pair=(AC|BD)' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
# (m) 동작점 — U·F × 맥락 X·Y, 게이팅·−2800
mrun() {   # $1 이름 $2 F 여부(0/1) $3 기저값 $4 맥락
  local g="$OUT/m_$1_$4_b15.log" ex=""
  [ "$2" = "1" ] && ex="--feat-weights $F15"
  if [ "$(grep -c '^RCEVAL ' "$g" 2>/dev/null)" = "4" ]; then return; fi
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE $ex --motor-tonic $3 --brain-seed 15 --kc-type-weights $KW15 \
    --decomp-weights $WB --decomp-mode none --rc-eval --rc-eval-ctx $4 --rc-seed 38300 --trials 96 --eval-diag > "$g" 2>&1
  echo "  (m $1 $4) RCEVAL $(grep -c '^RCEVAL ' "$g") | 특징 적재 $(grep -c '^\[E184 특징 가중치 적재\]' "$g") | 동작점 $(grep -oE '기저값 -?[0-9.]+' "$g" | sort -u | tr '\n' ' ')"
}
for X in X Y; do mrun U_2800 0 -2800 $X; mrun F_2800 1 -2800 $X; done
mpick() {   # $1 = first|grid — 규칙 적용, CAL 쓰기 또는 NEED_GRID 출력
  python3 - "$OUT" "$CAL" "$1" <<'PY'
import os, re, sys
out, cal, mode = sys.argv[1:4]
def mean_of(V):
    v = []
    for X in "XY":
        for ln in open(os.path.join(out, "m_%s_%s_b15.log" % (V, X)), encoding="utf-8", errors="replace"):
            if ln.startswith("RCEVAL "):
                for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", ln):
                    v += [float(a), float(b)]
    if len(v) != 32:
        raise SystemExit("[실패] %s motor 값 수 %d(32 이어야)" % (V, len(v)))
    return sum(v) / len(v)
mu, mf = mean_of("U_2800"), mean_of("F_2800")
print("  m_U(−2800) %.4f · m_F(−2800) %.4f · |차| %.4f(규칙 문턱 0.0170)" % (mu, mf, abs(mf - mu)))
if mode == "first":
    if abs(mf - mu) <= 0.0170:
        line = "T_F=-2800 m_U=%.4f m_F=%.4f diff=%.4f (규칙: |m_F − m_U| ≤ 0.0170 → −2800 그대로)" % (mu, mf, abs(mf - mu))
        open(cal, "w").write(line + "\n"); print("[동작점 규칙] " + line)
    else:
        print("NEED_GRID")
else:
    cand = []
    for T in (4900, 4600, 4300, 4000, 3700, 3400, 3100, 2800, 2500, 2200):
        m = mean_of("F_%d" % T)
        cand.append((abs(m - mu), abs(T - 2800), T, m))
        print("  후보 −%d: m_F %.4f · |m_F − m_U| %.4f" % (T, m, abs(m - mu)))
    d, _, T, m = min(cand)
    line = "T_F=-%d m_U=%.4f m_F=%.4f diff=%.4f (규칙: 격자 최소 — −2800 에서 |차| %.4f > 0.0170)" % (T, mu, m, d, abs(mf - mu))
    open(cal, "w").write(line + "\n"); print("[동작점 규칙] " + line)
PY
}
if [ -s "$CAL" ]; then
  echo "[동작점 규칙] 이미 적용됨(다시 적용하지 않음 — 결과 뒤 조정 없음): $(cat "$CAL")"
else
  mpick first > "$OUT/mpick_first.out" 2>&1; cat "$OUT/mpick_first.out"
  if grep -q NEED_GRID "$OUT/mpick_first.out"; then
    for T in 4900 4600 4300 4000 3700 3400 3100 2500 2200; do for X in X Y; do mrun F_$T 1 -$T $X; done; done
    mpick grid
  fi
fi
TF=$(grep -oE '^T_F=-[0-9]+' "$CAL" 2>/dev/null | cut -d= -f2)
[ -n "$TF" ] || { echo "[실패] T_F 를 정하지 못함"; echo "[E206 경로 검사] 종료"; exit 1; }
echo "[동작점] T_F $TF"
# (l)·(r) 학습 — F15 + 맥락 + 게이팅 + T_F + 새 세계·시행열
for spec in "l catctx -" "rx catrule -" "rv catrule rev" "rp catrule p2"; do
  set -- $spec; N=$1; TK=$2; AD=$3
  EXTRA="--rc-seed 38300 --rc-trial-seed 38515"
  [ "$TK" = "catctx" ] && EXTRA="$EXTRA --rc-ctx-seed 38615" || EXTRA="$EXTRA --rc-rule-ctx X"
  [ "$AD" = "rev" ] && EXTRA="$EXTRA --rc-reverse-after 20"; [ "$AD" = "p2" ] && EXTRA="$EXTRA --rc-phase2-after 20"
  f="$OUT/${N}_train_b15.log"
  timeout 1800 python reflex_override_task.py $B0 --kc-motor-eta 2.4 --env-seed 2 $ACT $FEAT --feat-weights $F15 $CW $GATE --motor-tonic $TF --rc-task $TK --episodes 2 --steps 20 \
    --transplant-eval --brain-seed 15 --kc-type-weights $KW15 --save-weights $WD/w_${N}_b15.npz $LRN $EXTRA > "$f" 2>&1; rc=$?
  echo "[($N) $TK $AD rc=$rc $(date '+%T')] $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$f" | cut -c1-120) | $(grep '^\[범주 규칙 맥락\] 시행' "$f") | F 적재 $(grep -c '^\[E184 특징 가중치 적재\].* 검증 일치' "$f") | 동작점 $(grep -oE '기저값 -?[0-9.]+' "$f" | sort -u | tr '\n' ' ')"
  grep '^\[맥락 범주 점검\]\|^\[E204 맥락 게이팅 점검\]\|^\[범주 과제\] catrule\|^\[범주 과제\] 시행\|^\[맥락 범주 과제\]\|^\[범주 반전\]\|^\[둘째 범주 쌍\]' "$f" | cut -c1-170 | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$f" | sed 's/^/      /'
done
# (e) 평가
for spec in "el all $WD/w_l_b15.npz X A,B" "en none $WD/w_l_b15.npz Y A,B" "ev all $WD/w_rv_b15.npz X A,B,C,D" "e0 none $WD/w_rv_b15.npz X A,B,C,D"; do
  set -- $spec; N=$1; XM=$2; WF=$3; X=$4; CATS=$5
  g="$OUT/ev_${N}_b15.log"
  timeout 1800 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $F15 $CW $GATE --motor-tonic $TF --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WF --decomp-mode $XM \
    --rc-eval --rc-eval-cats $CATS --rc-eval-ctx $X --rc-seed 38300 --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[(ev $N) $XM 맥락 $X 범주 $CATS rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep -oE 'pushed=[0-9]+' "$g" | head -1) | $(grep '^\[E204 평가 세계\]' "$g" | cut -c1-80) | F 적재 $(grep -c '^\[E184 특징 가중치 적재\].* 검증 일치' "$g") | 동작점 $(grep -oE '기저값 -?[0-9.]+' "$g" | sort -u | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
python3 - "$R" "$OUT" "$WD" "$SN15" "$WDL" "$TF" <<'PY'
import os, re, sys
import numpy as np
R, out, wd, sn, wdl, tf = sys.argv[1:7]; tf = int(tf)
sys.path.insert(0, os.path.join(R, "scripts"))
import judge_e191 as J1
from judge_e193 import read_eval
import judge_e206 as J6
P4 = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")
def L(fn): return os.path.join(out, fn)
def lines(fn, pref): return [ln.rstrip("\n") for ln in open(L(fn), encoding="utf-8", errors="replace") if ln.startswith(pref)]
def one(fn, pat):
    for ln in open(L(fn), encoding="utf-8", errors="replace"):
        m = re.search(pat, ln)
        if m: return m
    return None
def fl_ok(fn):
    fl, tn = J6.read_feat(L(fn)), J6.read_tonic(L(fn))
    return bool(fl) and all(p is not None and p.endswith("traces/E206/pathcheck/feat_formed_b15.npz") for p in fl) and bool(tn) and all(t == float(tf) for t in tn), len(fl), sorted(set(tn))
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
ok = True
# (f)
fm = J1.read_form(L("form_f_b15.log"))
w_ = one("form_f_b15.log", r"^\[원시 특징 세계\] .*seed (\d+) \| 원형 A ([0-9]+(?:,[0-9]+){4})")
jp = {(s, p): fm["rcj"].get((s, p)) for s in "lr" for p in ("AC", "BD")}
f_ok = J1.form_ok(fm) and bool(w_) and w_.group(1) == "38300" and w_.group(2) == "5,14,15,19,29" and all(v and v[1] <= 1000 for v in jp.values()) and os.path.getsize(os.path.join(wd, "feat_formed_b15.npz")) > 0
ok &= f_ok
print("[검사 (f)] 형성 — form_ok %s · 세계 %s · 겹침 J_pre/J_post(1e-4) %s → %s" % (J1.form_ok(fm), w_.groups() if w_ else None, jp, "통과" if f_ok else "실패"))
# (f2)
A, B = np.load(os.path.join(wd, "feat_formed_b15.npz")), np.load(os.path.join(wd, "feat_formed_b15_wb2.npz"))
same_w = sorted(A.files) == sorted(B.files) and all(np.array_equal(A[k], B[k]) for k in A.files)
same_j = lines("form_f_b15.log", "RCJ ") == lines("form_f2_b15.log", "RCJ ")
f2_ok = same_w and same_j and len(lines("form_f_b15.log", "RCJ ")) == 12
ok &= f2_ok
print("[검사 (f2)] 측정 가중치 무관성 — 저장 배열 %d개 같음 %s · RCJ 줄 같음 %s → %s" % (len(A.files), same_w, same_j, "통과" if f2_ok else "실패"))
# (m) 은 위 출력(calib_tonic.txt)
print("[보고 (m)] " + open(os.path.join(R, "research/experiments/logs/E206/calib_tonic.txt")).read().strip())
# (l)
gs = one("l_train_b15.log", r"\[E204 맥락 게이팅 KC\] 시행 (\d+) · .*비\(비활성/활성\) ([0-9.]+)")
gchk = [int(re.search(r"활성 무리 (\d+)", ln).group(1)) for ln in lines("l_train_b15.log", "[E204 맥락 게이팅 점검]")]
cchk = [tuple(int(v) for v in re.search(r"켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · 시행 끝 두 맥락 발화 (\d+)", ln).groups()) for ln in lines("l_train_b15.log", "[맥락 범주 점검]")]
sl = one("l_train_b15.log", r"세계 시드 (\d+)\).*배정 지역 난수 (\d+)"); cl = one("l_train_b15.log", r"catctx — .*지역 난수 (\d+)\)")
dW = np.load(os.path.join(wd, "w_l_b15.npz")); dU = np.load(wdl)
diff = not all(np.array_equal(dW[k], dU[k]) for k in P4)
dev, rng = kcsum(os.path.join(wd, "w_l_b15.npz"))
fo, nf, tn = fl_ok("l_train_b15.log")
l_ok = (bool(gs) and gs.group(1) == "40" and float(gs.group(2)) <= 0.01 and len(gchk) == 3 and all(a > 0 for a in gchk) and len(cchk) == 3 and all(a > 0 and b == 0 and c == 0 for a, b, c in cchk)
        and bool(sl and cl) and (sl.group(1), sl.group(2), cl.group(1)) == ("38300", "38515", "38615") and diff and dev <= 0.01 and rng and fo)
ok &= l_ok
print("[검사 (l)] catctx F — 게이팅 요약 %s · 점검 %s · 맥락 점검 %s · 시드 %s/%s · E205 dl(U)과 가중치 다름 %s · KC 합 최대 |S−300| %.2e · 범위 %s · F 적재 %d·동작점 %s → %s"
      % (gs.groups() if gs else None, gchk, cchk, sl.groups() if sl else None, cl.group(1) if cl else None, diff, dev, rng, nf, tn, "통과" if l_ok else "실패"))
# (r)
r_ok = True
for N, ad in (("rx", None), ("rv", "rev"), ("rp", "p2")):
    cnt = one("%s_train_b15.log" % N, r"\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 X 켬 (\d+)")
    rv, p2 = lines("%s_train_b15.log" % N, "[범주 반전]"), lines("%s_train_b15.log" % N, "[둘째 범주 쌍]")
    fo, nf, tn = fl_ok("%s_train_b15.log" % N)
    want_rv = 2 if ad == "rev" else 0; want_p2 = 2 if ad == "p2" else 0
    k_ok = bool(cnt) and cnt.groups() == ("40", "40") and len(rv) == want_rv and len(p2) == want_p2 and fo
    if ad == "rev": k_ok = k_ok and "반전 시행 20 (시행 20 부터, 전체 40)" in rv[1]
    if ad == "p2": k_ok = k_ok and "둘째 쌍 시행 20 (시행 20 부터, 전체 40)" in p2[1]
    r_ok &= k_ok
    print("[검사 (r) %s] 켬 %s · 반전 %d · 둘째 쌍 %d · F 적재 %d·동작점 %s → %s" % (N, cnt.groups() if cnt else None, len(rv), len(p2), nf, tn, "통과" if k_ok else "실패"))
ok &= r_ok
# (e)
e_ok = True
for N, ctx, cats, pw in (("el", "X", "AB", 8), ("en", "Y", "AB", 0), ("ev", "X", "ABCD", 8), ("e0", "X", "ABCD", 0)):
    rc, pushed, kt, fw, sat, bad = read_eval(L("ev_%s_b15.log" % N), ctx)
    ew = one("ev_%s_b15.log" % N, r"\[E204 평가 세계\] 세계 시드 (\d+) · 원형 A ([0-9,]+)")
    fo, nf, tn = fl_ok("ev_%s_b15.log" % N)
    k_ok = (set(rc) == {(c, s) for c in cats for s in ("train", "new")} and not bad and not sat and kt >= 1 and fw >= 1 and pushed == pw
            and bool(ew) and ew.group(1) == "38300" and ew.group(2).startswith("5,14,15,19,29") and fo)
    e_ok &= k_ok
    mm = J6.J5.motor_mean(L("ev_%s_b15.log" % N))
    print("[검사 (e) %s] RCEVAL %d · 나쁨 %s · 포화 %s · 종류 %d · 특징 %d · pushed %s · 세계 %s · F 적재 %d·동작점 %s · motor 평균 %.4f → %s"
          % (N, len(rc), bad, sat, kt, fw, pushed, ew.group(1) if ew else None, nf, tn, mm, "통과" if k_ok else "실패"))
ok &= e_ok
print("[E206 경로 검사] %s" % ("통과" if ok else "실패"))
PY
echo "[E206 경로 검사] 종료 $(date '+%F %T')"
