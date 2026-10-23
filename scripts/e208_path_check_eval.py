#!/usr/bin/env python3
"""E208 경로 검사 판정(GPU 실행 뒤 — 로그만 읽음). scripts/e208_path_check.sh 의 마지막 점검을 떼어 낸 것 + 정정 1((c) 게이팅 점검의 기준을 흥분 20 측정의 활성 무리 발화로).
사용: python3 scripts/e208_path_check_eval.py <R> <OUT> <WD> <SN15> <W6L> <TF>  — 출력 끝 줄 '[E208 경로 검사] 통과|실패'."""
import os, re, sys
import numpy as np
R, out, wd, sn, w6l, tf = sys.argv[1:7]; tf = int(tf)
sys.path.insert(0, os.path.join(R, "scripts"))
from judge_e193 import read_eval
import judge_e206 as J6
import judge_e207 as J7
import judge_e208 as J8
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
# (a) 회귀 — E206 경로 검사 (l) 커맨드 재현(가중치 4배열 같음)
dA, d6a = np.load(os.path.join(wd, "w_a20_b15.npz")), np.load(w6l)
a_ok = all(np.array_equal(dA[k], d6a[k]) for k in P4)
ok &= a_ok
print("[검사 (a)] 회귀 — E206 경로 검사 (l) 커맨드(흥분 20·T −1300) 가중치 4배열 = E206 w_l_b15: %s → %s" % (a_ok, "통과" if a_ok else "실패"))
# (m) 평가 경로 회귀 — 기준 m_ref 의 32값(로그 문자열)이 E206 보정 로그 m_F_1300 과 같음
def mvals(path):
    v = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("RCEVAL "):
            v += [x for ab in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", ln) for x in ab]
    return v
m6dir = os.path.join(R, "research", "experiments", "logs", "E206", "pathcheck")
mref = [mvals(L("m_ref_%s_b15.log" % X)) for X in "XY"]
m6 = [mvals(os.path.join(m6dir, "m_F_1300_%s_b15.log" % X)) for X in "XY"]
mr_ok = all(len(a) == 16 for a in mref) and mref == m6
ok &= mr_ok
print("[검사 (m) 회귀] 기준 m_ref 32값 = E206 보정 로그 m_F_1300(X·Y): %s → %s" % (mr_ok, "통과" if mr_ok else "실패"))
# (c) 맥락 흥분 0 — 기준 연결 수는 CW20 측정 로그
c20 = J8.read_ctxw(L("c20_meas_b15.log")); want = {}
for pop, _, w, p, nl, nr in c20:
    want.setdefault(pop, set()).add((w, nl, nr))
want_ok = set(want) == {"zz_ctx", "zz_ctx2"} and all(len(v) == 1 and next(iter(v))[0] == 20.0 for v in want.values())
WANT = {k: next(iter(v))[1:] for k, v in want.items()} if want_ok else {}
why = []
c_ok = want_ok and J8.check_ctxw("(c) c0", L("c0_meas_b15.log"), WANT, why)
r0, r20 = J7.read_meas(L("c0_meas_b15.log")), J7.read_meas(L("c20_meas_b15.log"))
for nm_, r in (("c0", r0), ("c20", r20)):
    cx = r["ctx"]
    # [정정 1] 게이팅 점검 — 비활성 무리 KC 발화 ≤ 흥분 20 측정(c20, 같은 tag)의 활성 무리 발화 × 1%(흥분 0 은 활성 무리 발화를 줄여 '비' 의 분모가 조작으로 바뀜)
    ref = r20["ctx"]
    k_ = (set(cx) == {"pre", "post"} and set(ref) == {"pre", "post"}
          and all(v and v[0] == "X" and v[1] == 288 and v[2] > 0 and v[3] == 0 and v[6] == "있음" and v[4] > 0 and ref[t] and 100 * v[5] <= ref[t][4] for t, v in cx.items())
          and len(r["rcj"]) == 12 and r["expo"] == 0)
    c_ok = c_ok and k_
    print("[검사 (c) %s] 측정 맥락 %s · RCJ %d · 노출 %s · S 첫/둘째 %d/%d · J(A,B) 첫 좌/우 %d/%d → %s" % (nm_, cx, len(r["rcj"]), r["expo"], J7.score(r, 0), J7.score(r, 1),
          r["rcj"][("l", "AB")][0], r["rcj"][("r", "AB")][0], "통과" if k_ else "실패"))
same_c = [r0["rcj"][k][0] for k in sorted(r0["rcj"])] == [r20["rcj"][k][0] for k in sorted(r20["rcj"])]
c_ok = c_ok and not same_c
ok &= c_ok
print("[검사 (c)] 맥락 흥분 무게 — 기준(CW20) 연결 %s · CW0 줄 %s · 같은 값(CW0 = CW20 RCJ) %s · 사유 %s → %s" % (WANT, J8.read_ctxw(L("c0_meas_b15.log"))[:2], same_c, why, "통과" if c_ok else "실패"))
print("[보고 (c)] 흥분 0 의 맥락 X 겹침 S 첫/둘째 %d/%d 대 흥분 20 %d/%d 대 맥락 없음(E207 경로 검사 (c) F15 none) 2740/1376" % (J7.score(r0, 0), J7.score(r0, 1), J7.score(r20, 0), J7.score(r20, 1)))
# (m) 은 위 출력(calib_tonic.txt)
print("[보고 (m)] " + open(os.path.join(R, "research/experiments/logs/E208/calib_tonic.txt")).read().strip())
def cw_ok(fn):
    w_ = []
    return J8.check_ctxw(fn, L(fn), WANT, w_), w_
# (l)
gs = one("l_train_b15.log", r"\[E204 맥락 게이팅 KC\] 시행 (\d+) · .*비\(비활성/활성\) ([0-9.]+)")
gchk = [int(re.search(r"활성 무리 (\d+)", ln).group(1)) for ln in lines("l_train_b15.log", "[E204 맥락 게이팅 점검]")]
cchk = [tuple(int(v) for v in re.search(r"켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · 시행 끝 두 맥락 발화 (\d+)", ln).groups()) for ln in lines("l_train_b15.log", "[맥락 범주 점검]")]
sl = one("l_train_b15.log", r"세계 시드 (\d+)\).*배정 지역 난수 (\d+)"); cl = one("l_train_b15.log", r"catctx — .*지역 난수 (\d+)\)")
dW = np.load(os.path.join(wd, "w_l_b15.npz")); d6 = np.load(w6l)
diff = not all(np.array_equal(dW[k], d6[k]) for k in P4)
dev, rng = kcsum(os.path.join(wd, "w_l_b15.npz"))
fo, nf, tn = fl_ok("l_train_b15.log")
cwo, cwy = cw_ok("l_train_b15.log")
l_ok = (bool(gs) and gs.group(1) == "40" and float(gs.group(2)) <= 0.01 and len(gchk) == 3 and all(a > 0 for a in gchk) and len(cchk) == 3 and all(a > 0 and b == 0 and c == 0 for a, b, c in cchk)
        and bool(sl and cl) and (sl.group(1), sl.group(2), cl.group(1)) == ("38300", "38515", "38615") and diff and dev <= 0.01 and rng and fo and cwo)
ok &= l_ok
print("[검사 (l)] catctx F15·흥분 0 — 게이팅 요약 %s · 점검 %s · 맥락 점검 %s · 시드 %s/%s · E206 경로 검사 (l)(흥분 20)과 가중치 다름 %s · KC 합 최대 |S−300| %.2e · 범위 %s · F 적재 %d·동작점 %s · 맥락 무게 %s %s → %s"
      % (gs.groups() if gs else None, gchk, cchk, sl.groups() if sl else None, cl.group(1) if cl else None, diff, dev, rng, nf, tn, cwo, cwy, "통과" if l_ok else "실패"))
# (r)
r_ok = True
for N, ad in (("rx", None), ("rv", "rev"), ("rp", "p2")):
    cnt = one("%s_train_b15.log" % N, r"\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 X 켬 (\d+)")
    rv, p2 = lines("%s_train_b15.log" % N, "[범주 반전]"), lines("%s_train_b15.log" % N, "[둘째 범주 쌍]")
    fo, nf, tn = fl_ok("%s_train_b15.log" % N)
    cwo, _ = cw_ok("%s_train_b15.log" % N)
    want_rv = 2 if ad == "rev" else 0; want_p2 = 2 if ad == "p2" else 0
    k_ok = bool(cnt) and cnt.groups() == ("40", "40") and len(rv) == want_rv and len(p2) == want_p2 and fo and cwo
    if ad == "rev": k_ok = k_ok and "반전 시행 20 (시행 20 부터, 전체 40)" in rv[1]
    if ad == "p2": k_ok = k_ok and "둘째 쌍 시행 20 (시행 20 부터, 전체 40)" in p2[1]
    r_ok &= k_ok
    print("[검사 (r) %s] 켬 %s · 반전 %d · 둘째 쌍 %d · F 적재 %d·동작점 %s · 맥락 무게 %s → %s" % (N, cnt.groups() if cnt else None, len(rv), len(p2), nf, tn, cwo, "통과" if k_ok else "실패"))
ok &= r_ok
# (e)
e_ok = True
for N, ctx, cats, pw in (("el", "X", "AB", 8), ("en", "Y", "AB", 0), ("ev", "X", "ABCD", 8), ("e0", "X", "ABCD", 0)):
    rc, pushed, kt, fw, sat, bad = read_eval(L("ev_%s_b15.log" % N), ctx)
    ew = one("ev_%s_b15.log" % N, r"\[E204 평가 세계\] 세계 시드 (\d+) · 원형 A ([0-9,]+)")
    fo, nf, tn = fl_ok("ev_%s_b15.log" % N)
    cwo, _ = cw_ok("ev_%s_b15.log" % N)
    k_ok = (set(rc) == {(c, s) for c in cats for s in ("train", "new")} and not bad and not sat and kt >= 1 and fw >= 1 and pushed == pw
            and bool(ew) and ew.group(1) == "38300" and ew.group(2).startswith("5,14,15,19,29") and fo and cwo)
    e_ok &= k_ok
    mm = J6.J5.motor_mean(L("ev_%s_b15.log" % N))
    print("[검사 (e) %s] RCEVAL %d · 나쁨 %s · 포화 %s · 종류 %d · 특징 %d · pushed %s · 세계 %s · F 적재 %d·동작점 %s · 맥락 무게 %s · motor 평균 %.4f → %s"
          % (N, len(rc), bad, sat, kt, fw, pushed, ew.group(1) if ew else None, nf, tn, cwo, mm, "통과" if k_ok else "실패"))
ok &= e_ok
print("[E208 경로 검사] %s" % ("통과" if ok else "실패"))
