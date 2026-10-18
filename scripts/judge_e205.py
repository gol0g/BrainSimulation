#!/usr/bin/env python3
"""E205 판정 — L4 맥락 게이팅 확증(새 뇌 16~20·세계 38300·새 시행열: 게이팅 GW 대 게이팅 없음 UW) + 같은 고정 체계(게이팅·T_G·맥락 X 고정)의
앞 능력(R 범주 규칙·V 반전·P 유지). 규칙 = research/experiments/logs/E205/criteria_fixed.txt.
L4 범주는 E193~E204 와 같고, 앞 능력 정의는 E185(개념 수준)·E188(반전 성공)·E190(유지) 원문 그대로(1e-4 정수, e = 학습 − 무학습, A 교차(−)·B 같은 쪽(+)·C 같은 쪽(+)·D 교차(−)).
입력: logs/E205/train_{GW|UW|R|V|P}_b{B}.log, ev_{GW|UW}_b{B}_{learn|none}_{X|Y}.log, ev_b{B}_{R_learn|V_learn|P_learn|none}.log,
  traces/E205/w_{GW|UW|R|V|P}_b{B}.npz, 스냅숏 traces/E176/snap/conn_b{B}.npz.
조작검증(하나라도 어기면 보류): L4 — E204 와 같은 검사(학습·특징·맥락 점검·규칙·기준선·시드·게이팅·가중치·평가·평가 세계) + 평가 원형 A 앞 5개;
  앞 능력 — 과제 줄(시행 수·범주 배정 범위·반전/둘째 쌍 줄 유무)·특징 점검·맥락 점검 3줄·맥락 고정 알림 1·'맥락 X 켬 N'·규칙 줄·기준선 줄·시드(세계·배정)·
  게이팅(구성 ≥ 2·점검 3 활성 > 0·요약 시행 N·비 ≤ 0.01)·가중치(범위·KC 별 합·V·P 가 R 과 다름)·평가(범주 집합·맥락 X ctx_on 96·비포화·종류·특징 0·pushed·세계·원형·게이팅 구성).
판정: L4 = 확증(GW 범주 ≥ 4/5·m_GW − m_UW ≥ 500 ≥ 4/5) / 체계 확증(게이팅 효과 미확증) / 미확증. 앞 능력 = 능력마다 보존·부분·잃음(유지는 기준·전제 미달이면 보류).
  종합: 조작검증 실패 → 보류 / L4 확증·셋 다 보존 → 통합 관문 통과 / 그 밖 → 통합 관문 미통과(판정 줄에 각 판정).
실행: python3 scripts/judge_e205.py (저장소 루트에서)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge_e193 import CELLS, read_eval, read_train  # noqa: E402
import judge_e199 as J9  # noqa: E402 — read_comp·weight_checks
import judge_e204 as J4  # noqa: E402 — read_da·read_gate·read_seeds·cells·category·weakest

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
WORLD, TRIAL0, CTX0 = 38300, 38500, 38600
PROTO_A = "5,14,15,19,29"
L4 = ("GW", "UW")
GATED = {"GW": True, "UW": False}
NL4 = 3000
TASKS = {"R": 1500, "V": 3000, "P": 3000}
BRANGE = {"R": (600, 900), "V": (1200, 1800), "P": (1200, 1800)}
EVALS = (("R_learn", "R", "ABCD", 8), ("V_learn", "V", "AB", 8), ("P_learn", "P", "ABCD", 8), ("none", None, "ABCD", 0))
ETA = 2.4
RMAX = 0.01
DMIN = 500


def read_rule(path):
    """맥락 고정·반전·둘째 쌍 줄 — 알림 [(맥락)], 요약 (시행, 맥락, 켬), 반전 줄들, 둘째 쌍 줄들."""
    r = {"ann": [], "cnt": None, "rev": [], "p2": []}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[범주 규칙 맥락] catrule"):
            m = re.search(r"맥락 ([XY])\(", ln)
            r["ann"].append(m.group(1) if m else "?")
        elif ln.startswith("[범주 규칙 맥락] 시행 "):
            m = re.search(r"시행 (\d+) 중 맥락 ([XY]) 켬 (\d+)", ln)
            r["cnt"] = (int(m.group(1)), m.group(2), int(m.group(3))) if m else ("?",)
        elif ln.startswith("[범주 반전] "):
            r["rev"].append(ln.strip())
        elif ln.startswith("[둘째 범주 쌍] "):
            r["p2"].append(ln.strip())
    return r


def read_eval_proto(path):
    """평가 로그 '[E204 평가 세계] 세계 시드 W · 원형 A a,b,c,d,e …' → [(W, 'a,b,c,d,e')]."""
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[E204 평가 세계] 세계 시드 "):
            m = re.search(r"세계 시드 (\d+) · 원형 A ([0-9]+(?:,[0-9]+){4})", ln)
            out.append((int(m.group(1)), m.group(2)) if m else (-1, ""))
    return out


def motor_mean(path):
    v = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("RCEVAL "):
            for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", ln):
                v += [float(a), float(b)]
    return sum(v) / len(v) if v else float("nan")


def check_common(tag, tp, n, gated, why):
    """규칙·기준선·게이팅·시드(세계·배정) 공통 검사. 반환 (ok, 기준선 stat, 게이팅 요약, 시드)."""
    ok = True
    cp, da, gt, sd = J9.read_comp(tp), J4.read_da(tp), J4.read_gate(tp), J4.read_seeds(tp)
    cm = cp["comp"]
    if not (cp["hard"] and cm is not None and len(cm) == 3 and cm[0] == n and cm[1] <= 1e-3 and cm[2] == 1
            and cp["eta"] is not None and abs(cp["eta"] - ETA) <= 1e-9):
        ok = False; why.append("%s 규칙 줄(hard %s·정규화 %s·eta %s)" % (tag, cp["hard"], cm, cp["eta"]))
    st = da["stat"]
    if not (st is not None and len(st) == 4 and st[0] == n and abs(st[1]) <= 0.05 and st[3] == 1 and da["chk"] == 3):
        ok = False; why.append("%s 기준선 줄(%s·점검 %d)" % (tag, st, da["chk"]))
    if gated:
        s_ = gt["sum"]
        if not (gt["build"] >= 2 and len(gt["chk"]) == 3 and all(a_ > 0 for a_, _ in gt["chk"])
                and s_ is not None and len(s_) == 4 and s_[0] == n and s_[1] > 0 and s_[3] <= RMAX):
            ok = False; why.append("%s 게이팅(구성 %d·점검 %s·요약 %s)" % (tag, gt["build"], gt["chk"], s_))
    elif gt["build"] or gt["chk"] or gt["sum"] is not None:
        ok = False; why.append("%s 게이팅 없는 팔에 게이팅 줄" % tag)
    return ok, st, gt["sum"], sd


def check_eval(tag, path, cats, ctx, pushed_want, gated, why):
    """평가 한 개 — 반환 (ok, rc[(범주, 세트)])."""
    ok = True
    rc, pushed, kt, fw, sat, bad = read_eval(path, ctx)
    want = {(c, s) for c in cats for s in ("train", "new")}
    if set(rc) != want or bad or sat or kt < 1 or fw != 0 or pushed != pushed_want:
        ok = False; why.append("%s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s)" % (tag, len(rc), bad, sat, kt, fw, pushed))
    pw = read_eval_proto(path)
    if pw != [(WORLD, PROTO_A)]:
        ok = False; why.append("%s 평가 세계·원형 %s" % (tag, pw))
    nb = J4.read_gate(path)["build"]
    if (gated and nb < 1) or (not gated and nb):
        ok = False; why.append("%s 평가 게이팅 구성 줄 %d" % (tag, nb))
    return ok, rc


def weights_ok(tag, wf, b, other, why):
    lo_hi, dmax, diff, fb, D = J9.weight_checks(wf, os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b), other)
    if not (lo_hi and dmax <= 0.01 and diff):
        why.append("%s 가중치(범위 %s·KC 합 최대 |S−300| %.4g·짝과 다름 %s)" % (tag, lo_hi, dmax, diff))
        return False, fb
    return True, fb


def check_l4(arm, b, why):
    L = os.path.join(EXP, "logs", "E205")
    tag = "%s b%d" % (arm, b)
    tp = os.path.join(L, "train_%s_b%d.log" % (arm, b))
    tr = read_train(tp)
    ok, st, gs, sd = check_common(tag, tp, NL4, GATED[arm], why)
    c, k = tr["ctx"], tr["cat"]
    if not (tr["catctx"] and tr["done"] and tr["kt"] >= 1 and c and k and c[0] == NL4 and c[1] + c[2] == NL4 and c[3] + c[4] == NL4
            and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800 and k[0] == NL4 and 1200 <= k[1] <= 1800):
        ok = False; why.append("%s 학습 줄(%s %s)" % (tag, c, k))
    if len(tr["fchk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["fchk"]):
        ok = False; why.append("%s 특징 점검 %s" % (tag, tr["fchk"]))
    if len(tr["cchk"]) != 3 or not all(x > 0 and y == 0 and z == 0 for x, y, z in tr["cchk"]):
        ok = False; why.append("%s 맥락 점검 %s" % (tag, tr["cchk"]))
    if not (sd["world"] == WORLD and sd["trial"] == TRIAL0 + b and sd["ctx"] == CTX0 + b):
        ok = False; why.append("%s 시드 줄(%s)" % (tag, sd))
    other = "UW" if arm == "GW" else "GW"
    w_ok, fb = weights_ok(tag, os.path.join(EXP, "traces", "E205", "w_%s_b%d.npz" % (arm, b)), b,
                          os.path.join(EXP, "traces", "E205", "w_%s_b%d.npz" % (other, b)), why)
    ok = ok and w_ok
    ev, good = {}, True
    for w in ("learn", "none"):
        for x in "XY":
            e_ok, rc = check_eval("%s %s %s" % (tag, w, x), os.path.join(L, "ev_%s_b%d_%s_%s.log" % (arm, b, w, x)), "AB", x,
                                  8 if w == "learn" else 0, GATED[arm], why)
            ok = ok and e_ok; good = good and e_ok
            ev[(w, x)] = (rc,)
    a = J4.cells(ev) if good else None
    mn = sum(motor_mean(os.path.join(L, "ev_%s_b%d_none_%s.log" % (arm, b, x))) for x in "XY") / 2
    rep = {"ratio": (gs[3] if GATED[arm] and gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb, "motor": mn}
    return ok, a, rep


def check_task(t, b, why):
    L = os.path.join(EXP, "logs", "E205")
    tag = "%s b%d" % (t, b)
    tp = os.path.join(L, "train_%s_b%d.log" % (t, b))
    n = TASKS[t]
    tr = read_train(tp)
    rr = read_rule(tp)
    ok, st, gs, sd = check_common(tag, tp, n, True, why)
    k = tr["cat"]
    lo, hi = BRANGE[t]
    if not (not tr["catctx"] and tr["done"] and tr["kt"] >= 1 and k and k[0] == n and lo <= k[1] <= hi and k[1] + k[2] == n):
        ok = False; why.append("%s 학습 줄(catctx %s·%s)" % (tag, tr["catctx"], k))
    if len(tr["fchk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["fchk"]):
        ok = False; why.append("%s 특징 점검 %s" % (tag, tr["fchk"]))
    if len(tr["cchk"]) != 3 or not all(x > 0 and y == 0 and z == 0 for x, y, z in tr["cchk"]):
        ok = False; why.append("%s 맥락 점검 %s" % (tag, tr["cchk"]))
    if rr["ann"] != ["X"] or rr["cnt"] != (n, "X", n):
        ok = False; why.append("%s 맥락 고정(알림 %s·켬 %s)" % (tag, rr["ann"], rr["cnt"]))
    rev_ok = (rr["rev"] == [] if t != "V" else
              (len(rr["rev"]) == 2 and rr["rev"][0].startswith("[범주 반전] 시행 1500 부터")
               and rr["rev"][1] == "[범주 반전] 반전 시행 1500 (시행 1500 부터, 전체 3000)"))
    p2_ok = (rr["p2"] == [] if t != "P" else
             (len(rr["p2"]) == 2 and rr["p2"][0].startswith("[둘째 범주 쌍] 시행 1500 부터")
              and rr["p2"][1] == "[둘째 범주 쌍] 둘째 쌍 시행 1500 (시행 1500 부터, 전체 3000)"))
    if not (rev_ok and p2_ok):
        ok = False; why.append("%s 반전·둘째 쌍 줄(%s / %s)" % (tag, rr["rev"], rr["p2"]))
    if not (sd["world"] == WORLD and sd["trial"] == TRIAL0 + b and sd["ctx"] is None):
        ok = False; why.append("%s 시드 줄(%s)" % (tag, sd))
    other = os.path.join(EXP, "traces", "E205", "w_%s_b%d.npz" % ("V" if t == "R" else "R", b))
    w_ok, fb = weights_ok(tag, os.path.join(EXP, "traces", "E205", "w_%s_b%d.npz" % (t, b)), b, other, why)
    rep = {"ratio": (gs[3] if gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb}
    return ok and w_ok, rep


def task_evals(b, why):
    L = os.path.join(EXP, "logs", "E205")
    ok, rcs = True, {}
    for nm, _, cats, pw in EVALS:
        e_ok, rc = check_eval("b%d %s" % (b, nm), os.path.join(L, "ev_b%d_%s.log" % (b, nm)), cats, "X", pw, True, why)
        ok = ok and e_ok
        rcs[nm] = rc
    if not ok:
        return False, None, float("nan")
    e = {}
    for nm, t, cats, _ in EVALS:
        if t is None:
            continue
        e[t] = {(c, s): rcs[nm][(c, s)] - rcs["none"][(c, s)] for c in cats for s in ("train", "new")}
    return True, e, motor_mean(os.path.join(L, "ev_b%d_none.log" % b))


def cat_rule(eR):
    """E185 — 개념 수준(확증)·사례 한정·한 범주만·획득 실패."""
    acqA, acqB = eR[("A", "train")] <= -1000, eR[("B", "train")] >= 1000
    gen = (eR[("A", "new")] <= -500 and eR[("B", "new")] >= 500 and 2 * eR[("A", "new")] <= eR[("A", "train")] and 2 * eR[("B", "new")] >= eR[("B", "train")])
    if acqA and acqB:
        return "concept" if gen else "memo"
    if acqA or acqB:
        return "one"
    return "fail"


def cat_rev(eV, eR):
    """E188 — 반전 성공·사례 한정·부분 반전·고착·그 밖."""
    acq = eV[("A", "train")] >= 1000 and eV[("B", "train")] <= -1000
    gen = (eV[("A", "new")] >= 500 and eV[("B", "new")] <= -500 and 2 * eV[("A", "new")] >= eV[("A", "train")] and 2 * eV[("B", "new")] <= eV[("B", "train")])
    dA, dB = eV[("A", "train")] - eR[("A", "train")], eV[("B", "train")] - eR[("B", "train")]
    if acq:
        return "success" if gen else "caseonly"
    if dA >= 1000 and dB <= -1000:
        return "partial"
    if abs(dA) < 500 and abs(dB) < 500:
        return "stuck"
    return "other"


def cat_keep(eP, eR):
    """E190 — 기준(첫 쌍)·전제(둘째 쌍)·유지·간섭. 반환 (base, pre, 범주)."""
    base = eR[("A", "train")] <= -1000 and eR[("B", "train")] >= 1000
    pre = eP[("C", "train")] >= 1000 and eP[("D", "train")] <= -1000
    if not (base and pre):
        return base, pre, "na"
    keep = 2 * eP[("A", "train")] <= eR[("A", "train")] and 2 * eP[("B", "train")] >= eR[("B", "train")]
    return base, pre, ("keep" if keep else "interf")


def status_rule(cats):
    n = lambda c: sum(v == c for v in cats.values())  # noqa: E731
    return "보존" if n("concept") >= 4 else ("잃음" if n("fail") >= 4 else "부분")


def status_rev(cats):
    n = lambda c: sum(v == c for v in cats.values())  # noqa: E731
    return "보존" if n("success") >= 4 else ("잃음" if n("stuck") >= 4 else "부분")


def status_keep(K):
    nb = sum(v[0] for v in K.values()); npre = sum(v[1] for v in K.values())
    if nb < 4:
        return "보류(기준 미달)"
    if npre < 4:
        return "보류(전제 미달)"
    nk = sum(v[2] == "keep" for v in K.values()); ni = sum(v[2] == "interf" for v in K.values())
    return "보존" if nk >= 4 else ("잃음" if ni >= 4 else "부분")


def decide_l4(cat, m):
    gw = sum(cat["GW"][b] == "conc" for b in BRAINS)
    gate = sum(m["GW"][b] - m["UW"][b] >= DMIN for b in BRAINS)
    if gw >= 4:
        return ("확증" if gate >= 4 else "체계 확증(게이팅 효과 미확증)"), gw, gate
    return "미확증", gw, gate


def decide(l4, sr, sv, sp):
    """종합 판정 문자열과 H125·H126 줄."""
    h125 = {"확증": "H125 확증(새 뇌·처음 쓰는 세계)", "체계 확증(게이팅 효과 미확증)": "H125 부분(체계는 서나 새 세계 게이팅 효과 미확증)",
            "미확증": "H125 미확증(새 뇌·새 세계)"}[l4]
    if sr == "보존" and sv == "보존" and sp == "보존":
        h126 = "H126 지지"
    elif sr == "보존" and sv == "보존" and sp == "잃음":
        h126 = "유지만 간섭"
    elif sr == "잃음" or sv == "잃음":
        h126 = "H126-null"
    else:
        h126 = "H126 부분·보류"
    if l4 == "확증" and h126 == "H126 지지":
        v = "통합 관문 통과"
    else:
        v = "통합 관문 미통과 — L4 %s · 범주 규칙 %s · 반전 %s · 유지 %s" % (l4, sr, sv, sp)
    return v, h125, h126


def fmt(a):
    return " ".join("%s·%s %+d/%+d" % (x, cc, a[(x, cc, "train")], a[(x, cc, "new")]) for x, cc, _ in CELLS)


def main():
    ok, why = True, []
    A, RL = {arm: {} for arm in L4}, {arm: {} for arm in L4}
    RT, E, MN = {t: {} for t in TASKS}, {}, {}
    for b in BRAINS:
        try:
            for arm in L4:
                ok_, a, rep = check_l4(arm, b, why)
                ok = ok and ok_
                if a is not None:
                    A[arm][b] = a
                RL[arm][b] = rep
            for t in TASKS:
                ok_, rep = check_task(t, b, why)
                ok = ok and ok_
                RT[t][b] = rep
            ok_, e, mn = task_evals(b, why)
            ok = ok and ok_
            if e is not None:
                E[b] = e
            MN[b] = mn
        except (OSError, ValueError, IndexError, KeyError, TypeError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
    if not ok or any(len(A[arm]) != len(BRAINS) for arm in L4) or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cat = {arm: {b: J4.category(A[arm][b]) for b in BRAINS} for arm in L4}
    m = {arm: {b: J4.weakest(A[arm][b]) for b in BRAINS} for arm in L4}
    for arm in L4:
        for b in BRAINS:
            r_ = RL[arm][b]
            print("%s b%d 규칙 방향 효과(1e-4, 노출/새) %s | 범주 %s · 가장 약한 칸 %+d | 게이팅 비 %s · 평균 DA %+.4f · 경계 시냅스 %.3f · 무학습 motor 평균 %.4f"
                  % (arm, b, fmt(A[arm][b]), cat[arm][b], m[arm][b], ("%.4f" % r_["ratio"]) if r_["ratio"] is not None else "—", r_["mda"], r_["fb"], r_["motor"]))
    CR, CV, K = {}, {}, {}
    for b in BRAINS:
        eR, eV, eP = E[b]["R"], E[b]["V"], E[b]["P"]
        CR[b], CV[b], K[b] = cat_rule(eR), cat_rev(eV, eR), cat_keep(eP, eR)
        rA = eP[("A", "train")] / eR[("A", "train")] if eR[("A", "train")] else float("nan")
        rB = eP[("B", "train")] / eR[("B", "train")] if eR[("B", "train")] else float("nan")
        print("b%d 범주 규칙(R) A %+d/%+d B %+d/%+d → %s | 반전(V) A %+d/%+d B %+d/%+d · Δ A %+d B %+d → %s | 유지(P) A %+d B %+d · C %+d/%+d D %+d/%+d · r_A %.2f r_B %.2f → %s"
              " | (보고) R 의 C %+d·D %+d · 게이팅 비 R %.4f V %.4f P %.4f · 평균 DA R %+.4f V %+.4f P %+.4f · 무학습 motor 평균(맥락 X) %.4f"
              % (b, eR[("A", "train")], eR[("A", "new")], eR[("B", "train")], eR[("B", "new")], CR[b],
                 eV[("A", "train")], eV[("A", "new")], eV[("B", "train")], eV[("B", "new")], eV[("A", "train")] - eR[("A", "train")], eV[("B", "train")] - eR[("B", "train")], CV[b],
                 eP[("A", "train")], eP[("B", "train")], eP[("C", "train")], eP[("C", "new")], eP[("D", "train")], eP[("D", "new")], rA, rB,
                 K[b][2] if K[b][2] != "na" else ("기준 미달" if not K[b][0] else "전제 미달"),
                 eR[("C", "train")], eR[("D", "train")], RT["R"][b]["ratio"], RT["V"][b]["ratio"], RT["P"][b]["ratio"],
                 RT["R"][b]["mda"], RT["V"][b]["mda"], RT["P"][b]["mda"], MN[b]))
    for arm in L4:
        print("%s: 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · 부분 %d/5 · 실패 %d/5" % (arm, *(sum(cat[arm][b] == c for b in BRAINS) for c in ("conc", "memo", "part", "fail"))))
    l4, ngw, ngate = decide_l4(cat, m)
    sr, sv, sp = status_rule(CR), status_rev(CV), status_keep(K)
    print("L4: GW 맥락 의존 범주 %d/5 · 게이팅 효과(m_GW − m_UW ≥ %d) %d/5 → %s" % (ngw, DMIN, ngate, l4))
    print("범주 규칙: 개념 수준 %d/5 · 사례 한정 %d/5 · 한 범주만 %d/5 · 획득 실패 %d/5 → %s" % (*(sum(v == c for v in CR.values()) for c in ("concept", "memo", "one", "fail")), sr))
    print("반전: 반전 성공 %d/5 · 사례 한정 %d/5 · 부분 반전 %d/5 · 고착 %d/5 · 그 밖 %d/5 → %s" % (*(sum(v == c for v in CV.values()) for c in ("success", "caseonly", "partial", "stuck", "other")), sv))
    print("유지: 기준 %d/5 · 전제 %d/5 · 유지 %d/5 · 간섭 %d/5 → %s" % (sum(v[0] for v in K.values()), sum(v[1] for v in K.values()),
                                                                    sum(v[2] == "keep" for v in K.values()), sum(v[2] == "interf" for v in K.values()), sp))
    v, h125, h126 = decide(l4, sr, sv, sp)
    print("가설: %s · %s" % (h125, h126))
    print("조작검증 통과")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
