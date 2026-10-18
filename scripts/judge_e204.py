#!/usr/bin/env python3
"""E204 판정 — 맥락 게이팅 KC 무리(G1): 맥락 집단이 반대 무리 KC 를 직접 억제해 무리마다 맥락 전용 판독이 되면 같은 학습 규칙(E200 체계)으로 L4 가 서는가.
규칙은 research/experiments/logs/E204/criteria_fixed.txt. 뇌·팔 범주는 E193~E203 과 같다.
팔(뇌 10~14): G18 게이팅·세계 18300·기본 시행열 / GN 게이팅·새 원형 세계 28300·새 시행열(배정 28500+뇌·맥락 28600+뇌·env 1) / UN 게이팅 없음·GN 과 같은 새 세계
  / U18 = E200 기존 자료(같은 뇌·시드 — logs/E200/ev_b{B}_{learn|none}_{X|Y}.log).
입력: logs/E204/train_{팔}_b{B}.log·ev_{팔}_b{B}_{learn|none}_{X|Y}.log, traces/E204/w_{팔}_b{B}.npz, 스냅숏 traces/E176/snap/conn_b{B}.npz, 비교 traces/E200/w_b{B}.npz.
조작검증(하나라도 어기면 보류): 학습 줄·특징·맥락 점검·규칙 줄(정규화 3000회·뒤 ≤ 1e-3·hard=1·η 2.4)·기준선 줄(시행 3000·|평균 DA| ≤ 0.05·기준선 1·점검 3)·
  저장 가중치(범위·KC 별 합 300 ≤ 0.01·E200 가중치와 다름)·평가 줄(E193 과 같은 검사)·세계·시행열 시드(학습 로그 세계·배정·맥락, 평가 로그 세계),
  게이팅 팔: 구성 줄(학습 로그 ≥ 2 — 학습 뇌·내부 이식 평가 뇌, 평가 로그 ≥ 1)·점검 3줄(활성 > 0)·요약 시행 3000·비 ≤ 0.01 / 게이팅 없는 팔: 게이팅 줄 0.
판정(우선순위): 조작검증 실패 → 보류 / G18 실패 ≥ 4/5 → H125-null / G18 맥락 의존 범주 ≥ 4/5 이고 개발 세계 게이팅 효과(m_G18 − m_E200 ≥ 500 인 뇌 ≥ 4/5):
  GN 범주 ≥ 4/5 이고 새 세계 게이팅 효과(m_GN − m_UN ≥ 500 인 뇌 ≥ 4/5) → H125 / GN 범주 < 4/5 → H125-세계 한정 / 그 밖 → 보류(새 세계는 게이팅 없이도 섬)
  / G18 범주 ≥ 4/5 이나 개발 세계 게이팅 효과 없음 → 게이팅 무관 / 그 밖 보류.  m = 뇌의 가장 약한 칸 a_train(1e-4).
실행: python3 scripts/judge_e204.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge_e193 import CELLS, read_eval, read_train  # noqa: E402
import judge_e199 as J9  # noqa: E402 — read_comp·weight_checks(합성 24/24)

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = ("G18", "GN", "UN")
GATED = {"G18": True, "GN": True, "UN": False}
WORLD = {"G18": 18300, "GN": 28300, "UN": 28300}
TRIAL0 = {"G18": 18500, "GN": 28500, "UN": 28500}
CTX0 = {"G18": 18600, "GN": 28600, "UN": 28600}
NT = 3000
ETA = 2.4
RMAX = 0.01
DMIN = 500


def read_da(path):
    r = {"stat": None, "chk": 0}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[E200 도파민 통계] "):
            m = re.search(r"시행 (\d+) · 평균 DA ([+-]?[0-9.]+|nan) · 평균 \|DA\| ([0-9.]+|nan) · 기준선 (\d)", ln)
            r["stat"] = (int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))) if m else ("?",)
        elif ln.startswith("[E200 도파민 기준선] 시행 "):
            r["chk"] += 1
    return r


def read_gate(path):
    """학습·평가 로그의 E204 게이팅 줄 — 구성 줄 수, 점검 [(활성, 비활성)], 요약 (시행, 활성 합, 비활성 합, 비)."""
    r = {"build": 0, "chk": [], "sum": None}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if "[E204 맥락 게이팅 KC] 무리 " in ln:
            r["build"] += 1
        elif ln.startswith("[E204 맥락 게이팅 점검] "):
            m = re.search(r"활성 무리 (\d+) · 비활성 무리 (\d+)", ln)
            r["chk"].append((int(m.group(1)), int(m.group(2))) if m else (-1, -1))
        elif ln.startswith("[E204 맥락 게이팅 KC] 시행 "):
            m = re.search(r"시행 (\d+) · 행동 창 끝 KC 발화 활성 무리 합 (\d+) · 비활성 무리 합 (\d+) · 비\(비활성/활성\) ([0-9.]+)", ln)
            r["sum"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))) if m else ("?",)
    return r


def read_seeds(path):
    """학습 로그 — '[범주 과제] catrule … (세계 시드 W) … 배정 지역 난수 T' 와 '[맥락 범주 과제] catctx … (지역 난수 C)'."""
    r = {"world": None, "trial": None, "ctx": None}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[범주 과제] catrule"):
            m = re.search(r"세계 시드 (\d+)\).*배정 지역 난수 (\d+)", ln)
            if m:
                r["world"], r["trial"] = int(m.group(1)), int(m.group(2))
        elif ln.startswith("[맥락 범주 과제] catctx"):
            m = re.search(r"지역 난수 (\d+)\)", ln)
            if m:
                r["ctx"] = int(m.group(1))
    return r


def read_eval_world(path):
    """평가 로그 '[E204 평가 세계] 세계 시드 W' 의 W 목록."""
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[E204 평가 세계] 세계 시드 "):
            m = re.search(r"세계 시드 (\d+)", ln)
            out.append(int(m.group(1)) if m else -1)
    return out


def cells(ev):
    """ev[(learn|none, X|Y)] = read_eval 결과 → a[(x, cc, train|new)] 규칙 방향(1e-4)."""
    return {(x, cc, s): sg * (ev[("learn", x)][0][(cc, s)] - ev[("none", x)][0][(cc, s)]) for x, cc, sg in CELLS for s in ("train", "new")}


def category(a):
    acq = sum(a[(x, cc, "train")] >= 1000 for x, cc, _ in CELLS)
    gen = all(a[(x, cc, "new")] >= 500 and 2 * a[(x, cc, "new")] >= a[(x, cc, "train")] for x, cc, _ in CELLS)
    if acq == 4:
        return "conc" if gen else "memo"
    if 2 <= acq <= 3:
        return "part"
    return "fail"


def weakest(a):
    return min(a[(x, cc, "train")] for x, cc, _ in CELLS)


def decide(cat, m):
    """cat[팔][뇌] ∈ conc·memo·part·fail, m[팔][뇌] = 가장 약한 칸 — 팔 G18·GN·UN·U18. 반환 (판정, 세부)."""
    def n(arm, c):
        return sum(cat[arm][b] == c for b in BRAINS)
    det = {"G18_conc": n("G18", "conc"), "G18_fail": n("G18", "fail"), "GN_conc": n("GN", "conc"),
           "dev_gate": sum(m["G18"][b] - m["U18"][b] >= DMIN for b in BRAINS), "new_gate": sum(m["GN"][b] - m["UN"][b] >= DMIN for b in BRAINS)}
    if det["G18_fail"] >= 4:
        return "H125-null", det
    if det["G18_conc"] >= 4:
        if det["dev_gate"] < 4:
            return "게이팅 무관", det
        if det["GN_conc"] >= 4:
            return ("H125" if det["new_gate"] >= 4 else "보류(새 세계는 게이팅 없이도 섬)"), det
        return "H125-세계 한정", det
    return "보류", det


def check_arm_brain(arm, b, why):
    """한 팔·뇌의 조작검증과 칸. 반환 (ok, a 또는 None, 보고 dict)."""
    ok = True
    L = os.path.join(EXP, "logs", "E204")
    tp = os.path.join(L, "train_%s_b%d.log" % (arm, b))
    wf = os.path.join(EXP, "traces", "E204", "w_%s_b%d.npz" % (arm, b))
    sn = os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b)
    tr = read_train(tp)
    cp = J9.read_comp(tp)
    da = read_da(tp)
    gt = read_gate(tp)
    sd = read_seeds(tp)
    ev = {(w, x): read_eval(os.path.join(L, "ev_%s_b%d_%s_%s.log" % (arm, b, w, x)), x) for w in ("learn", "none") for x in "XY"}
    eg = {(w, x): read_gate(os.path.join(L, "ev_%s_b%d_%s_%s.log" % (arm, b, w, x))) for w in ("learn", "none") for x in "XY"}
    ew = {(w, x): read_eval_world(os.path.join(L, "ev_%s_b%d_%s_%s.log" % (arm, b, w, x))) for w in ("learn", "none") for x in "XY"}
    wc = J9.weight_checks(wf, sn, os.path.join(EXP, "traces", "E200", "w_b%d.npz" % b))
    tag = "%s b%d" % (arm, b)
    c, k = tr["ctx"], tr["cat"]
    if not (tr["catctx"] and tr["done"] and tr["kt"] >= 1 and c and k and c[0] == NT and c[1] + c[2] == NT and c[3] + c[4] == NT
            and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800 and k[0] == NT and 1200 <= k[1] <= 1800):
        ok = False; why.append("%s 학습 줄(%s %s)" % (tag, c, k))
    if len(tr["fchk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["fchk"]):
        ok = False; why.append("%s 특징 점검 %s" % (tag, tr["fchk"]))
    if len(tr["cchk"]) != 3 or not all(x > 0 and y == 0 and z == 0 for x, y, z in tr["cchk"]):
        ok = False; why.append("%s 맥락 점검 %s" % (tag, tr["cchk"]))
    cm = cp["comp"]
    if not (cp["hard"] and cm is not None and len(cm) == 3 and cm[0] == NT and cm[1] <= 1e-3 and cm[2] == 1
            and cp["eta"] is not None and abs(cp["eta"] - ETA) <= 1e-9):
        ok = False; why.append("%s 규칙 줄(hard %s·정규화 %s·eta %s)" % (tag, cp["hard"], cm, cp["eta"]))
    st = da["stat"]
    if not (st is not None and len(st) == 4 and st[0] == NT and abs(st[1]) <= 0.05 and st[3] == 1 and da["chk"] == 3):
        ok = False; why.append("%s 기준선 줄(%s·점검 %d)" % (tag, st, da["chk"]))
    if not (sd["world"] == WORLD[arm] and sd["trial"] == TRIAL0[arm] + b and sd["ctx"] == CTX0[arm] + b):
        ok = False; why.append("%s 시드 줄(%s)" % (tag, sd))
    if GATED[arm]:
        s_ = gt["sum"]
        if not (gt["build"] >= 2 and len(gt["chk"]) == 3 and all(a_ > 0 for a_, _ in gt["chk"])
                and s_ is not None and len(s_) == 4 and s_[0] == NT and s_[1] > 0 and s_[3] <= RMAX):
            ok = False; why.append("%s 게이팅(구성 %d·점검 %s·요약 %s)" % (tag, gt["build"], gt["chk"], s_))
    elif gt["build"] or gt["chk"] or gt["sum"] is not None:
        ok = False; why.append("%s 게이팅 없는 팔에 게이팅 줄" % tag)
    lo_hi, dmax, diff, fb, D = wc
    if not (lo_hi and dmax <= 0.01 and diff):
        ok = False; why.append("%s 가중치(범위 %s·KC 합 최대 |S−300| %.4g·E200 와 다름 %s)" % (tag, lo_hi, dmax, diff))
    good_ev = True
    for (w, x), (rc, pushed, kt, fw, sat, bad) in ev.items():
        if set(rc) != {(cc, s) for cc in "AB" for s in ("train", "new")} or bad or sat or kt < 1 or fw != 0 or pushed != (8 if w == "learn" else 0):
            ok = False; good_ev = False; why.append("%s %s %s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s)" % (tag, w, x, len(rc), bad, sat, kt, fw, pushed))
        if ew[(w, x)] != [WORLD[arm]]:
            ok = False; why.append("%s %s %s 평가 세계 %s" % (tag, w, x, ew[(w, x)]))
        if GATED[arm] and eg[(w, x)]["build"] < 1:
            ok = False; why.append("%s %s %s 평가 게이팅 구성 줄 없음" % (tag, w, x))
        if not GATED[arm] and eg[(w, x)]["build"]:
            ok = False; why.append("%s %s %s 평가에 게이팅 구성 줄" % (tag, w, x))
    a = cells(ev) if good_ev else None
    rep = {"ratio": (gt["sum"][3] if GATED[arm] and gt["sum"] and len(gt["sum"]) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb}
    return ok, a, rep


def u18(b):
    ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E200", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
    return cells(ev)


def fmt(a):
    return " ".join("%s·%s %+d/%+d" % (x, cc, a[(x, cc, "train")], a[(x, cc, "new")]) for x, cc, _ in CELLS)


def main():
    ok, why = True, []
    A, R = {arm: {} for arm in ARMS + ("U18",)}, {arm: {} for arm in ARMS}
    for arm in ARMS:
        for b in BRAINS:
            try:
                ok_, a, rep = check_arm_brain(arm, b, why)
            except (OSError, ValueError, IndexError, KeyError, TypeError) as ex:
                print("판정: 보류(결측 — %s b%d %s)" % (arm, b, type(ex).__name__))
                return 0
            ok = ok and ok_
            if a is not None:
                A[arm][b] = a
            R[arm][b] = rep
    for b in BRAINS:
        try:
            A["U18"][b] = u18(b)
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — E200 b%d %s)" % (b, type(ex).__name__))
            return 0
    if not ok or any(len(A[arm]) != len(BRAINS) for arm in ARMS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cat = {arm: {b: category(A[arm][b]) for b in BRAINS} for arm in A}
    m = {arm: {b: weakest(A[arm][b]) for b in BRAINS} for arm in A}
    for arm in ARMS + ("U18",):
        for b in BRAINS:
            extra = ""
            if arm in R:
                extra = " | 게이팅 비 %s · 평균 DA %+.4f · 경계 시냅스 %.3f" % (("%.4f" % R[arm][b]["ratio"]) if R[arm][b]["ratio"] is not None else "—", R[arm][b]["mda"], R[arm][b]["fb"])
            print("%s b%d 규칙 방향 효과(1e-4, 노출/새) %s | 범주 %s · 가장 약한 칸 %+d%s" % (arm, b, fmt(A[arm][b]), cat[arm][b], m[arm][b], extra))
    for arm in ARMS + ("U18",):
        print("%s: 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · 부분 %d/5 · 실패 %d/5" % (arm, *(sum(cat[arm][b] == c for b in BRAINS) for c in ("conc", "memo", "part", "fail"))))
    v, det = decide(cat, m)
    print("게이팅 효과(m 차 ≥ %d 인 뇌): 개발 세계 G18−E200 %d/5 · 새 세계 GN−UN %d/5" % (DMIN, det["dev_gate"], det["new_gate"]))
    print("조작검증 통과")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
