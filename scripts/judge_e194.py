#!/usr/bin/env python3
"""E194 판정 — L4 XOR 학습의 학습량 의존: E193 과 같은 설정(맥락 연결 C2·motor 기저값 −4900·catctx)으로 9,000시행. 규칙은 research/experiments/logs/E194/criteria_fixed.txt.
입력(뇌 10~14): 학습 logs/E194/train_b{B}.log(9,000), 평가 logs/E194/ev_b{B}_{learn|none}_{X|Y}.log; 3,000 기준 = logs/E193 같은 파일(같은 시드 — 앞 3,000 같은 시행열).
지표(1e-4 정수): e = m(학습) − m(무학습); 규칙 방향 효과 a = −e(교차 칸 X·A·Y·B), +e(같은 쪽 칸 X·B·Y·A). 칸 평균 m = 네 칸 a_train 평균.
(가) L4(E187 규칙, 9,000): 획득 = 네 칸 a_train ≥ 1000; 일반화 = 네 칸 a_new ≥ 500·2·a_new ≥ a_train — 맥락 의존 범주 ≥ 4/5 / 사례 한정 ≥ 4/5 / 부분(2~3 칸) ≥ 4/5 / 획득 실패(≤ 1 칸) ≥ 4/5 / 그 밖 보류.
(나) 학습량 의존: Δ = m(9,000) − m(3,000, E193 같은 뇌); 자람 = Δ ≥ +300; 상한 = |Δ| < 150 — 자람 ≥ 4/5 → 자람(H115) / 상한 ≥ 4/5 → 상한(H115-null) / 그 밖 보류.
판정 줄: "판정: 학습량 <(나)> · L4 <(가)>" — 조작검증 실패 → 보류(조작검증 실패), 결측 → 보류(결측).
실행: python3 scripts/judge_e194.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
TRIALS = 96
NT = 9000
LO, HI = int(0.4 * NT), int(0.6 * NT)
CELLS = (("X", "A", -1), ("X", "B", +1), ("Y", "A", +1), ("Y", "B", -1))


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path, ctx):
    rc, pushed, kt, fw, sat, bad = {}, None, 0, 0, False, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCEVAL "):
                kv = {}
                for x in ln.split(" | ")[0].split()[1:]:
                    k, v = x.split("=", 1); kv[k] = v
                rc[(kv["cat"], kv["set"])] = q4(kv["mod"])
                if int(kv["trials"]) != TRIALS or int(kv["feat_on"]) != TRIALS or kv.get("ctx") != ctx or int(kv.get("ctx_on", "-1")) != TRIALS:
                    bad = True
                for seg in ln.split(" | ")[1:]:
                    for part in seg.split("motor L/R ")[1:]:
                        a, b = part.split()[0].split("/")
                        if q4(a) >= 6600 and q4(b) >= 6600:
                            sat = True
            elif ln.startswith("=> DECOMP ") and "rceval" in ln:
                pushed = int(ln.split("pushed=")[1].split()[0])
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                kt += 1
            elif ln.startswith("[E184 특징 가중치 적재]"):
                fw += 1
    return rc, pushed, kt, fw, sat, bad


def read_train(path):
    r = {"catctx": False, "ctx": None, "cat": None, "fchk": [], "cchk": [], "done": False, "kt": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[맥락 범주 과제] catctx"):
                r["catctx"] = True
            elif ln.startswith("[맥락 범주 과제] 시행 "):
                r["ctx"] = (int(ln.split("시행 ")[1].split()[0]), int(ln.split("맥락 X ")[1].split()[0]), int(ln.split("· Y ")[1].split()[0]),
                            int(ln.split("정답 같은 쪽 ")[1].split()[0]), int(ln.split("· 교차 ")[1].split()[0]))
            elif ln.startswith("[범주 과제] 시행 "):
                p = ln.split()
                r["cat"] = (int(p[3]), int(ln.split("범주 B(같은 쪽) ")[1].split()[0]), int(ln.split("A(교차) ")[1].split()[0]))
            elif ln.startswith("[범주 과제 점검]"):
                r["fchk"].append((int(ln.split("행동 창 끝 특징 발화 ")[1].split()[0]), int(ln.split("시행 끝 ")[1].split()[0])))
            elif ln.startswith("[맥락 범주 점검]"):
                r["cchk"].append((int(ln.split("켠 맥락 발화 ")[1].split()[0]), int(ln.split("끈 맥락 발화 ")[1].split()[0]),
                                  int(ln.split("시행 끝 두 맥락 발화 ")[1].split()[0])))
            elif ln.startswith("[학습] "):
                r["done"] = True
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                r["kt"] += 1
    return r


def a_of(exp, b):
    """같은 뇌 다른 실험(exp)의 칸별 규칙 방향 효과(노출, 1e-4) — 없거나 칸이 모자라면 None."""
    try:
        ev = {(w, x): read_eval(os.path.join(EXP, "logs", exp, "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
        return {(x, cc): sg * (ev[("learn", x)][0][(cc, "train")] - ev[("none", x)][0][(cc, "train")]) for x, cc, sg in CELLS}
    except (OSError, ValueError, IndexError, KeyError):
        return None


def main():
    ok, why, E = True, [], {}
    for b in BRAINS:
        try:
            tr = read_train(os.path.join(EXP, "logs", "E194", "train_b%d.log" % b))
            ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E194", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        c, k = tr["ctx"], tr["cat"]
        if not (tr["catctx"] and tr["done"] and tr["kt"] >= 1 and c and k and c[0] == NT and c[1] + c[2] == NT and c[3] + c[4] == NT
                and LO <= c[1] <= HI and LO <= c[3] <= HI and k[0] == NT and LO <= k[1] <= HI):
            ok = False; why.append("b%d 학습 줄(%s %s)" % (b, c, k))
        if len(tr["fchk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["fchk"]):
            ok = False; why.append("b%d 특징 점검 %s" % (b, tr["fchk"]))
        if len(tr["cchk"]) != 3 or not all(x > 0 and y == 0 and z == 0 for x, y, z in tr["cchk"]):
            ok = False; why.append("b%d 맥락 점검 %s" % (b, tr["cchk"]))
        m = {}
        for (w, x), (rc, pushed, kt, fw, sat, bad) in ev.items():
            if set(rc) != {(cc, s) for cc in "AB" for s in ("train", "new")} or bad or sat or kt < 1 or fw != 0 or pushed != (8 if w == "learn" else 0):
                ok = False; why.append("b%d %s %s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s)" % (b, w, x, len(rc), bad, sat, kt, fw, pushed))
                continue
            for (cc, s), v in rc.items():
                m[(w, x, cc, s)] = v
        if len(m) == 16:
            E[b] = {(x, cc, s): m[("learn", x, cc, s)] - m[("none", x, cc, s)] for x in "XY" for cc in "AB" for s in ("train", "new")}
    if not ok or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    n = {"conc": 0, "memo": 0, "elem": 0, "part": 0, "fail": 0}
    dose = {}
    for b in BRAINS:
        e = E[b]
        a = {(x, cc, s): sg * e[(x, cc, s)] for x, cc, sg in CELLS for s in ("train", "new")}
        acq_cells = sum(a[(x, cc, "train")] >= 1000 for x, cc, _ in CELLS)
        acq = acq_cells == 4
        gen = all(a[(x, cc, "new")] >= 500 and 2 * a[(x, cc, "new")] >= a[(x, cc, "train")] for x, cc, _ in CELLS)
        elem = all(abs(e[("X", cc, "train")] - e[("Y", cc, "train")]) < 1000 for cc in "AB")
        n["conc"] += acq and gen
        n["memo"] += acq and not gen
        n["elem"] += elem
        n["part"] += 2 <= acq_cells <= 3
        n["fail"] += acq_cells <= 1
        print("b%d 규칙 방향 효과(1e-4, 노출/새) X·A %+d/%+d X·B %+d/%+d Y·A %+d/%+d Y·B %+d/%+d | 획득 칸 %d | 맥락 차 A %+d B %+d"
              % (b, a[("X", "A", "train")], a[("X", "A", "new")], a[("X", "B", "train")], a[("X", "B", "new")],
                 a[("Y", "A", "train")], a[("Y", "A", "new")], a[("Y", "B", "train")], a[("Y", "B", "new")], acq_cells,
                 e[("X", "A", "train")] - e[("Y", "A", "train")], e[("X", "B", "train")] - e[("Y", "B", "train")]))
        r3 = a_of("E193", b)
        m9 = sum(a[(x, cc, "train")] for x, cc, _ in CELLS) / 4.0
        if r3 is None:
            dose[b] = None
            print("    (나) 3,000 기준(E193) 없음")
        else:
            m3 = sum(r3.values()) / 4.0
            dose[b] = m9 - m3
            print("    (나) 칸 평균 3,000 %+.0f → 9,000 %+.0f (Δ %+.0f) | 3,000 칸 X·A %+d X·B %+d Y·A %+d Y·B %+d"
                  % (m3, m9, m9 - m3, r3[("X", "A")], r3[("X", "B")], r3[("Y", "A")], r3[("Y", "B")]))
    print("조작검증 통과 | 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · (보고) 맥락 무관 %d/5 · 부분 %d/5 · 획득 실패 %d/5"
          % (n["conc"], n["memo"], n["elem"], n["part"], n["fail"]))
    v = ("맥락 의존 범주" if n["conc"] >= 4 else "맥락 의존(사례 한정)" if n["memo"] >= 4
         else "부분" if n["part"] >= 4 else "획득 실패" if n["fail"] >= 4 else "보류")
    if any(d is None for d in dose.values()):
        print("판정: 보류(결측 — 3,000 기준 E193)")
        return 0
    grow = sum(d >= 300 for d in dose.values()); flat = sum(abs(d) < 150 for d in dose.values())
    dv = "자람(H115)" if grow >= 4 else "상한(H115-null)" if flat >= 4 else "보류"
    print("(나) 학습량: 자람 %d/5 · 상한 %d/5" % (grow, flat))
    print("판정: 학습량 %s · L4 %s" % (dv, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
