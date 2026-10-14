#!/usr/bin/env python3
"""E193 판정 — 결합 KC 가 많은 맥락 연결(C2 p 0.02·w 20)에서 맥락 의존 범주(헌장 L4) — 판정 규칙은 E187 과 같음: 범주 × 맥락 XOR(X·A 교차, X·B 같은 쪽, Y·A 같은 쪽, Y·B 교차). 규칙은 research/experiments/logs/E193/criteria_fixed.txt.
입력(뇌 10~14): 학습 logs/E193/train_b{B}.log, 평가 logs/E193/ev_b{B}_{learn|none}_{X|Y}.log(RCEVAL 4줄 — 범주 A·B × train·new, ctx=X|Y ctx_on).
지표(1e-4 정수): e_{x,c,s} = m(학습) − m(무학습); 규칙 방향 효과 a = −e(교차 칸: X·A, Y·B), +e(같은 쪽 칸: X·B, Y·A).
뇌 단위: 획득 = 네 칸 a_train ≥ 1000; 일반화 = 네 칸 a_new ≥ 500 이면서 2·a_new ≥ a_train;
  획득 칸 수 n = #{a_train ≥ 1000}. (보고만) 맥락 무관 = 두 범주 모두 |e_{X,c,train} − e_{Y,c,train}| < 1000 — XOR 에서는 맥락을 무시하면 효과가 0 근처라 획득 실패와 겹치므로 판정 범주로 쓰지 않는다.
판정(우선순위): 조작검증 실패 → 보류 / 맥락 의존 범주(H114) 획득·일반화 ≥ 4/5 / 맥락 의존(사례 한정) 획득·일반화 아님 ≥ 4/5
  / 부분(2 ≤ n ≤ 3) ≥ 4/5 / 획득 실패(n ≤ 1) ≥ 4/5 / 그 밖 보류.
실행: python3 scripts/judge_e193.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
TRIALS = 96
NT = 3000
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


def e187_a(b):
    """(보고) 같은 뇌 E187(C0 연결 p 0.10·w 4) 칸별 규칙 방향 효과(노출, 1e-4) — 없으면 None."""
    try:
        ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E187", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
        return {(x, cc): sg * (ev[("learn", x)][0][(cc, "train")] - ev[("none", x)][0][(cc, "train")]) for x, cc, sg in CELLS}
    except (OSError, ValueError, IndexError, KeyError):
        return None


def main():
    ok, why, E = True, [], {}
    for b in BRAINS:
        try:
            tr = read_train(os.path.join(EXP, "logs", "E193", "train_b%d.log" % b))
            ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E193", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        c, k = tr["ctx"], tr["cat"]
        if not (tr["catctx"] and tr["done"] and tr["kt"] >= 1 and c and k and c[0] == NT and c[1] + c[2] == NT and c[3] + c[4] == NT
                and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800 and k[0] == NT and 1200 <= k[1] <= 1800):
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
        r0 = e187_a(b)
        if r0:
            print("    (보고) 같은 뇌 E187(C0) X·A %+d X·B %+d Y·A %+d Y·B %+d → 차 %+d %+d %+d %+d"
                  % (r0[("X", "A")], r0[("X", "B")], r0[("Y", "A")], r0[("Y", "B")],
                     a[("X", "A", "train")] - r0[("X", "A")], a[("X", "B", "train")] - r0[("X", "B")], a[("Y", "A", "train")] - r0[("Y", "A")], a[("Y", "B", "train")] - r0[("Y", "B")]))
    print("조작검증 통과 | 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · (보고) 맥락 무관 %d/5 · 부분 %d/5 · 획득 실패 %d/5"
          % (n["conc"], n["memo"], n["elem"], n["part"], n["fail"]))
    v = ("맥락 의존 범주(H114)" if n["conc"] >= 4 else "맥락 의존(사례 한정)" if n["memo"] >= 4
         else "부분" if n["part"] >= 4 else "획득 실패(H114-null)" if n["fail"] >= 4 else "보류")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
