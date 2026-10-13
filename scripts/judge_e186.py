#!/usr/bin/env python3
"""E186 판정 — 원시 특징 범주의 원형 추상화(원형 효과·전형성 기울기). 평가만. 규칙은 research/experiments/logs/E186/criteria_fixed.txt.
입력: logs/E186/ev_b{B}_{learn|none}.log — RCEVAL 8줄(범주 A·B × 집합 train·new·proto·far), '=> DECOMP … rceval … pushed=N', 종류 적재 줄.
  주 표본 뇌 16~20(E185 1,500 가중치), 재현 표본 뇌 10~14(E184 U 가중치).
지표(1e-4 정수): e_{c,s} = m(학습) − m(무학습); 규칙 방향 효과 a_{A,s} = −e_{A,s}(A 정답 교차 = 음수), a_{B,s} = +e_{B,s}.
뇌 단위(두 범주 모두):
  원형 효과(강) = a_proto ≥ a_train − 50 이면서 a_proto > 0
  원형 우위    = a_proto ≥ a_new + 50 이면서 a_proto > 0
  전형성 기울기 = a_new ≥ a_far + 50 이면서 a_new > 0   (정답 방향 효과가 있을 때만 모양을 셈)
판정(표본마다, 우선순위): 조작검증 실패 → 보류 / 원형 추상화(강 ≥ 4/5·기울기 ≥ 4/5) / 원형 우위·기울기(우위 ≥ 4/5·기울기 ≥ 4/5)
  / 기울기만(기울기 ≥ 4/5) / 원형 열세(a_proto ≤ a_new − 50 두 범주 ≥ 4/5) / 보류. 재현 = 두 표본 같은 판정.
실행: python3 scripts/judge_e186.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
SAMPLES = {"주(16~20)": (16, 17, 18, 19, 20), "재현(10~14)": (10, 11, 12, 13, 14)}
SETS = ("train", "new", "proto", "far")
TRIALS = 96
M = 50


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path):
    rc, pushed, kt, fw, sat = {}, None, 0, 0, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCEVAL "):
                kv = {}
                for x in ln.split(" | ")[0].split()[1:]:
                    k, v = x.split("=", 1); kv[k] = v
                rc[(kv["cat"], kv["set"])] = (q4(kv["mod"]), int(kv["trials"]), int(kv["feat_on"]))
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
    return rc, pushed, kt, fw, sat


def classify(a):
    both = lambda f: all(f(c) for c in "AB")
    return {"strong": both(lambda c: a[(c, "proto")] >= a[(c, "train")] - M and a[(c, "proto")] > 0),
            "adv": both(lambda c: a[(c, "proto")] >= a[(c, "new")] + M and a[(c, "proto")] > 0),
            "grad": both(lambda c: a[(c, "new")] >= a[(c, "far")] + M and a[(c, "new")] > 0),
            "inf": both(lambda c: a[(c, "proto")] <= a[(c, "new")] - M)}


def verdict(cl):
    n = lambda k: sum(c[k] for c in cl)
    if n("strong") >= 4 and n("grad") >= 4:
        return "원형 추상화"
    if n("adv") >= 4 and n("grad") >= 4:
        return "원형 우위·기울기"
    if n("grad") >= 4:
        return "기울기만"
    if n("inf") >= 4:
        return "원형 열세"
    return "보류"


def main():
    ok, why, out = True, [], {}
    for name, brains in SAMPLES.items():
        cl = []
        for b in brains:
            try:
                ev = {w: read_eval(os.path.join(EXP, "logs", "E186", "ev_b%d_%s.log" % (b, w))) for w in ("learn", "none")}
            except (OSError, ValueError, IndexError, KeyError) as ex:
                print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
                return 0
            for w, (rc, pushed, kt, fw, sat) in ev.items():
                if set(rc) != {(c, s) for c in "AB" for s in SETS}:
                    ok = False; why.append("b%d %s RCEVAL 결측(%d)" % (b, w, len(rc))); continue
                if any(t != TRIALS or fo != TRIALS for _, t, fo in rc.values()):
                    ok = False; why.append("b%d %s 시행·특징 켬" % (b, w))
                if pushed != (8 if w == "learn" else 0) or kt < 1 or fw != 0 or sat:
                    ok = False; why.append("b%d %s pushed %s·종류 %d·특징 %d·포화 %s" % (b, w, pushed, kt, fw, sat))
            if not ok:
                continue
            e = {k: ev["learn"][0][k][0] - ev["none"][0][k][0] for k in ev["learn"][0]}
            a = {(c, s): (-e[(c, s)] if c == "A" else e[(c, s)]) for c in "AB" for s in SETS}
            c_ = classify(a)
            cl.append(c_)
            print("%s b%d 규칙 방향 효과(1e-4) A: 원형 %+d 노출 %+d 새 %+d 먼 %+d · B: 원형 %+d 노출 %+d 새 %+d 먼 %+d | 강 %d 우위 %d 기울기 %d 열세 %d"
                  % (name, b, a[("A", "proto")], a[("A", "train")], a[("A", "new")], a[("A", "far")],
                     a[("B", "proto")], a[("B", "train")], a[("B", "new")], a[("B", "far")], c_["strong"], c_["adv"], c_["grad"], c_["inf"]))
        out[name] = cl
    if not ok:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    vs = {name: verdict(cl) for name, cl in out.items()}
    for name, cl in out.items():
        print("%s: 강 %d/5 · 우위 %d/5 · 기울기 %d/5 · 열세 %d/5 → %s" % (name, sum(c["strong"] for c in cl), sum(c["adv"] for c in cl),
                                                                    sum(c["grad"] for c in cl), sum(c["inf"] for c in cl), vs[name]))
    prim, rep = vs["주(16~20)"], vs["재현(10~14)"]
    print("판정: %s%s" % (prim, " (재현)" if prim == rep else " (재현 표본: %s)" % rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
