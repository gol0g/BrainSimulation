#!/usr/bin/env python3
"""E106 판정 — E106.md 4절. 64런 전에는 수치 미출력. --selftest."""
import re
import sys

LINE = re.compile(r"^\s*(learn|shuf) w(\d) t(\d+): => MINCIRC .*?reward=([0-9.]+) \*\*eval=([0-9.]+)\*\*")
W, TS = [5, 6, 7, 8], list(range(400, 408))


def decide(nl, ns):
    if nl >= 28 and ns <= 4:
        return "귀속 성립 — w_max 2 기본값 채택"
    if nl <= 16 or ns >= 12:
        return "귀속 실패"
    return "보류"


def selftest():
    cases = [((28, 4), "귀속 성립"), ((27, 0), "보류"), ((32, 5), "보류"), ((16, 0), "귀속 실패"), ((32, 12), "귀속 실패")]
    ok = sum(decide(*a).startswith(e) for a, e in cases)
    print("자체 검증 %d/%d 통과" % (ok, len(cases))); return ok == len(cases)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = {}
    try:
        for ln in open("research/experiments/E106.log", encoding="utf-8"):
            m = LINE.match(ln)
            if m:
                c, s, t, rw, ev = m.groups(); r[(c, int(s), int(t))] = (float(rw), float(ev))
    except FileNotFoundError:
        pass
    need = {(c, s, t) for c in ("learn", "shuf") for s in W for t in TS}
    if len(need & set(r)) < 64:
        print("[E106] %d/64런 완료 — **판정 보류. 결과가 다 모일 때까지 수치를 출력하지 않는다.**" % len(need & set(r))); sys.exit(0)
    bad = sum(r[("shuf", s, t)][0] != r[("learn", s, t)][0] for s in W for t in TS)
    print("조작검증: shuffled 보상 총량 = learn %d/32 일치%s" % (32 - bad, "" if bad == 0 else "  ← 불일치"))
    for s in W:
        print("  배선 %d: learn %s | shuf %s" % (s, " ".join("%.0f" % r[("learn", s, t)][1] for t in TS), " ".join("%.0f" % r[("shuf", s, t)][1] for t in TS)))
    nl = sum(r[("learn", s, t)][1] >= 90 for s in W for t in TS); ns = sum(r[("shuf", s, t)][1] >= 90 for s in W for t in TS)
    print("판정: learn %d/32, shuffled %d/32 → %s" % (nl, ns, decide(nl, ns)))
