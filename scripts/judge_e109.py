#!/usr/bin/env python3
"""E109 판정 — E109.md 4절. 20런 전에는 수치 미출력. --selftest."""
import os
import re
import sys

RAW = "research/experiments/logs/E109"
POST = re.compile(r"\[사후\] .*?\*\*변조폭 ([-+0-9.]+)\*\*")
PRE = re.compile(r"\[사전\] .*?\*\*변조폭 ([-+0-9.]+)\*\*")
DMOD = re.compile(r"변조폭 변화 ([-+0-9.]+)")
TRN = re.compile(r"\[이식\] (\d+)개 경로: (.*)")
ETAS, BR = ("0.15", "0.03"), range(5)


def rd(eta, c, b):
    p = os.path.join(RAW, "eta%s_%s_b%d.log" % (eta, c, b))
    if not os.path.exists(p):
        return None
    t = open(p, encoding="utf-8").read()
    m1, m0, d, tr = POST.search(t), PRE.search(t), DMOD.search(t), TRN.search(t)
    if not (m1 and m0 and d):
        return None
    return {"post": float(m1.group(1)), "pre": float(m0.group(1)), "d": float(d.group(1)),
            "trn": tr.group(2) if tr else ""}


def decide(e, post):
    m = sum(e) / len(e); neg = sum(x < 0 for x in e)
    if m <= -0.10 and neg >= 4:
        return ("강한 지지(행동 역전)" if sum(p < 0 for p in post) >= 4 else "H037 지지"), m
    if abs(m) < 0.05:
        return "H037-scale", m
    if neg < 4 and sum(x > 0 for x in e) < 4:
        return "H037-credit", m
    return "보류", m


def selftest():
    cs = [(([-0.2] * 5, [0.3] * 5), "H037 지지"), (([-0.2] * 5, [-0.1] * 4 + [0.1]), "강한 지지"),
          (([-0.2] * 3 + [0.1] * 2, [0.3] * 5), "H037-credit"), (([0.01] * 5, [0.5] * 5), "H037-scale"),
          (([-0.09] * 5, [0.4] * 5), "보류")]
    ok = sum(decide(*a)[0].startswith(e) for a, e in cs)
    print("자체 검증 %d/%d 통과" % (ok, len(cs))); return ok == len(cs)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    D = {(e, c, b): rd(e, c, b) for e in ETAS for c in ("학습", "무학습") for b in BR}
    have = sum(v is not None for v in D.values())
    if have < 20:
        print("[E109] %d/20런 — **판정 보류. 다 모일 때까지 수치 미출력.**" % have); sys.exit(0)
    print("[E109] 20/20런")
    zero = all(D[(e, "무학습", b)]["d"] == 0.0 for e in ETAS for b in BR)
    print("조작검증: 무학습 변조폭 변화 전부 0.0000: %s" % zero)
    km = all(sum(("kc_%s_to_motor_%s" % (k, m)) in D[(e, "학습", b)]["trn"] for k in "lr" for m in "lr") == 4 for e in ETAS for b in BR)
    print("조작검증: 이식 목록에 kc_*_to_motor_* 4개 포함(학습 10런): %s" % km)
    base = [D[(e, "무학습", b)]["post"] for e in ETAS for b in BR]
    print("조작검증: 무학습 사후 변조폭 %s (보정 zero 0.563~0.578 ±10%%: %s)" % (
        " ".join("%.3f" % x for x in base), all(0.5 <= x <= 0.64 for x in base)))
    res = []
    for e in ETAS:
        eff = [D[(e, "학습", b)]["post"] - D[(e, "무학습", b)]["post"] for b in BR]
        post = [D[(e, "학습", b)]["post"] for b in BR]
        same = sum(D[(e, "학습", b)]["post"] == D[(e, "무학습", b)]["post"] for b in BR)
        v, m = decide(eff, post)
        res.append(v)
        print("eta %s: 효과 %s 평균 %+.4f 음수 %d/5 | 학습 사후 변조폭 %s | 학습=무학습 동일 %d/5 → %s" % (
            e, " ".join("%+.4f" % x for x in eff), m, sum(x < 0 for x in eff), " ".join("%+.3f" % x for x in post), same, v))
    print("판정: %s" % ("강한 지지" if any(r.startswith("강한") for r in res) else ("H037 지지" if any(r == "H037 지지" for r in res) else " / ".join(res))))
