#!/usr/bin/env python3
"""E101 판정 — research/experiments/E101.md 4절 사전기준 구현.

  (인자 없음) : E101.log + logs/E101/ (+ E100 쌍둥이·기준 로그). **120런이 다 모이기 전에는 수치 미출력.**
  --selftest  : 합성 자료로 (a) 판정 경계 검증.
"""
import os
import re
import sys

EXP = "research/experiments"
W = [5, 6, 7]
TS = list(range(300, 308))
ARMS = ["W2-L4", "W2-rev8", "W2-L8", "W2-rev16", "W20-rev16"]
MINC = re.compile(r"first=([0-9.]+) last=([0-9.]+) .*?\*\*eval=([0-9.]+)\*\*")
REVL = re.compile(r"\(reversal\) 최종 규칙 기준 ([0-9.]+)% \| 원래 규칙 기준 ([0-9.]+)%")
DW = re.compile(r"kc_out_l: n=\d+ \|Δ\|평균 ([0-9.]+) .*?평균 [0-9.]+→([0-9.]+) \| std [0-9.]+→([0-9.]+)")
ADIFF = re.compile(r"A전용 KC → .*?차이 ([+-][0-9.]+)")


def read_run(path, rev):
    if not os.path.exists(path):
        return None
    t = open(path, encoding="utf-8").read()
    m = [ln for ln in t.splitlines() if ln.startswith("=> MINCIRC")]
    if not m:
        return None
    first, last, ev = map(float, MINC.search(m[-1]).groups())
    d = {"first": first, "eval": ev}
    dw = DW.search(t)
    d["dw"] = tuple(map(float, dw.groups())) if dw else None
    a = ADIFF.search(t)
    d["adiff"] = float(a.group(1)) if a else None
    if rev:
        mm = REVL.search(t)
        if not mm:
            return None
        d["new"], d["orig"] = float(mm.group(1)), float(mm.group(2))
    return d


def load():
    r = {}
    for arm in ARMS:
        for s in W:
            for t in TS:
                x = read_run(os.path.join(EXP, "logs/E101", "%s_w%d_t%d.log" % (arm, s, t)), "rev" in arm)
                if x:
                    r[(arm, s, t)] = x
    for arm, fn, rev in (("E100-L4", "L4", False), ("E100-L8", "L8", False), ("E100-rev", "rev", True)):
        for s in W:
            for t in TS:
                x = read_run(os.path.join(EXP, "logs/E100", "%s_w%d_t%d.log" % (fn, s, t)), rev)
                if x:
                    r[(arm, s, t)] = x
    return r


def arm_rate(r, rev_arm, twin_arm):
    valid = ok = twin_bad = 0
    for s in W:
        for t in TS:
            rv, tw = r[(rev_arm, s, t)], r[(twin_arm, s, t)]
            if rv["first"] != tw["first"]:
                twin_bad += 1
                continue
            if tw["eval"] < 90.0:
                continue
            valid += 1
            ok += rv["new"] >= 90.0
    return valid, ok, twin_bad


def decide(v2, s2, v20, s20, v16, s16):
    if v2 < 12 or v20 < 12:
        return "판정 불가(유효 쌍 부족)"
    S2, S20 = s2 / v2, s20 / v20
    S16 = (s16 / v16) if v16 >= 12 else None
    if S2 >= 0.5 and S20 <= 0.2:
        return "비대칭 지지"
    if S20 >= 0.5 and S2 >= 0.5:
        return "둘 다 기여"
    if S20 >= 0.5 and S2 <= 0.2:
        return "속도 지지"
    if S2 <= 0.2 and S20 <= 0.2 and S16 is not None and S16 <= 0.2:
        return "둘 다 아님(H030-other)"
    return "보류"


def judge(r):
    out = []
    same = sum(r[("W2-L4", s, t)]["dw"] == r[("E100-L4", s, t)]["dw"] for s in W for t in TS)
    out.append("조작검증: W2-L4 가중치 변화가 E100 L4(w_max 20)와 동일 %d/24%s" % (same, "  ← 조작 무효 의심" if same == 24 else ""))
    acq = sum(r[("W2-L4", s, t)]["eval"] >= 90.0 for s in W for t in TS)
    out.append("(0) 획득 전제: W2-L4 ≥90%% %d/24 (%s)" % (acq, "충족" if acq >= 12 else "미충족 — w_max 2가 획득을 깨뜨림, (a)는 W2-rev16으로만"))
    v2, s2, b2 = arm_rate(r, "W2-rev8", "W2-L4")
    v16, s16, b16 = arm_rate(r, "W2-rev16", "W2-L8")
    v20, s20, b20 = arm_rate(r, "W20-rev16", "E100-L8")
    for name, v, s_, b in (("W2-rev8 (400+400)", v2, s2, b2), ("W2-rev16 (800+800)", v16, s16, b16), ("W20-rev16 (800+800)", v20, s20, b20)):
        out.append("  %-20s 반전 성공 %d/%d%s  쌍둥이 불일치 %d" % (name, s_, v, (" (%.0f%%)" % (100.0 * s_ / v)) if v else "", b))
    for arm in ARMS:
        out.append("  %-10s 평가 %s" % (arm, " | ".join(" ".join("%.0f" % r[(arm, s, t)].get("new", r[(arm, s, t)]["eval"]) for t in TS) for s in W)))
    if acq >= 12:
        a = decide(v2, s2, v20, s20, v16, s16)
    else:
        a = decide(v16, s16, v20, s20, v16, s16) + " [W2-rev16 기준]"
    out.append("(a) 원인 판별: %s" % a)
    out.append("(b) W2-rev16 반전 성공 %d/%d" % (s16, v16))
    neg2 = sum((r[("W2-rev8", s, t)]["adiff"] or 0) < 0 for s in W for t in TS)
    neg20 = sum((r[("E100-rev", s, t)]["adiff"] or 0) < 0 for s in W for t in TS)
    out.append("기전 보조: 반전 후 A전용 KC 차이가 음수(새 방향)로 넘어간 런 — W2-rev8 %d/24 vs E100 rev(w_max 20) %d/24" % (neg2, neg20))
    d2 = [abs(r[("W2-L4", s, t)]["adiff"] or 0) for s in W for t in TS]
    d20 = [abs(r[("E100-L4", s, t)]["adiff"] or 0) for s in W for t in TS]
    out.append("배제 못 한 설명 1 확인: 획득 후 |A전용 차이| 평균 W2 %.4f vs W20 %.4f" % (sum(d2) / 24, sum(d20) / 24))
    return out, a


def selftest():
    cases = [((24, 12, 24, 4, 24, 20), "비대칭 지지"), ((24, 11, 24, 4, 24, 20), "보류"),
             ((24, 12, 24, 5, 24, 20), "보류"), ((24, 4, 24, 12, 24, 4), "속도 지지"),
             ((24, 12, 24, 12, 24, 12), "둘 다 기여"), ((24, 4, 24, 4, 24, 4), "둘 다 아님(H030-other)"),
             ((24, 4, 24, 4, 24, 5), "보류"), ((11, 11, 24, 0, 24, 20), "판정 불가(유효 쌍 부족)")]
    n = 0
    for args, exp in cases:
        got = decide(*args)
        n += got == exp
        if got != exp:
            print("실패", args, exp, "→", got)
    print("자체 검증 %d/%d 통과" % (n, len(cases)))
    return n == len(cases)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = load()
    need = {(a, s, t) for a in ARMS for s in W for t in TS}
    have = need & set(r)
    if len(have) < len(need):
        print("[E101] %d/%d런 완료 — **판정 보류. 결과가 다 모일 때까지 수치를 출력하지 않는다.**" % (len(have), len(need)))
        sys.exit(0)
    print("[E101] %d/%d런 완료" % (len(have), len(need)))
    out, _ = judge(r)
    print("\n".join(out))
