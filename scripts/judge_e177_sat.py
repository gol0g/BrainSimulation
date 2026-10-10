#!/usr/bin/env python3
"""E177 정정 1 판정(logs/E177/criteria_fixed.txt '정정 1' — 2026-10-11 01:36:49) — 사전 판정(judge_e177.py, 원 규칙 그대로)에
측정 타당성 조작검증을 더한다: 맥락 켬 평가가 조향을 표현할 수 있는가(motor 포화).
입력: research/experiments/E177.log(본실험 평가 줄) + logs/E177/sat_check/main/summary.out(scripts/e177_sat_check.sh main 10 11 12 13 14).
뇌마다(모두 만족해야 그 뇌 통과):
  · 재측정 4개(학습·무학습 × 끔·켬) 모두 rc 0·DECOMP 줄·진단 줄 좌·우 2개
  · 재현: 재측정 변조폭과 본실험 평가 줄의 1e-4 정수 차 ≤ 2(비트 수준 비결정성 범위)
  · 비포화: 켬 평가 2개(학습·무학습)의 쪽별 motor 발화율 8개(자극 좌·우 × motor 좌·우)가 **모두 ≥ 0.66** 이면 포화(측정 불능) → 실패
정정 판정: 사전 판정이 '보류(조작검증 실패)'면 그대로. 아니면 한 뇌라도 위 검사 실패 → '보류(정정 1 — ...)'. 모두 통과 → 사전 판정 그대로.
끔 규칙 학습(e_off ≤ −0.10)은 판정이 아닌 관측으로 따로 출력.
실행: python3 scripts/judge_e177_sat.py (저장소 루트에서)"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e177 as J  # noqa: E402

SAT = 0.66
REPRO = 2
HD = re.compile(r"^b(\d+) (all|none) (off|on) rc=(-?\d+) \| (.*)$")
MOD = re.compile(r"=> DECOMP mode=\w+ mod=([-+0-9.]+)")
DG = re.compile(r"side=(left|right) n=\d+ motor L/R ([0-9.]+)/([0-9.]+) KC L/R ([0-9.]+)/([0-9.]+)")
WMAP = {"all": "learn", "none": "none"}


def parse_summary(text):
    """{(b, 'learn'|'none', 'off'|'on'): {'rc', 'mod'(None 가능), 'diag': {side: (mL, mR, kL, kR)}}} — 같은 키가 여러 번이면 마지막 줄."""
    out = {}
    for ln in text.splitlines():
        h = HD.match(ln.strip())
        if not h:
            continue
        m = MOD.search(h.group(5))
        dg = {d.group(1): tuple(float(d.group(i)) for i in (2, 3, 4, 5)) for d in DG.finditer(h.group(5))}
        out[(int(h.group(1)), WMAP[h.group(2)], h.group(3))] = {"rc": int(h.group(4)), "mod": float(m.group(1)) if m else None, "diag": dg}
    return out


def brain_check(b, EV, SATD):
    """(통과 여부, 사유 목록, 포화 여부 또는 None)."""
    why = []
    recs = {k: SATD.get((b,) + k) for k in J.EVS}
    if any(r is None or r["rc"] != 0 or r["mod"] is None or set(r["diag"]) != {"left", "right"} for r in recs.values()):
        return False, ["재측정 결측·실패"], None
    for k, r in recs.items():
        if (b,) + k not in EV:
            return False, ["본실험 평가 줄 결측"], None
        if abs(J.i4(r["mod"]) - J.i4(EV[(b,) + k])) > REPRO:
            why.append("재현 실패 %s %s(%+.4f vs %+.4f)" % (k[0], k[1], r["mod"], EV[(b,) + k]))
    rates = [x for w in ("learn", "none") for s in ("left", "right") for x in recs[(w, "on")]["diag"][s][:2]]
    sat = all(x >= SAT for x in rates)
    if sat:
        why.append("켬 평가 포화(motor 최소 %.4f)" % min(rates))
    return (not why), why, sat


def amend(res, EV, SATD):
    """res = judge_e177.judge 의 결과 사전(None 이면 사전 판정 결측). 반환 (정정 판정 문자열, 뇌별 (통과, 사유, 포화))."""
    per = {b: brain_check(b, EV, SATD) for b in J.BRAINS}
    if res is None:
        return "보류(사전 판정 결측)", per
    if not res["ok"]:
        return res["verdict"], per
    bad = [b for b in J.BRAINS if not per[b][0]]
    if bad:
        nsat = sum(1 for b in J.BRAINS if per[b][2])
        return "보류(정정 1 — 측정 타당성 실패 %d/5: 켬 평가 포화 %d/5, 그 밖 %d/5) — 사전 판정은 '%s'" % (
            len(bad), nsat, len(bad) - sum(1 for b in bad if per[b][2] and len(per[b][1]) == 1), res["verdict"]), per
    return res["verdict"], per


def main():
    K, TRN, EV, S, RAW, EL = J.load()
    checks, res = J.judge(K, TRN, EV, S, RAW, EL)
    print("[사전 판정 — judge_e177.py 그대로]")
    J.report(checks, res, K, TRN)
    f = os.path.join(J.EXP, "logs", "E177", "sat_check", "main", "summary.out")
    SATD = parse_summary(open(f, encoding="utf-8", errors="replace").read()) if os.path.exists(f) else {}
    v, per = amend(res, EV, SATD)
    print("[정정 1 — 측정 타당성]")
    for b in J.BRAINS:
        ok, why, sat = per[b]
        r_on = [SATD.get((b, w, "on")) for w in ("learn", "none")]
        rates = " ".join("%s %s" % (w, " ".join("%.4f/%.4f" % r["diag"][s][:2] for s in ("left", "right"))) for w, r in zip(("학습", "무학습"), r_on) if r and set(r["diag"]) == {"left", "right"})
        print("b%d %s | 켬 motor L/R(자극 좌·우) %s | %s" % (b, "통과" if ok else "실패", rates, "; ".join(why) if why else "-"))
    if res is not None:
        n_off = sum(1 for b in J.BRAINS if res["eoff"][b] <= -1000)
        print("[관측 — 판정 밖] 끔 규칙 학습(e_off ≤ −0.10) %d/5" % n_off)
    print("정정 판정: %s" % v)


if __name__ == "__main__":
    main()
