#!/usr/bin/env python3
"""E150 사후 분석(탐색적 — 사전 판정 아님, 판정은 judge_e150.py). 원 로그·추적에서 직접 계산한다.
1) AB 팔의 앞 1,500시행이 A단독 팔과 같은가: 행동 열(2 side_r, 3 explore, 4 probe_r, 6 ex_r, 7 correct)이 모두 같은 시행 비율, |Δv|(열 5) 최대.
2) 과제 B 학습 단위당 과제 A 자극 이동 T = (eA − eA1)/eB (양수 = 과제 B 규칙 쪽으로 이동) — E150(차단) vs E148(기본, logs/E148/judge.out — 독립 대조 일치 값).
3) 과제 B 구간 블록(100시행) 보상 수 — E150 AB vs E148.
실행: python3 scripts/e150_posthoc.py [E150|E151] (저장소 루트에서; 기본 E150 — E151 도 같은 파일 구조)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
EID = "E150"
BRAINS = (10, 11, 12, 13, 14)
BEH = [2, 3, 4, 6, 7]
E148RE = re.compile(r"^b(\d+) 과제 A: .* eA ([-+]\d+\.\d{4}) \(과제 A 만 학습 ([-+]\d+\.\d{4}), .*\| 과제 B: eB ([-+]\d+\.\d{4}) .*블록 보상 ([\d ]+)$")


def mod(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    return float(re.search(r"^=> DECOMP mode=\S+ mod=([-+]?\d+\.\d{4})", txt, re.M).group(1))


def e150(b):
    L = lambda w, s: mod(os.path.join(EXP, "logs", EID, "ev_b%d_%s_%s.log" % (b, w, s)))
    return L("A", "base") - L("none", "base"), L("AB", "base") - L("none", "base"), L("AB", "bad") - L("none", "bad")


def e148():
    out = {}
    for ln in open(os.path.join(EXP, "logs", "E148", "judge.out"), encoding="utf-8", errors="replace"):
        m = E148RE.match(ln.strip())
        if m:
            out[int(m.group(1))] = (float(m.group(3)), float(m.group(2)), float(m.group(4)), [int(x) for x in m.group(5).split()])
    return out


def main():
    E8 = e148()
    res = {}
    for b in BRAINS:
        A = np.load(os.path.join(EXP, "traces", EID, "tr_A_b%d.npz" % b))["rows"]
        AB = np.load(os.path.join(EXP, "traces", EID, "tr_AB_b%d.npz" % b))["rows"]
        n = len(A)
        same = float(np.mean(np.all(A[:, BEH] == AB[:n, BEH], axis=1)))
        dv = float(np.max(np.abs(A[:, 5] - AB[:n, 5])))
        blk = [int(AB[n + i * 100:n + (i + 1) * 100, 7].sum()) for i in range((len(AB) - n) // 100)]
        eA1, eA, eB = e150(b)
        T = (eA - eA1) / eB if eB else float("nan")
        p = E8.get(b)
        T8 = (p[1] - p[0]) / p[2] if p else float("nan")
        res[b] = {"same": same, "dv": dv, "blk": blk, "T": T, "T8": T8, "eB": eB, "eB8": p[2] if p else float("nan")}
        print(("b%d A단계 행동 일치 %.4f |Δv|max %.2e | " + EID + " T %+.2f (eA1 %+.4f eA %+.4f eB %+.4f) | 기본 E148 T %+.2f (eB %+.4f)")
              % (b, same, dv, T, eA1, eA, eB, T8, res[b]["eB8"]))
        print("    과제 B 블록 보상 %s %s | E148 %s" % (EID, " ".join(map(str, blk)), " ".join(map(str, p[3])) if p else "-"))
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EID = sys.argv[1]
    main()
    sys.exit(0)
