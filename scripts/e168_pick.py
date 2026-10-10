#!/usr/bin/env python3
"""E168 보정 선택 — 기준 logs/E168/criteria_fixed.txt 규칙. 원 로그 logs/E168/calib/W{0,2,5,10,20}_b15.log.
선택 = 다음을 모두 만족하는 가장 작은 W(> 0): 연결 줄 1, 보상 시행 보상 창 KC 발화율 ≤ W=0 의 10%, 결정 단계 KC 발화율 ≥ W=0 의 90%,
[사전] = W=0 [사전] ±0.002(1e-4 정수 ±20). 없으면 'none'(본실험 없이 미해결 종료). 결과를 logs/E168/pick.txt 에 'W=<값>' 으로 쓴다.
부지표(선택 밖): 보상 시행 보상 창 새 흔적 크기(추적, W=0 대비).
실행: python3 scripts/e168_pick.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
WS = (0, 2, 5, 10, 20)
R_STAR = (1.0 - 1.0 / 12.0) ** 20
KL = re.compile(r"^\[E168 KC 발화\] 결정 단계\(3처리 끝\) 평균 ([0-9.na]+) n=(\d+) \| 보상 창 보상 시행 평균 ([0-9.na]+) n=(\d+) \| "
                r"보상 창 처벌 시행 평균 ([0-9.na]+) n=(\d+) \| da_kc_inh=([0-9.]+)", re.M)


def i4(x):
    return int(round(float(x) * 1e4))


def parse(t):
    k = KL.search(t) if t else None
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    b = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    if not (k and a and b):
        return None
    return {"dec": float(k.group(1)), "rew": float(k.group(3)), "pun": float(k.group(5)), "nrew": int(k.group(4)), "w": float(k.group(7)),
            "pre": i4(a.group(1)), "post": i4(b.group(1)), "conn": len(re.findall(r"^  \[E168 도파민→KC억제\]", t, re.M))}


def newtrace(rows):
    rw = rows[:, 7] == 1
    if not rw.any():
        return float("nan")
    return float(np.abs(rows[rw][:, 21:25] - R_STAR * rows[rw][:, 13:17]).sum(axis=1).mean())


def pick(C):
    """C[W] = parse 결과. 반환 (선택 W 또는 None, 행별 판정 사전)."""
    if 0 not in C:
        return None, {}
    z = C[0]
    rows = {}
    for w in WS[1:]:
        c = C.get(w)
        if c is None:
            rows[w] = None
            continue
        # 로그 정밀도(소수 6자리)의 정수로 비교한다 — 합성 시험이 부동소수 경계 결함을 찾음(0.9×0.05 > 0.045, 2026-10-10 첫 판 수정)
        i6 = lambda x: int(round(x * 1e6))
        ok = (c["conn"] == 1 and 10 * i6(c["rew"]) <= i6(z["rew"]) and 10 * i6(c["dec"]) >= 9 * i6(z["dec"]) and abs(c["pre"] - z["pre"]) <= 20)
        rows[w] = ok
    sel = next((w for w in WS[1:] if rows.get(w)), None)
    return sel, rows


def main():
    C, T = {}, {}
    for w in WS:
        f = os.path.join(EXP, "logs", "E168", "calib", "W%d_b15.log" % w)
        t = open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None
        p = parse(t)
        if p:
            C[w] = p
        g = os.path.join(EXP, "traces", "E168", "calib", "tr_W%d_b15.npz" % w)
        if os.path.exists(g):
            T[w] = newtrace(np.load(g)["rows"])
    if 0 not in C or C[0]["conn"] != 0:
        print("[E168 선택] W=0 기준 로그 결측 또는 연결 줄이 있음 — 선택 보류")
        return 0
    sel, rows = pick(C)
    z = C[0]
    for w in WS:
        if w not in C:
            print("  W=%d 결측" % w)
            continue
        c = C[w]
        print("  W=%d 연결 %d | 결정 KC %.6f(%.3f) | 보상 창 KC 보상 %.6f(%.3f) 처벌 %.6f | [사전] %+.4f(차 %+d) [사후] %+.4f | 보상 창 새 흔적(보상 시행) %s | %s"
              % (w, c["conn"], c["dec"], c["dec"] / z["dec"] if z["dec"] else float("nan"), c["rew"], c["rew"] / z["rew"] if z["rew"] else float("nan"),
                 c["pun"], c["pre"] / 1e4, c["pre"] - z["pre"], c["post"] / 1e4,
                 ("%.1f(%.3f)" % (T[w], T[w] / T[0])) if (w in T and 0 in T and T[0]) else "-", "기준" if w == 0 else ("충족" if rows.get(w) else "미충족")))
    os.makedirs(os.path.join(EXP, "logs", "E168"), exist_ok=True)
    with open(os.path.join(EXP, "logs", "E168", "pick.txt"), "w", encoding="utf-8") as fh:
        fh.write("W=%s\n" % (sel if sel is not None else "none"))
    print("[E168 선택] W* = %s" % (sel if sel is not None else "없음 — 본실험 없이 미해결 종료(기준 파일 규칙)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
