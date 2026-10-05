#!/usr/bin/env python3
"""E141 독립 대조 — judge_e141.py 를 쓰지 않고 원 자료에서 다시 계산한다.
- 효과: 요약 줄(E141.log)이 아니라 런별 원 로그(logs/E141/b*.log)의 [사전]·[사후] 줄에서.
- 조작검증: 추적 npz 를 시행별로 — 역할별 잔차를 시행 단위 최대값으로도 보고, 결정 단계 흔적은 E139 같은 뇌와 시행 평균 비.
- 판정: criteria_fixed.txt 문장을 이 파일 안에서 다시 구현(정수 단위 비교 — 4자리 값을 1e4 배 정수로).
실행: python3 scripts/verify_e141_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
E119 = {10: (195, -1032), 11: (150, -724), 12: (262, -689), 13: (320, -809), 14: (165, -607)}   # ([사전], 효과) × 1e4
RS = (11.0 / 12.0) ** 20
PAT = re.compile(r"변조폭 ([-+]?\d+\.\d{4})")


def i4(s):
    """'+0.0195' → 195 (정수, 1e-4 단위). 4자리가 아니면 실패."""
    m = re.fullmatch(r"([-+]?)(\d+)\.(\d{4})", s)
    if not m:
        raise ValueError("4자리 값 아님: %r" % s)
    v = int(m.group(2)) * 10000 + int(m.group(3))
    return -v if m.group(1) == "-" else v


def raw(b):
    pre = post = rew = None
    for ln in open(os.path.join(EXP, "logs", "E141", "b%d.log" % b), encoding="utf-8"):
        if ln.startswith("[사전]"):
            pre = i4(PAT.search(ln).group(1))
        elif ln.startswith("[사후]"):
            post = i4(PAT.search(ln).group(1))
        elif ln.startswith("[학습]"):
            rew = int(re.search(r"보상 (\d+)회", ln).group(1))
    return pre, post, rew


def tr(path):
    R = np.load(path)["rows"]
    rw = R[:, 7] == 1
    out = {"n": R.shape[0]}
    # 시행·역할별 잔차: |e_end − r*·e_da| / max(|e_da|, 1) 의 최대(흔적이 0 인 칸은 분모 1)
    num = np.abs(R[:, 21:25] - RS * R[:, 13:17])
    out["res_max"] = float((num / np.maximum(np.abs(R[:, 13:17]), 1.0)).max())
    out["res_sum"] = float(num.sum() / np.abs(R[:, 13:17]).sum())
    out["eda"] = float(np.mean(np.abs(R[:, 13]) + np.abs(R[:, 14])))
    out["pre"] = float(abs(R[:, 17:21].sum()) / abs(R[:, 12].sum()))
    a, bb = R[rw, 8].sum(), R[rw, 9].sum()
    c, p = R[~rw, 8].sum(), R[~rw, 9].sum()
    out["BA"], out["CP"] = bb / a, c / p
    out["dD"] = float(R[:, 8].sum() - R[:, 9].sum())
    out["rew_rows"] = int(rw.sum())
    return out


def main():
    ok = True
    rows = {}
    for b in sorted(E119):
        pre, post, rew = raw(b)
        t = tr(os.path.join(EXP, "traces", "E141", "tr_b%d.npz" % b))
        t0 = tr(os.path.join(EXP, "traces", "E139", "tr_b%d.npz" % b))
        rows[b] = (pre, post, rew, t, t0)
    print("뇌  [사전] [사후]   e      d(E119) 보상(로그/추적) | 잔차 합·최대        | 결정흔적/E139 | 도파민전  | B/A    C/P    | ΔD/E139")
    m1 = m1b = m2 = m3 = 0
    es, ds = {}, {}
    for b, (pre, post, rew, t, t0) in rows.items():
        e = post - pre; d = e - E119[b][1]
        es[b], ds[b] = e, d
        m1 += t["res_sum"] <= 1e-3
        m1b += t["eda"] >= 0.25 * t0["eda"]
        m2 += abs(pre - E119[b][0]) <= 20
        m3 += (t["n"] == 500 and t["pre"] <= 1e-3)
        print("b%d %+5d %+6d %+6d %+6d   %d/%d     | %.1e %.1e | %.3f | %.1e | %+.3f %+.3f | %.2f"
              % (b, pre, post, e, d, rew, t["rew_rows"], t["res_sum"], t["res_max"], t["eda"] / t0["eda"], t["pre"], t["BA"], t["CP"], t["dD"] / t0["dD"]))
        ok &= rew == t["rew_rows"]
    print("조작검증: M1 %d/5 M1b %d/5 M2 %d/5 M3 %d/5 | 보상 수 로그=추적 %s" % (m1, m1b, m2, m3, "일치" if ok else "불일치"))
    mean_x5 = sum(es.values())          # 효과 합(1e-4 단위) — 평균 ≤ −0.10 ⇔ 합 ≤ −5000
    if not (m1 == m1b == m2 == m3 == 5):
        v = "보류(조작검증 실패)"
    elif mean_x5 <= -5000 and all(d < 0 for d in ds.values()):
        v = "지지(H064)"
    elif sum(d >= 100 for d in ds.values()) >= 4:
        v = "반대(H064-rev)"
    elif sum(abs(d) < 100 for d in ds.values()) >= 4:
        v = "기각(H064-null)"
    else:
        v = "보류"
    print("효과 평균 %+.4f | d<0 %d/5 | d≥+0.01 %d/5 | |d|<0.01 %d/5 | 기전 B/A<0·C/P<0 %d/5 | ΔD>E139 %d/5"
          % (mean_x5 / 5e4, sum(d < 0 for d in ds.values()), sum(d >= 100 for d in ds.values()), sum(abs(d) < 100 for d in ds.values()),
             sum(r[3]["BA"] < 0 and r[3]["CP"] < 0 for r in rows.values()), sum(r[3]["dD"] > r[4]["dD"] for r in rows.values())))
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
