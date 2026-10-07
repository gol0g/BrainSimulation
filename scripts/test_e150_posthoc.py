#!/usr/bin/env python3
"""e150_posthoc.py 합성 시험: 답을 아는 추적·평가 로그·E148 판정 줄로 행동 일치 비율·|Δv|·T·블록 보상을 확인한다.
실행: python3 scripts/test_e150_posthoc.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e150_posthoc as P


def build(td):
    for d in ("logs/E150", "logs/E148", "traces/E150"):
        os.makedirs(os.path.join(td, d))
    lines = []
    for b in P.BRAINS:
        AB = np.zeros((3000, 37)); AB[:, 2] = np.arange(3000) % 2; AB[:, 5] = 0.1
        AB[1500:, 7] = (np.arange(1500) % 100 < 40 + b).astype(float)        # 과제 B 블록 보상 = 40+b
        A = AB[:1500].copy(); A[7, 6] = 1.0; A[9, 5] = 0.1 + 0.25               # 1 시행 행동 다름, |Δv| 0.25
        np.savez_compressed(os.path.join(td, "traces", "E150", "tr_A_b%d.npz" % b), rows=A)
        np.savez_compressed(os.path.join(td, "traces", "E150", "tr_AB_b%d.npz" % b), rows=AB)
        vals = {("A", "base"): -0.0600, ("AB", "base"): -0.0300, ("AB", "bad"): 0.1000, ("none", "base"): 0.0300, ("none", "bad"): 0.0200}
        for (w, s), m in vals.items():   # eA1 −0.09, eA −0.06, eB +0.08 → T = 0.03/0.08 = 0.375
            open(os.path.join(td, "logs", "E150", "ev_b%d_%s_%s.log" % (b, w, s)), "w", encoding="utf-8").write(
                "x\n=> DECOMP mode=all mod=%+.4f acc=0.0\n" % m)
        lines.append("b%d 과제 A: 사후 +0.1181 eA +0.1000 (과제 A 만 학습 -0.3000, 유지 몫 -0.33) | 과제 B: eB +0.3200 | 학습 사후 +0.1181 보상 1969 | 과제 B 구간 블록 보상 37 58 62" % b)
    open(os.path.join(td, "logs", "E148", "judge.out"), "w", encoding="utf-8").write("[조작검증] x\n" + "\n".join(lines) + "\n판정: x\n")


def build_as(td, eid):
    build(td)
    if eid != "E150":
        os.rename(os.path.join(td, "logs", "E150"), os.path.join(td, "logs", eid))
        os.rename(os.path.join(td, "traces", "E150"), os.path.join(td, "traces", eid))


ok = True
for eid in ("E150", "E151"):
  with tempfile.TemporaryDirectory() as td:
    build_as(td, eid)
    P.EXP = td; P.EID = eid
    with contextlib.redirect_stdout(io.StringIO()):
        R = P.main()
  print("[%s]" % eid)
  for b in P.BRAINS:
    r = R[b]
    chk = (abs(r["same"] - 1499 / 1500) < 1e-12, abs(r["dv"] - 0.25) < 1e-9, abs(r["T"] - 0.375) < 1e-9,
           abs(r["T8"] - 0.4 / 0.32) < 1e-9, r["blk"] == [40 + b] * 15, abs(r["eB8"] - 0.32) < 1e-12)
    ok &= all(chk)
    print("b%d %s %s" % (b, chk, "✓" if all(chk) else "✗"))
print("전체: %s" % ("통과" if ok else "실패"))
sys.exit(0 if ok else 1)
