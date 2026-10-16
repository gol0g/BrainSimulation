#!/usr/bin/env python3
"""posthoc_e199 합성 시험 — 손으로 정한 Δ·SXI·보상·칸 효과로 출력 값을 대조.
쪽 l 반응 KC0~9: SXI 강 +0.8 ×5·약 +0.2 ×5, Δ_E199 = 300·300·260·−300·100 | 300·−300·50·−50·0, Δ_E193 = 10·−5·3·−2·4·6·−1·2·3·1.
정답: |Δ|≥250 6/10, 부호 일치 전체 6/9·강 4/5·약 2/4·가중 0.750, E193 부호와 같음 7/9, motor_l 3·motor_r 6,
경계 시냅스 = (±300 인 KC 5 개 × 2 집단 × 2) / 24000 = 0.001, 보상 1700·1573, 칸 평균 E199 +0.2000·E193 +0.0200·lsgnR +0.1500.
실행: python3 scripts/test_posthoc_e199.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_e199 as P  # noqa: E402
from judge_e193 import CELLS  # noqa: E402

NK = 3000
POPS = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")


def wfile(path, dl):
    """쪽 l: 교차 = motor_r = 150 + Δ/2, 같은 쪽 = motor_l = 150 − Δ/2; 쪽 r 은 150."""
    d = np.zeros(NK); d[:10] = dl
    W = {"kc_l_to_motor_l": np.repeat(150 - d / 2, 2).astype(np.float32), "kc_l_to_motor_r": np.repeat(150 + d / 2, 2).astype(np.float32),
         "kc_r_to_motor_l": np.full(2 * NK, 150, np.float32), "kc_r_to_motor_r": np.full(2 * NK, 150, np.float32)}
    np.savez(path, **W)


def evlog(path, ctx, a_tr, a_nw, pushed):
    with open(path, "w", encoding="utf-8") as f:
        for x, c, sg in CELLS:
            if x != ctx:
                continue
            for s, a in (("train", a_tr), ("new", a_nw)):
                f.write("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0000 acc=50.0 trials=96 feat_on=96 ctx=%s ctx_on=96 | side=left n=48 motor L/R 0.500000/0.400000\n" % (c, s, sg * a, ctx))
        f.write("=> DECOMP mode=x rceval A_train=+0.0 pushed=%d\n" % pushed)


def build(root):
    E = os.path.join(root, "research", "experiments")
    for d in ("traces/E199", "traces/E193", "traces/E176/snap", "traces/E196", "logs/E199", "logs/E193", "logs/E198"):
        os.makedirs(os.path.join(E, d), exist_ok=True)
    for b in P.BRAINS:
        pre = np.repeat(np.arange(NK), 2).astype(np.uint32)
        np.savez(os.path.join(E, "traces/E176/snap/conn_b%d.npz" % b), **{p + "__pre": pre for p in POPS})
        wfile(os.path.join(E, "traces/E199/w_b%d.npz" % b), [300, 300, 260, -300, 100, 300, -300, 50, -50, 0])
        wfile(os.path.join(E, "traces/E193/w_b%d.npz" % b), [10, -5, 3, -2, 4, 6, -1, 2, 3, 1])
        Z = {}
        for sd in "lr":
            Z["sxi_" + sd] = np.zeros(NK); Z["resp_" + sd] = np.zeros(NK, bool); Z["tot_" + sd] = np.zeros(NK); Z["csi_" + sd] = np.zeros(NK)
        Z["sxi_l"][:5] = 0.8; Z["sxi_l"][5:10] = 0.2; Z["resp_l"][:10] = True
        np.savez(os.path.join(E, "traces/E196/sxi_b%d.npz" % b), **Z)
        open(os.path.join(E, "logs/E199/train_b%d.log" % b), "w", encoding="utf-8").write("[학습] 30ep 완료, 보상 1700회 (탐색 주입 1800회, ε=0.60)\n")
        open(os.path.join(E, "logs/E193/train_b%d.log" % b), "w", encoding="utf-8").write("[학습] 30ep 완료, 보상 1573회 (탐색 주입 1807회, ε=0.60)\n")
        for x in "XY":
            evlog(os.path.join(E, "logs/E199/ev_b%d_learn_%s.log" % (b, x)), x, 0.2, 0.1, 8)
            evlog(os.path.join(E, "logs/E199/ev_b%d_none_%s.log" % (b, x)), x, 0.0, 0.0, 0)
            evlog(os.path.join(E, "logs/E193/ev_b%d_learn_%s.log" % (b, x)), x, 0.02, 0.01, 8)
            evlog(os.path.join(E, "logs/E193/ev_b%d_none_%s.log" % (b, x)), x, 0.0, 0.0, 0)
            evlog(os.path.join(E, "logs/E198/ev_b%d_lsgnR_%s.log" % (b, x)), x, 0.15, 0.1, 4)
            evlog(os.path.join(E, "logs/E198/ev_b%d_none_%s.log" % (b, x)), x, 0.0, 0.0, 0)


def main():
    root = tempfile.mkdtemp()
    try:
        build(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.main(root)
        out = buf.getvalue()
    finally:
        shutil.rmtree(root)
    checks = [
        ("보상", "보상 E199 1700 · E193 1573"),
        ("경계·|Δ|≥250", "경계 시냅스 0.001 · 반응 KC |Δ|≥250 6/10"),
        ("부호 일치 강/약·가중", "부호 일치 전체 6/9(0.667) · 강 4/5(0.800) · 약 2/4(0.500) · |SXI| 가중 0.750"),
        ("E193 부호와 같음", "E193 부호와 같음 7/9(0.778)"),
        ("좌우 수", "300 받는 반응 KC motor_l 3 · motor_r 6"),
        ("칸 값", "칸 노출 E199 +0.2000 +0.2000 +0.2000 +0.2000 · 새 +0.1000 +0.1000 +0.1000 +0.1000 | E193 +0.0200 +0.0200 +0.0200 +0.0200 | E198 lsgnR +0.1500"),
        ("요약 E199", "e199   노출 평균 +0.2000 범위 +0.2000~+0.2000 ≥0.10 20/20 정답 방향 20/20"),
        ("요약 E193", "e193   노출 평균 +0.0200 범위 +0.0200~+0.0200 ≥0.10 0/20 정답 방향 20/20"),
        ("요약 lsgnR", "lsgnR  노출 평균 +0.1500"),
    ]
    ok = 0
    for nm, s in checks:
        good = s in out
        ok += good
        print("[%s] %s" % ("통과" if good else "실패", nm))
    if ok != len(checks):
        print(out)
    print("합성 %d/%d" % (ok, len(checks)))
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
