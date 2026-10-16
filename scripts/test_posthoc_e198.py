#!/usr/bin/env python3
"""posthoc_e198 합성 시험 — 손으로 정한 칸 효과·Δ·SXI 로 출력 값을 대조.
정답: sgnR 노출 +0.2000·새 +0.1000, lsgnR +0.1000·+0.0500, lsgnA +0.1500·+0.0750 → lsgnR/sgnR 0.50·lsgnA/sgnR 0.75·새/노출 0.50.
반응 KC(쪽 l 만) 10 개: SXI +0.8 ×5(강)·+0.2 ×5(약), Δ 강 + + + + −, 약 + + − − 0 → 전체 6/9, 강 4/5, 약 2/4, 가중 (4·0.8 + 2·0.2)/(5·0.8 + 4·0.2) = 0.750, Δ=0 1, 300 받는 KC motor_r 6(Δ > 0)·motor_l 3(Δ < 0), motor 최대 0.500.
중앙값 보정: 반응·Δ ≠ 0 KC0~8 의 Δ(10,20,5,7,−9,3,4,−2,−6) 중앙값 4 → Δ−4 = 6,16,1,3,−13,−1,0,−6,−10 → KC6(0) 빼고 4/8, 쪽 r 반응 없음 → 중앙값 0.
실행: python3 scripts/test_posthoc_e198.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e198 as J  # noqa: E402
import posthoc_e198 as P  # noqa: E402
from judge_e196 import CELLS  # noqa: E402

NK = 3000
A = {"sgnR": (2000, 1000), "lsgnR": (1000, 500), "lsgnA": (1500, 750)}


def write_eval(path, ctx, mods):
    with open(path, "w", encoding="utf-8") as f:
        for c in "AB":
            for s in ("train", "new"):
                f.write("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0000 acc=50.0 trials=96 feat_on=96 ctx=%s ctx_on=96 | side=left n=48 motor L/R 0.500000/0.400000\n"
                        % (c, s, mods[(c, s)] / 10000.0, ctx))
        f.write("=> DECOMP mode=x rceval A_train=+0.0 pushed=4\n")


def build(root):
    E = os.path.join(root, "research", "experiments")
    for d in ("logs/E198", "traces/E196", "traces/E193", "traces/E176/snap"):
        os.makedirs(os.path.join(E, d), exist_ok=True)
    for b in J.BRAINS:
        Z = {}
        for sd in "lr":
            Z["sxi_" + sd] = np.zeros(NK); Z["resp_" + sd] = np.zeros(NK, dtype=bool); Z["tot_" + sd] = np.zeros(NK); Z["csi_" + sd] = np.zeros(NK)
        Z["sxi_l"][:5] = 0.8; Z["sxi_l"][5:10] = 0.2; Z["resp_l"][:10] = True
        np.savez(os.path.join(E, "traces/E196/sxi_b%d.npz" % b), **Z)
        dl = np.array([10, 20, 5, 7, -9, 3, 4, -2, -6, 0], dtype=float)      # KC0~9 의 Δ(교차 − 같은 쪽)
        W, S = {}, {}
        k = np.arange(10)
        W["kc_l_to_motor_r"] = (150 + dl).astype(np.float32); S["kc_l_to_motor_r__pre"] = k.astype(np.uint32)       # 교차
        W["kc_l_to_motor_l"] = np.full(10, 150, np.float32); S["kc_l_to_motor_l__pre"] = k.astype(np.uint32)       # 같은 쪽
        for nm in ("kc_r_to_motor_l", "kc_r_to_motor_r"):
            W[nm] = np.array([150.0], np.float32); S[nm + "__pre"] = np.array([0], np.uint32)
        np.savez(os.path.join(E, "traces/E193/w_b%d.npz" % b), **W)
        np.savez(os.path.join(E, "traces/E176/snap/conn_b%d.npz" % b), **S)
        for x in "XY":
            write_eval(os.path.join(E, "logs/E198/ev_b%d_none_%s.log" % (b, x)), x, {(c, s): 0 for c in "AB" for s in ("train", "new")})
            for w, (tr, nw) in A.items():
                mods = {}
                for xx, c, sg in CELLS:
                    if xx == x:
                        mods[(c, "train")] = sg * tr; mods[(c, "new")] = sg * nw
                write_eval(os.path.join(E, "logs/E198/ev_b%d_%s_%s.log" % (b, w, x)), x, mods)


def main():
    root = tempfile.mkdtemp()
    ok = 0; n = 0
    try:
        build(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.main(root)
        out = buf.getvalue()
        checks = [
            ("sgnR 노출·새", "sgnR  노출 평균 +0.2000 범위 +0.2000~+0.2000 ≥0.10 20/20 정답 방향 20/20 | 새 평균 +0.1000"),
            ("lsgnR 노출·새", "lsgnR 노출 평균 +0.1000 범위 +0.1000~+0.1000 ≥0.10 20/20 정답 방향 20/20 | 새 평균 +0.0500"),
            ("lsgnA 노출·새", "lsgnA 노출 평균 +0.1500"),
            ("새/노출 0.50", "새/노출 0.50"),
            ("비율", "b10 lsgnR/sgnR 0.50 · lsgnA/sgnR 0.75 · lsgnR 최소 +0.1000 · 새 최소 +0.0500"),
            ("부호 일치 전체·강·약", "부호 일치 전체 6/9(0.667) · |SXI|≥0.5 4/5(0.800) · <0.5 2/4(0.500)"),
            ("가중·Δ=0", "|SXI| 가중 0.750 · 반응 중 Δ=0 1"),
            ("motor 별 설계 수·motor 최대", "lsgnR 300 받는 KC motor_l 3 · motor_r 6 · 평가 motor 최대 lsgnR 0.500 · sgnR 0.500"),
            ("중앙값 보정 일치", "(탐색) 중앙값 보정 일치 4/8(0.500) — 중앙값 l +4.00 · r +0.00"),
            ("뇌 5 점", "부호 일치율 0.667 0.667 0.667 0.667 0.667 대 lsgnR 노출 평균 +0.1000"),
        ]
        for nm, s in checks:
            n += 1
            good = s in out
            ok += good
            print("[%s] %s" % ("통과" if good else "실패", nm))
        if ok != n:
            print(out)
    finally:
        shutil.rmtree(root)
    print("합성 %d/%d" % (ok, n))
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
