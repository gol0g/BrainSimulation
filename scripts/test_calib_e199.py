#!/usr/bin/env python3
"""calib_e199 합성 시험 — 강한 선택 반응 KC 중 |Δ| ≥ 250 비율을 η 마다 정해 두고 선택이 사전 규칙(비율 ≥ 0.5 인 가장 작은 η, 없으면 2.4)과 같은지.
실행: python3 scripts/test_calib_e199.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calib_e199 as C  # noqa: E402

NK = 3000
POPS = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")


def build(root, fr, mod=None):
    """fr[η] = 강한 선택 반응 KC 중 |Δ| = 260(≥ 250) 비율(나머지 |Δ| = 100). 쪽마다 강한 KC 100 개(0~99), 약한 반응 KC 100 개(100~199)."""
    mod = mod or {}
    E = os.path.join(root, "research", "experiments")
    for d in ("traces/E176/snap", "traces/E197/pathcheck", "traces/E199/calib", "logs/E199/calib"):
        os.makedirs(os.path.join(E, d), exist_ok=True)
    pre = np.repeat(np.arange(NK), 2).astype(np.uint32)
    np.savez(os.path.join(E, "traces/E176/snap/conn_b15.npz"), **{p + "__pre": pre for p in POPS})
    Z = {}
    for sd in "lr":
        sx = np.zeros(NK); sx[:100] = np.where(np.arange(100) % 2 == 0, 0.8, -0.6); sx[100:200] = 0.2
        rp = np.zeros(NK, bool); rp[:200] = True
        Z["sxi_" + sd] = sx; Z["resp_" + sd] = rp; Z["tot_" + sd] = np.ones(NK); Z["csi_" + sd] = np.zeros(NK)
    np.savez(os.path.join(E, "traces/E197/pathcheck/sxi_b15.npz"), **Z)
    for et in C.ETAS:
        if mod.get(et) == "missing":
            continue
        d = np.zeros(NK)
        k = int(round(fr[et] * 100))
        d[:100] = 100.0; d[:k] = mod.get("dv", {}).get(et, 260.0)
        d[100:200] = 260.0                                     # 약한 KC 는 판정에 안 씀(보고만)
        d[1::2] *= -1                                          # 부호 섞기(크기만 본다)
        W = {}
        for kn in "lr":
            ml = 150 - d / 2 if kn == "l" else 150 + d / 2
            W["kc_%s_to_motor_l" % kn] = np.repeat(ml, 2).astype(np.float32)
            W["kc_%s_to_motor_r" % kn] = np.repeat(300 - ml, 2).astype(np.float32)
        if mod.get(et) == "sums":
            W["kc_l_to_motor_l"][10:12] += 30
        np.savez(os.path.join(E, "traces/E199/calib/w_b15_eta%s.npz" % et), **W)
        with open(os.path.join(E, "logs/E199/calib/train_b15_eta%s.log" % et), "w", encoding="utf-8") as f:
            f.write("    KC→motor [E199 가산형(경성 상한) 갱신 — g += eta·DA·e, clip]\n")
            f.write("    KC→motor [E109 R-STDP 4방향]: init_w=150.0, w_max=300.0, eta=%s, tau_e=12.0, sparsity=0.25\n" % ("9.9" if mod.get(et) == "eta" else et))
            f.write("[학습] 30ep 완료, 보상 1600회\n")
            f.write("[E199 경쟁 정규화] 적용 3000회 · 정규화 전 max|S−목표| 평균 1.0000 최대 2.0000 · 정규화 뒤 최대 4.00e-13 · hard=1\n")


def run(fr, mod=None):
    root = tempfile.mkdtemp()
    try:
        build(root, fr, mod)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            C.main(root)
        out = buf.getvalue()
        p = os.path.join(root, "research/experiments/logs/E199/calib_eta.txt")
        txt = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        return out, txt
    finally:
        shutil.rmtree(root)


CASES = [
    ("C1 0.2·0.6·0.9 → 0.6", {"0.15": 0.2, "0.6": 0.6, "2.4": 0.9}, None, "선택 eta=0.6 "),
    ("C2 0.15 에서 0.5(경계) → 0.15", {"0.15": 0.5, "0.6": 0.8, "2.4": 1.0}, None, "선택 eta=0.15 "),
    ("C3 기준 충족 없음 → 2.4", {"0.15": 0.1, "0.6": 0.2, "2.4": 0.4}, None, "선택 eta=2.4 (사전 규칙: 기준 충족 η 없음"),
    ("C4 η 0.6 로그 eta 불일치 → 보정 실패", {"0.15": 0.2, "0.6": 0.6, "2.4": 0.9}, {"0.6": "eta"}, "보정: 선택 없음(보정 실패)"),
    ("C5 η 2.4 KC 합 ≠ 300 → 보정 실패", {"0.15": 0.2, "0.6": 0.6, "2.4": 0.9}, {"2.4": "sums"}, "보정: 선택 없음(보정 실패)"),
    ("C6 η 0.15 결측 → 보정 실패", {"0.15": 0.2, "0.6": 0.6, "2.4": 0.9}, {"0.15": "missing"}, "보정: 선택 없음(보정 실패)"),
    ("C7 |Δ| = 250 정확(경계 포함) → 0.15", {"0.15": 0.5, "0.6": 0.8, "2.4": 1.0}, {"dv": {"0.15": 250.0}}, "선택 eta=0.15 "),
    ("C8 |Δ| = 249.9 → 0.15 미달 → 0.6", {"0.15": 0.5, "0.6": 0.8, "2.4": 1.0}, {"dv": {"0.15": 249.9}}, "선택 eta=0.6 "),
]


def main():
    ok = 0
    for name, fr, mod, want in CASES:
        out, txt = run(fr, mod)
        good = want in out and (("선택 없음" in want and txt == "") or ("선택 없음" not in want and want in txt))
        ok += good
        print("[%s] %s → %s" % ("통과" if good else "실패", name, [ln for ln in out.splitlines() if "선택" in ln or "실패" in ln][-1:]))
        if not good:
            print(out, txt)
    print("합성 %d/%d" % (ok, len(CASES)))
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
