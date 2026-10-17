#!/usr/bin/env python3
"""judge_e203 합성 시험 — 가짜 학습·평가 로그·가중치로 판정이 사전 규칙과 같은지(활동 가중 정규화 줄·저장 가중치 활동 가중 M 경계 포함).
가짜 자료 함수는 test_judge_e199 를 재사용하고 경로만 E203(비교 E200)으로. 실행: python3 scripts/test_judge_e203.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e203 as J  # noqa: E402
import test_judge_e199 as T9  # noqa: E402

NK = 3000
POPS = T9.POPS


def write_train(path, o):
    T9.write_train(path, n_comp=o.get("n_comp", 3000), eta=o.get("eta", 2.4), cchk=o.get("cchk", 3))
    with open(path, "a", encoding="utf-8") as f:
        for t in range(o.get("chk", 3)):
            f.write("[E200 도파민 기준선] 시행 %d r=+1.00 r̄=0.2500 → DA=+0.7500\n" % (t + 1))
        f.write("[E200 도파민 통계] 시행 %d · 평균 DA %+.4f · 평균 |DA| 0.7400 · 기준선 %d · r̄ 끝 0.2700 · τ 100 · r̄₀ 0.25\n"
                % (o.get("da_n", 3000), o.get("da_mean", 0.0012), o.get("baseline", 1)))
        f.write("[E203 활동 가중 정규화] 반응 KC l 600·r 600 · 가중 합 l 600·r 600 · 파일 x\n")
        for t in range(o.get("a_chk", 3)):
            f.write("[E203 활동 가중 정규화] 시행 %d — M 전 l +1.0000·r -1.0000 → 뒤 1.00e-12·1.00e-12 (반복 1)\n" % (t + 1))
        f.write("[E203 활동 가중 정규화] 적용 %d회 · 정규화 전 |M| 평균 0.1234 최대 0.5678 · 뒤 최대 %.2e · 반복 최대 1\n" % (o.get("a_n", 3000), o.get("a_res", 1e-12)))


def make_weights(root, b, rng, o):
    E = os.path.join(root, "research/experiments")
    for d_ in ("traces/E203", "traces/E200", "traces/E176/snap", "traces/E196"):
        os.makedirs(os.path.join(E, d_), exist_ok=True)
    pre = np.repeat(np.arange(NK), 2).astype(np.uint32)
    Z = {}
    for sd in "lr":
        Z["sxi_" + sd] = np.round(rng.uniform(-1, 1, NK), 3); Z["resp_" + sd] = rng.rand(NK) < 0.2
        Z["csi_" + sd] = np.zeros(NK); Z["tot_" + sd] = rng.randint(1, 6, NK).astype(float)
    W = {}
    for kn in "lr":
        d = rng.uniform(-130, 130, NK); d[:50] = 130.0
        rp, tw = Z["resp_" + kn], Z["tot_" + kn]
        d[rp] -= np.sum(tw[rp] * d[rp]) / np.sum(tw[rp])                   # 반응 KC 활동 가중 평균 0 → M = 0
        W["kc_%s_to_motor_l" % kn] = (150 + np.repeat(d, 2)).astype(np.float32)
        W["kc_%s_to_motor_r" % kn] = (150 - np.repeat(d, 2)).astype(np.float32)
    if o.get("sums_off"):
        W["kc_l_to_motor_l"][200:202] += o["sums_off"]
    if o.get("act_off"):                                                  # KC 합은 그대로·l 쪽 반응 KC 의 교차(l→r) +X/2·같은 쪽(l→l) −X/2 → M_l = +X
        rr = np.repeat(Z["resp_l"], 2)
        W["kc_l_to_motor_r"][rr] += o["act_off"] / 2; W["kc_l_to_motor_l"][rr] -= o["act_off"] / 2
    W9 = {p: (W[p].copy() if o.get("same_as_e199") else np.full(pre.size, 150.0, np.float32)) for p in POPS}
    np.savez(os.path.join(E, "traces/E203/w_b%d.npz" % b), **W)
    if not o.get("no_e199w"):
        np.savez(os.path.join(E, "traces/E200/w_b%d.npz" % b), **W9)
    np.savez(os.path.join(E, "traces/E176/snap/conn_b%d.npz" % b), **{p + "__pre": pre for p in POPS})
    np.savez(os.path.join(E, "traces/E196/sxi_b%d.npz" % b), **Z)


def build(root, av_by_brain, mod=None):
    E = os.path.join(root, "research/experiments")
    for d_ in ("logs/E203", "logs/E200", "logs/E193"):
        os.makedirs(os.path.join(E, d_), exist_ok=True)
    mod = mod or {}
    for b in J.BRAINS:
        rng = np.random.RandomState(b)
        o = mod.get(b, {})
        make_weights(root, b, rng, o)
        if not o.get("no_train"):
            write_train(os.path.join(E, "logs/E203/train_b%d.log" % b), o)
        lm = T9.learn_mods(av_by_brain[b])
        for x in "XY":
            T9.write_eval(os.path.join(E, "logs/E203/ev_b%d_learn_%s.log" % (b, x)), x, lm, 8, sat=o.get("sat", False))
            nm = dict(T9.NONE_M)
            if o.get("none_off"):
                nm[("X", "A", "train")] += o["none_off"]
            T9.write_eval(os.path.join(E, "logs/E203/ev_b%d_none_%s.log" % (b, x)), x, nm, 0)
            T9.write_eval(os.path.join(E, "logs/E193/ev_b%d_none_%s.log" % (b, x)), x, T9.NONE_M, 0)
            T9.write_eval(os.path.join(E, "logs/E200/ev_b%d_learn_%s.log" % (b, x)), x, T9.learn_mods({k: (1500, 900) for k in av_by_brain[b]}), 8)
            T9.write_eval(os.path.join(E, "logs/E200/ev_b%d_none_%s.log" % (b, x)), x, T9.NONE_M, 0)


def run(av, mod=None):
    root = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        build(root, av, mod)
        os.chdir(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
        return buf.getvalue()
    finally:
        os.chdir(cwd)
        shutil.rmtree(root)


uni, ALL = T9.uni, T9.ALL
CASES = [
    ("T01 맥락 의존 범주 5/5 → H124", ALL(uni(2000, 1500)), None, "판정: 활동 가중 정규화로 L4 학습 — 맥락 의존 범주(H124)"),
    ("T02 실패 5/5 → H124-null", ALL(uni(200, 100)), None, "판정: 획득 실패(H124-null)"),
    ("T03 부분 5/5", ALL({**uni(2000, 1500), ("Y", "B"): (500, 300)}), None, "판정: 부분"),
    ("T04 사례 한정 5/5", ALL(uni(2000, 400)), None, "판정: 맥락 의존(사례 한정)"),
    ("T05 혼합 → 보류", {10: uni(2000, 1500), 11: uni(2000, 1500), 12: {**uni(2000, 1500), ("Y", "B"): (0, 0)},
                       13: {**uni(2000, 1500), ("Y", "B"): (0, 0)}, 14: uni(0, 0)}, None, "판정: 보류\n"),
    ("T06 학습 로그 없음 → 결측", ALL(uni(2000, 1500)), {12: {"no_train": True}}, "판정: 보류(결측"),
    ("T07 정규화 2999회 → 조작검증 실패", ALL(uni(2000, 1500)), {11: {"n_comp": 2999}}, "판정: 보류(조작검증 실패)"),
    ("T08 eta 0.15 → 조작검증 실패", ALL(uni(2000, 1500)), {10: {"eta": 0.15}}, "판정: 보류(조작검증 실패)"),
    ("T09 도파민 통계 2999시행 → 조작검증 실패", ALL(uni(2000, 1500)), {13: {"da_n": 2999}}, "판정: 보류(조작검증 실패)"),
    ("T10 평균 DA +0.0600 → 조작검증 실패", ALL(uni(2000, 1500)), {14: {"da_mean": 0.06}}, "판정: 보류(조작검증 실패)"),
    ("T11 평균 DA −0.0500(경계) → 통과", ALL(uni(2000, 1500)), {14: {"da_mean": -0.05}}, "판정: 활동 가중 정규화로 L4 학습"),
    ("T12 기준선 0 → 조작검증 실패", ALL(uni(2000, 1500)), {12: {"baseline": 0}}, "판정: 보류(조작검증 실패)"),
    ("T13 기준선 점검 2줄 → 조작검증 실패", ALL(uni(2000, 1500)), {11: {"chk": 2}}, "판정: 보류(조작검증 실패)"),
    ("T14 E200 가중치와 같음 → 조작검증 실패", ALL(uni(2000, 1500)), {10: {"same_as_e199": True}}, "판정: 보류(조작검증 실패)"),
    ("T15 KC 합 ≠ 300 → 조작검증 실패", ALL(uni(2000, 1500)), {13: {"sums_off": 20.0}}, "판정: 보류(조작검증 실패)"),
    ("T16 none 회귀 차 3 → 조작검증 실패", ALL(uni(2000, 1500)), {11: {"none_off": 3}}, "판정: 보류(조작검증 실패)"),
    ("T17 평가 포화 → 조작검증 실패", ALL(uni(2000, 1500)), {12: {"sat": True}}, "판정: 보류(조작검증 실패)"),
    ("T18 E200 가중치 없음 → 결측", ALL(uni(2000, 1500)), {14: {"no_e199w": True}}, "판정: 보류(결측"),
    ("T19 맥락 점검 2줄 → 조작검증 실패", ALL(uni(2000, 1500)), {10: {"cchk": 2}}, "판정: 보류(조작검증 실패)"),
    ("T20 a_train 1000·a_new 500 경계 → 범주", ALL(uni(1000, 500)), None, "판정: 활동 가중 정규화로 L4 학습"),
    ("T21 활동 가중 정규화 2999회 → 조작검증 실패", ALL(uni(2000, 1500)), {12: {"a_n": 2999}}, "판정: 보류(조작검증 실패)"),
    ("T22 저장 가중치 활동 가중 M 2 → 조작검증 실패", ALL(uni(2000, 1500)), {11: {"act_off": 2.0}}, "판정: 보류(조작검증 실패)"),
    ("T23 저장 가중치 활동 가중 M 0.9(경계 안) → 통과", ALL(uni(2000, 1500)), {11: {"act_off": 0.9}}, "판정: 활동 가중 정규화로 L4 학습"),
    ("T24 활동 가중 정규화 뒤 잔차 0.01 → 조작검증 실패", ALL(uni(2000, 1500)), {13: {"a_res": 0.01}}, "판정: 보류(조작검증 실패)"),
    ("T25 활동 가중 점검 2줄 → 조작검증 실패", ALL(uni(2000, 1500)), {10: {"a_chk": 2}}, "판정: 보류(조작검증 실패)"),
]


def main():
    ok = 0
    for name, av, mod, want in CASES:
        out = run(av, mod)
        good = want in out
        ok += good
        last = [ln for ln in out.splitlines() if ln.startswith("판정")]
        print("[%s] %s → %s" % ("통과" if good else "실패", name, last[-1] if last else out.strip()[-200:]))
        if not good:
            print(out)
    print("합성 %d/%d" % (ok, len(CASES)))
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
