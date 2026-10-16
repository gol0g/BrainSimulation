#!/usr/bin/env python3
"""judge_e199 합성 시험 — 임시 저장소에 가짜 학습·평가 로그·가중치·스냅숏을 만들어 판정이 사전 규칙과 같은지 본다.
실행: python3 scripts/test_judge_e199.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e199 as J  # noqa: E402

NK = 3000
CELLS = (("X", "A", -1), ("X", "B", +1), ("Y", "A", +1), ("Y", "B", -1))
NONE_M = {("X", "A", "train"): 500, ("X", "A", "new"): 400, ("X", "B", "train"): -300, ("X", "B", "new"): -200,
          ("Y", "A", "train"): 100, ("Y", "A", "new"): 50, ("Y", "B", "train"): -600, ("Y", "B", "new"): -500}
POPS = J.POPS


def fmt(v):
    return "%+.4f" % (v / 10000.0)


def write_train(path, n_comp=3000, d1=4.5e-13, hard=True, hard_line=True, eta=0.6, cchk=3, nt=3000):
    with open(path, "w", encoding="utf-8") as f:
        if hard_line:
            f.write("    KC→motor [E199 가산형(경성 상한) 갱신 — g += eta·DA·e, clip]\n")
        f.write("    KC→motor [E109 R-STDP 4방향]: init_w=150.0, w_max=300.0, eta=%s, tau_e=12.0, sparsity=0.25\n" % eta)
        f.write("[E153 종류 입력 적재] x.npz 검증 일치 — a=1\n")
        f.write("[맥락 범주 과제] catctx — 맥락 X·Y 반반\n")
        for t in range(3):
            f.write("[범주 과제 점검] 시행 %d 범주 B 사례 5 쪽 left 정답 교차 — 행동 창 끝 특징 발화 120 · 시행 끝 0\n" % t)
            if t < cchk:
                f.write("[맥락 범주 점검] 시행 %d 맥락 Y 범주 B 정답 교차 — 행동 창 끝 켠 맥락 발화 600 · 끈 맥락 발화 0 · 시행 끝 두 맥락 발화 0\n" % t)
        f.write("[E199 경쟁 정규화] 시행 1 적용 — 정규화 전 max|S−300| 쪽 l 0.1·r 0.1 → 뒤 1e-13·1e-13\n")
        f.write("[학습] 30ep 완료, 보상 1573회 (탐색 주입 1807회, ε=0.60)\n")
        f.write("[E199 경쟁 정규화] 적용 %d회 · 정규화 전 max|S−목표| 평균 1.2345 최대 5.6789 · 정규화 뒤 최대 %.2e · hard=%d\n" % (n_comp, d1, int(hard)))
        f.write("[범주 과제] 시행 %d 중 범주 B(같은 쪽) 1523 · A(교차) %d\n" % (nt, nt - 1523))
        f.write("[맥락 범주 과제] 시행 %d 중 맥락 X 1455 · Y %d · 정답 같은 쪽 1522 · 교차 %d\n" % (nt, nt - 1455, nt - 1522))


def write_eval(path, ctx, mods, pushed, sat=False, trials=96):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[E153 종류 입력 적재] 검증 일치 n=4\n")
        for c in "AB":
            for s in ("train", "new"):
                mo = "0.700000/0.700000" if sat else "0.500000/0.400000"
                f.write("RCEVAL cat=%s set=%s mod=%s off=+0.0000 acc=50.0 trials=%d feat_on=%d ctx=%s ctx_on=%d | side=left n=48 motor L/R %s\n"
                        % (c, s, fmt(mods[(ctx, c, s)]), trials, trials, ctx, trials, mo))
        f.write("=> DECOMP mode=x rceval A_train=+0.0 pushed=%d\n" % pushed)


def learn_mods(av):
    """av[(x,c)] = (a_train, a_new) → learn mod = none + sg·a."""
    mm = {}
    for x, c, sg in CELLS:
        for s, i in (("train", 0), ("new", 1)):
            mm[(x, c, s)] = NONE_M[(x, c, s)] + sg * av[(x, c)][i]
    return mm


def make_weights(root, b, rng, sums_off=0.0, out_of_range=False, same_as_e193=False):
    pre = np.repeat(np.arange(NK), 2).astype(np.uint32)
    d = rng.uniform(-150, 150, NK)
    d[:50] = 150.0; d[50:100] = -150.0                         # 경계(0·300) 시냅스
    W, S, W3 = {}, {}, {}
    for kn in "lr":
        W["kc_%s_to_motor_l" % kn] = (150 + np.repeat(d, 2)).astype(np.float32)
        W["kc_%s_to_motor_r" % kn] = (150 - np.repeat(d, 2)).astype(np.float32)
    if sums_off:
        W["kc_l_to_motor_l"][200:202] += sums_off
    if out_of_range:
        W["kc_r_to_motor_r"][0] = 301.0
    for p in POPS:
        S[p + "__pre"] = pre
        S[p + "__post"] = np.zeros(pre.size, np.uint32)
        W3[p] = W[p].copy() if same_as_e193 else np.full(pre.size, 150.0, np.float32)
    for d_ in ("traces/E199", "traces/E193", "traces/E176/snap", "traces/E196"):
        os.makedirs(os.path.join(root, "research/experiments", d_), exist_ok=True)
    np.savez(os.path.join(root, "research/experiments/traces/E199/w_b%d.npz" % b), **W)
    np.savez(os.path.join(root, "research/experiments/traces/E193/w_b%d.npz" % b), **W3)
    np.savez(os.path.join(root, "research/experiments/traces/E176/snap/conn_b%d.npz" % b), **S)
    Z = {}
    for sd in "lr":
        Z["sxi_" + sd] = np.round(rng.uniform(-1, 1, NK), 3); Z["resp_" + sd] = rng.rand(NK) < 0.2
        Z["csi_" + sd] = np.zeros(NK); Z["tot_" + sd] = np.ones(NK)
    np.savez(os.path.join(root, "research/experiments/traces/E196/sxi_b%d.npz" % b), **Z)


def build(root, av_by_brain, mod=None, calib=True, calib_eta=0.6):
    E = os.path.join(root, "research/experiments")
    for d_ in ("logs/E199", "logs/E193"):
        os.makedirs(os.path.join(E, d_), exist_ok=True)
    if calib:
        with open(os.path.join(E, "logs/E199/calib_eta.txt"), "w", encoding="utf-8") as f:
            f.write("[E199 보정] 선택 eta=%s (사전 규칙)\n" % calib_eta)
    mod = mod or {}
    for b in J.BRAINS:
        rng = np.random.RandomState(b)
        o = mod.get(b, {})
        make_weights(root, b, rng, o.get("sums_off", 0.0), o.get("out_of_range", False), o.get("same_as_e193", False))
        if not o.get("no_train"):
            write_train(os.path.join(E, "logs/E199/train_b%d.log" % b), n_comp=o.get("n_comp", 3000), d1=o.get("d1", 4.5e-13),
                        hard=o.get("hard", True), hard_line=o.get("hard_line", True), eta=o.get("eta", calib_eta), cchk=o.get("cchk", 3))
        lm = learn_mods(av_by_brain[b])
        for x in "XY":
            write_eval(os.path.join(E, "logs/E199/ev_b%d_learn_%s.log" % (b, x)), x, lm, o.get("pushed_learn", 8),
                       sat=o.get("sat", False), trials=o.get("trials", 96))
            nm = dict(NONE_M)
            if o.get("none_off"):
                nm[("X", "A", "train")] += o["none_off"]
            write_eval(os.path.join(E, "logs/E199/ev_b%d_none_%s.log" % (b, x)), x, nm, 0)
            write_eval(os.path.join(E, "logs/E193/ev_b%d_none_%s.log" % (b, x)), x, NONE_M, 0)
            write_eval(os.path.join(E, "logs/E193/ev_b%d_learn_%s.log" % (b, x)), x, learn_mods({k: (200, 100) for k in av_by_brain[b]}), 8)


def run(av_by_brain, mod=None, calib=True, calib_eta=0.6):
    root = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        build(root, av_by_brain, mod, calib, calib_eta)
        os.chdir(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
        return buf.getvalue()
    finally:
        os.chdir(cwd)
        shutil.rmtree(root)


def uni(tr, nw):
    return {(x, c): (tr, nw) for x, c, _ in CELLS}


ALL = lambda av: {b: av for b in J.BRAINS}  # noqa: E731
CASES = [
    ("T01 맥락 의존 범주 5/5 → H120", ALL(uni(2000, 1500)), None, {}, "판정: 경쟁형 규칙으로 L4 학습 — 맥락 의존 범주(H120)"),
    ("T02 실패 5/5 → H120-null", ALL(uni(200, 100)), None, {}, "판정: 획득 실패(H120-null)"),
    ("T03 부분(3칸) 5/5 → 부분", ALL({("X", "A"): (2000, 1500), ("X", "B"): (2000, 1500), ("Y", "A"): (2000, 1500), ("Y", "B"): (500, 300)}), None, {}, "판정: 부분"),
    ("T04 사례 한정 5/5", ALL(uni(2000, 400)), None, {}, "판정: 맥락 의존(사례 한정)"),
    ("T05 혼합(범주 2·부분 2·실패 1) → 보류", {10: uni(2000, 1500), 11: uni(2000, 1500), 12: {**uni(2000, 1500), ("Y", "B"): (0, 0)},
                                         13: {**uni(2000, 1500), ("Y", "B"): (0, 0)}, 14: uni(0, 0)}, None, {}, "판정: 보류\n"),
    ("T06 학습 로그 없음 → 보류(결측)", ALL(uni(2000, 1500)), {12: {"no_train": True}}, {}, "판정: 보류(결측"),
    ("T07 정규화 2999회 → 조작검증 실패", ALL(uni(2000, 1500)), {11: {"n_comp": 2999}}, {}, "판정: 보류(조작검증 실패)"),
    ("T08 정규화 뒤 잔차 0.01 → 조작검증 실패", ALL(uni(2000, 1500)), {11: {"d1": 0.01}}, {}, "판정: 보류(조작검증 실패)"),
    ("T09 가산형 줄 없음 → 조작검증 실패", ALL(uni(2000, 1500)), {13: {"hard_line": False}}, {}, "판정: 보류(조작검증 실패)"),
    ("T10 학습 eta ≠ 보정 → 조작검증 실패", ALL(uni(2000, 1500)), {10: {"eta": 0.15}}, {}, "판정: 보류(조작검증 실패)"),
    ("T11 가중치 범위 밖 → 조작검증 실패", ALL(uni(2000, 1500)), {14: {"out_of_range": True}}, {}, "판정: 보류(조작검증 실패)"),
    ("T12 KC 합 ≠ 300 → 조작검증 실패", ALL(uni(2000, 1500)), {12: {"sums_off": 20.0}}, {}, "판정: 보류(조작검증 실패)"),
    ("T13 E193 가중치와 같음 → 조작검증 실패", ALL(uni(2000, 1500)), {12: {"same_as_e193": True}}, {}, "판정: 보류(조작검증 실패)"),
    ("T14 평가 포화 → 조작검증 실패", ALL(uni(2000, 1500)), {10: {"sat": True}}, {}, "판정: 보류(조작검증 실패)"),
    ("T15 none 회귀 차 3 → 조작검증 실패", ALL(uni(2000, 1500)), {11: {"none_off": 3}}, {}, "판정: 보류(조작검증 실패)"),
    ("T16 none 회귀 차 2(경계) → 통과", ALL(uni(2000, 1500)), {11: {"none_off": 2}}, {}, "판정: 경쟁형 규칙으로 L4 학습"),
    ("T17 learn pushed 0 → 조작검증 실패", ALL(uni(2000, 1500)), {13: {"pushed_learn": 0}}, {}, "판정: 보류(조작검증 실패)"),
    ("T18 trials 95 → 조작검증 실패", ALL(uni(2000, 1500)), {14: {"trials": 95}}, {}, "판정: 보류(조작검증 실패)"),
    ("T19 보정 파일 없음 → 보류(결측)", ALL(uni(2000, 1500)), None, {"calib": False}, "판정: 보류(결측 — 보정 학습률"),
    ("T20 a_train 1000 경계 = 획득", ALL(uni(1000, 500)), None, {}, "판정: 경쟁형 규칙으로 L4 학습"),
    ("T21 a_train 999 = 미획득 → 0칸 실패", ALL(uni(999, 600)), None, {}, "판정: 획득 실패(H120-null)"),
    ("T22 a_new 499 → 일반화 아님(사례 한정)", ALL(uni(1000, 499)), None, {}, "판정: 맥락 의존(사례 한정)"),
    ("T23 맥락 점검 2줄 → 조작검증 실패", ALL(uni(2000, 1500)), {10: {"cchk": 2}}, {}, "판정: 보류(조작검증 실패)"),
    ("T24 hard=0 → 조작검증 실패", ALL(uni(2000, 1500)), {12: {"hard": False}}, {}, "판정: 보류(조작검증 실패)"),
]


def main():
    ok = 0
    for name, av, mod, kw, want in CASES:
        out = run(av, mod, **kw)
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
