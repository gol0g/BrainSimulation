#!/usr/bin/env python3
"""judge_e192.py 합성 시험 — 설정·뇌마다 합성 KC 부호(덧셈·결합·잡음)로 가짜 RCKX 로그를 만들어 판정 범주·조작검증·결측 확인.
실행: python3 scripts/test_judge_e192.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e192 as J  # noqa: E402

N_KC = 3000


def synth_rows(kind, seed, n_ex=6, reps=3):
    rs = np.random.RandomState(seed)
    n_use = 400                                   # 앞 400 KC 만 쓰는 희소 부호(나머지 0)
    base = rs.uniform(0.0, 0.2, n_use)
    cat_u = {c: rs.rand(n_use) < 0.1 for c in "AB"}
    ctx_u = {x: rs.rand(n_use) < 0.1 for x in "XY"}
    comb_u = {(c, x): rs.rand(n_use) < 0.05 for c in "AB" for x in "XY"}
    out = []
    for st in ("train", "new"):
        ex = {c: [rs.rand(n_use) < 0.03 for _ in range(n_ex)] for c in "AB"}
        for c in "AB":
            for i in range(n_ex):
                for x in "XY":
                    for r in range(reps):
                        lam = base + 1.0 * ex[c][i]
                        if kind in ("additive", "conj"):
                            lam = lam + 2.0 * cat_u[c] + 2.0 * ctx_u[x]
                        if kind == "conj":
                            lam = lam + 3.0 * comb_u[(c, x)]
                        if kind == "nocat":                        # 범주 정보 없음(측정 무효 꼴)
                            lam = lam + 2.0 * ctx_u[x]
                        v = rs.poisson(lam)
                        out.append((st, c, x, i, r, v))
    return out


def log_text(kind, seed, p, w, npres=288, ctx_off=0, drop=0, conn_scale=1.0, feat=True, gsame=1):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n"
    n = int(200 * N_KC * p * conn_scale)
    t += "  [맥락 입력] 맥락 전용 집단 zz_ctx 200 → KC 좌·우, w=%.2f, p=%.3f, 연결 좌 %d·우 %d\n" % (w, p, n, n)
    t += "  [맥락 입력 2] 맥락 전용 집단 zz_ctx2 200 → KC 좌·우, w=%.2f, p=%.3f, 연결 좌 %d·우 %d\n" % (w, p, n, n)
    if feat:
        t += "  [원시 특징] 특징 집단 zz_feat 좌·우 각 200 → 같은 쪽 KC, w=24.00, p=0.100, 연결 좌 59816·우 60158, 정적\n"
    k = 0
    for sd in "lr":
        for (st, c, x, i, r, v) in synth_rows(kind, seed + (0 if sd == "l" else 1000)):
            k += 1
            if k <= drop:
                continue
            nz = np.nonzero(v)[0]
            t += "RCKX rep=%d side=%s set=%s cat=%s ctx=%s i=%d n=%d ids=%s\n" % (r, sd, st, c, x, i, nz.size, ",".join("%d:%d" % (j, v[j]) for j in nz))
    t += ("RCKXCHK npres=%d ctx_on_spikes=50000 ctx_off_spikes=%d feat_inactive_spikes=0 base_l=1 base_r=1 g_same=%d ctx_w=%.2f ctx_p=%.3f ctx2_w=%.2f ctx2_p=%.3f\n"
          % (npres, ctx_off, gsame, w, p, w, p))
    return t


def run(kinds, mods=None, miss=False):
    mods = mods or {}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E192"))
        for b in J.BRAINS:
            for c, p, w in J.CONFIGS:
                if miss and b == 12 and c == "C1":
                    continue
                kind = kinds[c] if isinstance(kinds[c], str) else kinds[c].get(b, kinds[c]["*"])
                open(os.path.join(td, "logs", "E192", "meas_b%d_%s.log" % (b, c)), "w", encoding="utf-8").write(
                    log_text(kind, 100 * b + int(c[1]), p, w, **mods.get((b, c), {})))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


cases = [
    ("C0 결합 → 표현 충분", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}), "판정: 표현 충분(H113-null)"),
    ("C0 덧셈·C2 결합 → 설정 의존 C2", dict(kinds={"C0": "additive", "C1": "additive", "C2": "conj"}), "판정: 설정 의존(H113) — C2"),
    ("C0 덧셈·C1 결합 → 설정 의존 C1", dict(kinds={"C0": "additive", "C1": "conj", "C2": "additive"}), "판정: 설정 의존(H113) — C1"),
    ("모두 덧셈 → 결합 부호 부족", dict(kinds={"C0": "additive", "C1": "additive", "C2": "additive"}), "판정: 결합 부호 부족(설정 무관)"),
    ("C0 결합 3/5 → 보류", dict(kinds={"C0": {"*": "conj", 13: "additive", 14: "additive"}, "C1": "additive", "C2": "additive"}), "판정: 보류"),
    ("범주 정보 없음(측정 무효) → 보류", dict(kinds={"C0": "nocat", "C1": "nocat", "C2": "nocat"}), "판정: 보류"),
    ("끈 맥락 발화", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}, mods={(10, "C0"): {"ctx_off": 3}}), "판정: 보류(조작검증 실패)"),
    ("제시 수 287", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}, mods={(11, "C1"): {"npres": 287}}), "판정: 보류(조작검증 실패)"),
    ("RCKX 결측 1줄", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}, mods={(12, "C2"): {"drop": 1}}), "판정: 보류(조작검증 실패)"),
    ("맥락 연결 수 p 와 안 맞음", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}, mods={(13, "C2"): {"conn_scale": 5.0}}), "판정: 보류(조작검증 실패)"),
    ("특징 연결 줄 없음", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}, mods={(14, "C0"): {"feat": False}}), "판정: 보류(조작검증 실패)"),
    ("측정 중 g 변함", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}, mods={(10, "C1"): {"gsame": 0}}), "판정: 보류(조작검증 실패)"),
    ("로그 결측", dict(kinds={"C0": "conj", "C1": "conj", "C2": "conj"}, miss=True), "판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "판정: 보류" or last == "판정: 보류")
    ok_all &= good
    print("%-30s → %-34s %s" % (name, last, "✓" if good else "✗\n" + out[-1500:]))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
