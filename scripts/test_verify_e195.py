#!/usr/bin/env python3
"""verify_e195_independent.py 합성 시험 — test_judge_e195.py 와 같은 가짜 로그(복사)로 독립 대조 확인(판정 논리는 metrics 를 바꿔서).
실행: python3 scripts/test_verify_e195.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e195_independent as J  # noqa: E402
import rcxor  # noqa: E402

N_KC = 3000


def log_text(t, ctx_off=0, inhg=None, cfgw=None, line195=True, drop=0, p=0.02, w=20.0, kind="additive", seed=0):
    s = "[E153 종류 입력 적재] x 검증 일치 — y\n"
    if line195 and abs(t - 1.0) > 1e-9:
        s += "[E195 KC 되먹임 억제] 배율 %.2f (kc_inh_to_kc_weight %.1f)\n" % (t, -15.0 * t)
    n = int(200 * N_KC * p)
    s += "  [맥락 입력] 맥락 전용 집단 zz_ctx 200 → KC 좌·우, w=%.2f, p=%.3f, 연결 좌 %d·우 %d\n" % (w, p, n, n)
    s += "  [맥락 입력 2] 맥락 전용 집단 zz_ctx2 200 → KC 좌·우, w=%.2f, p=%.3f, 연결 좌 %d·우 %d\n" % (w, p, n, n)
    s += "  [원시 특징] 특징 집단 zz_feat 좌·우 각 200 → 같은 쪽 KC, w=24.00, p=0.100, 연결 좌 59816·우 60158, 정적\n"
    k = 0
    for sd in "lr":
        for (st, c, x, i, r, v) in rcxor._synthetic(kind, seed + (0 if sd == "l" else 50)):
            k += 1
            if k <= drop:
                continue
            nz = np.nonzero(v)[0]
            s += "RCKX rep=%d side=%s k=%d set=%s cat=%s ctx=%s i=%d n=%d ids=%s\n" % (r, sd, k, st, c, x, i, nz.size, ",".join("%d:%d" % (j, v[j]) for j in nz))
    gw = -15.0 * t
    s += ("RCKXCHK npres=288 ctx_on_spikes=50000 ctx_off_spikes=%d feat_inactive_spikes=0 base_l=1 base_r=1 g_same=1 ctx_w=%.2f ctx_p=%.3f ctx2_w=%.2f ctx2_p=%.3f kc_target=0.050 cfg_kc_target=0.050 kc_inh_scale=%.2f cfg_kc_inh_w=%.2f inh_g_l=%.3f inh_g_r=%.3f\n"
          % (ctx_off, w, p, w, p, t, gw if cfgw is None else cfgw, gw if inhg is None else inhg, gw))
    return s


def run(metrics=None, mods=None, miss=False, real=False):
    mods = mods or {}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E195"))
        for b in J.BRAINS:
            for lv, t in J.LEVELS:
                if miss and b == 12 and lv == "SR":
                    continue
                open(os.path.join(td, "logs", "E195", "meas_b%d_%s.log" % (b, lv)), "w", encoding="utf-8").write(log_text(t, **mods.get((b, lv), {})))
        J.EXP = td
        orig = J.metrics
        if not real:
            def fake(path, t, _m=metrics):
                b = int(os.path.basename(path).split("_b")[1].split("_")[0]); lv = os.path.basename(path).split("_")[2].split(".")[0]
                return dict(_m(b, lv))
            J.metrics = fake
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                J.main()
        finally:
            J.metrics = orig
    return buf.getvalue()


def M(p5=0.10, p2=0.16, n5=300.0, n2=200.0, x2=0.95, cat2=1.0, ctx2=1.0, per=None):
    per = per or {}

    def f(b, lv):
        q = dict(p5=p5, p2=p2, n5=n5, n2=n2, x2=x2, cat2=cat2, ctx2=ctx2)
        q.update(per.get(b, {}))
        if lv == "S1":
            return {"p": q["p5"], "n": q["n5"], "hi": 80, "xor": 0.99, "cat": 1.0, "ctx": 1.0}
        if lv == "SR":
            return {"p": (q["p5"] + q["p2"]) / 2, "n": (q["n5"] + q["n2"]) / 2, "hi": 90, "xor": 0.99, "cat": 1.0, "ctx": 1.0}
        return {"p": q["p2"], "n": q["n2"], "hi": 100, "xor": q["x2"], "cat": q["cat2"], "ctx": q["ctx2"]}
    return f


cases = [
    ("순도 상승", dict(metrics=M()), "독립 판정: 순도 상승(H116)"),
    ("경계 Δp 0.05", dict(metrics=M(p2=0.15)), "독립 판정: 순도 상승(H116)"),
    ("결합 소실", dict(metrics=M(x2=0.74)), "독립 판정: 결합 소실"),
    ("변화 없음", dict(metrics=M(p2=0.119)), "독립 판정: 변화 없음(H116-null)"),
    ("희소화 안 들음", dict(metrics=M(n2=273.0)), "독립 판정: 보류(조작 무효"),
    ("cfg 억제 가중치 불일치(실제 load)", dict(metrics=None, real=True, mods={(10, "SP"): {"cfgw": -15.0}}), "독립 판정: 보류(조작검증 실패)"),
    ("로그 결측(실제 metrics — 가짜는 파일을 안 열어 결측을 못 봄)", dict(metrics=None, real=True, miss=True), "독립 판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "독립 판정: 보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-26s → %-30s %s" % (name, last, "✓" if good else "✗\n" + out[-1200:]))
out = run(real=True)
last = out.strip().splitlines()[-1]
good = last.startswith("독립 판정: 보류(조작 무효")
ok_all &= good
print("%-26s → %-30s %s" % ("배관(실제 metrics)", last, "✓" if good else "✗\n" + out[-1200:]))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases) + 1))
sys.exit(0 if ok_all else 1)
