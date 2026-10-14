#!/usr/bin/env python3
"""judge_e195.py 합성 시험 — (1) 판정 논리: measure 를 정한 값으로 바꿔 경계·범주 (2) 조작검증: 가짜 로그 (3) measure 배관: 합성 RCKX 로그.
실행: python3 scripts/test_judge_e195.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e195 as J  # noqa: E402
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
        orig = J.measure
        if not real:
            def fake(path, _m=metrics):
                b = int(os.path.basename(path).split("_b")[1].split("_")[0]); lv = os.path.basename(path).split("_")[2].split(".")[0]
                return dict(_m(b, lv))
            J.measure = fake
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                J.main()
        finally:
            J.measure = orig
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
    ("순도 상승", dict(metrics=M()), "판정: 순도 상승(H116)"),
    ("경계 Δp 0.05 → 상승", dict(metrics=M(p2=0.15)), "판정: 순도 상승(H116)"),
    ("Δp 0.0499 → 보류", dict(metrics=M(p2=0.1499)), "판정: 보류"),
    ("결합 소실(XOR 0.74)", dict(metrics=M(x2=0.74)), "판정: 결합 소실"),
    ("변화 없음 |Δp| 0.019", dict(metrics=M(p2=0.119)), "판정: 변화 없음(H116-null)"),
    ("Δp −0.03 → 보류", dict(metrics=M(p2=0.07)), "판정: 보류"),
    ("희소화 안 들음(n 0.91배)", dict(metrics=M(n2=273.0)), "판정: 보류(조작 무효"),
    ("희소화 경계 n 0.9배 → 상승", dict(metrics=M(n2=270.0)), "판정: 순도 상승(H116)"),
    ("범주 해독 0.89(무효) 2 뇌 → 보류", dict(metrics=M(per={13: {"cat2": 0.89}, 14: {"cat2": 0.89}})), "판정: 보류"),
    ("장치 억제 g 불일치(배율 안 닿음)", dict(metrics=M(), mods={(10, "SP"): {"inhg": -15.0}}), "판정: 보류(조작검증 실패)"),
    ("E195 배율 줄 없음", dict(metrics=M(), mods={(11, "SR"): {"line195": False}}), "판정: 보류(조작검증 실패)"),
    ("끈 맥락 발화", dict(metrics=M(), mods={(12, "S1"): {"ctx_off": 2}}), "판정: 보류(조작검증 실패)"),
    ("RCKX 결측 1줄", dict(metrics=M(), mods={(13, "SP"): {"drop": 1}}), "판정: 보류(조작검증 실패)"),
    ("맥락 p 0.10(설정 틀림)", dict(metrics=M(), mods={(14, "S1"): {"p": 0.10, "w": 4.0}}), "판정: 보류(조작검증 실패)"),
    ("로그 결측", dict(metrics=M(), miss=True), "판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "판정: 보류" or last == "판정: 보류")
    ok_all &= good
    print("%-30s → %-32s %s" % (name, last, "✓" if good else "✗\n" + out[-1500:]))
# 배관: 실제 measure 로 합성 로그(덧셈 부호 — 수준 차 없음) → 희소화 안 들음으로 보류(조작 무효)
out = run(real=True)
last = out.strip().splitlines()[-1]
good = last.startswith("판정: 보류(조작 무효") and "순도 S1" in out
ok_all &= good
print("%-30s → %-32s %s" % ("배관(실제 measure·합성 로그)", last, "✓" if good else "✗\n" + out[-1500:]))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases) + 1))
sys.exit(0 if ok_all else 1)
