#!/usr/bin/env python3
"""verify_e171_independent.py 합성 시험: KC 보존·감소·보류, 경계(0.85·1.15), 조작검증 실패(n·회귀·적재), 결측.
실행: python3 scripts/test_verify_e171.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e171_independent as V


def run(kr=1.0, per=None, n249=False, regress=False, pushed7=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E171"), ("logs", "E170")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            f = (per or {}).get(b, kr)
            for wn in ("AB", "none"):
                for v in V.VARS:
                    if miss and b == 14 and wn == "none" and v == "agree":
                        continue
                    k = 0.006 * f if v == "agree" else 0.006
                    mod = -0.5 if wn == "AB" else 0.01
                    n = 249 if (n249 and b == 11 and wn == "AB" and v == "base") else 250
                    pu = 7 if (pushed7 and wn == "AB" and b == 12 and v == "bad") else (8 if wn == "AB" else 0)
                    body = "[E146 변형] variant=%s vseed=0\n=> DECOMP mode=x mod=%+.4f acc=50.0 off=0.0 pushed=%d kc_means[x]\n" % (v, mod, pu)
                    for sd in ("left", "right"):
                        body += "[E171 평가 진단] variant=%s side=%s n=%d motor L/R 0.020000/0.020000 KC L/R %.6f/%.6f\n" % (v, sd, n, k, k)
                    w(("logs", "E171", "ev_b%d_%s_%s.log" % (b, wn, v)), body)
                    w(("logs", "E170", "ev_b%d_%s_%s.log" % (b, wn, v)), "=> DECOMP mode=x mod=%+.4f acc=50.0 off=0.0 pushed=%d kc_means[x]\n"
                      % (mod + (0.0002 if (regress and b == 13 and wn == "none" and v == "bad") else 0.0), pu))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("KC 보존", {}, "KC 보존(H094)"), ("KC 감소 0.6", {"kr": 0.6}, "KC 감소(H094-alt)"),
                       ("0.85 정확 → 보존", {"kr": 0.85}, "KC 보존(H094)"), ("1.16 → 보류", {"kr": 1.16}, "보류"),
                       ("3 보존·2 감소 → 보류", {"per": {12: 0.6, 13: 0.6}}, "보류"),
                       ("n 249", {"n249": True}, "보류(조작검증 실패)"), ("회귀 어긋남", {"regress": True}, "보류(조작검증 실패)"),
                       ("적재 7", {"pushed7": True}, "보류(조작검증 실패)"), ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want) and (want != "보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-22s → %-36s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
