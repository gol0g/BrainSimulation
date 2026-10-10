#!/usr/bin/env python3
"""verify_e170_independent.py 합성 시험: 합성 성공·없음·부분, 경계(ρ 0.80), 충돌 상쇄·비가산, 조작검증 실패(자극·회귀·적재·전제), 결측.
실행: python3 scripts/test_verify_e170.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e170_independent as V

STIM = {"agree": ("[E170 자극] variant=agree side=left good L/R 0.90/0.00 bad L/R 0.00/0.90 food L/R 0.90/0.90\n"
                  "[E170 자극] variant=agree side=right good L/R 0.00/0.90 bad L/R 0.90/0.00 food L/R 0.90/0.90\n"),
        "conflict": ("[E170 자극] variant=conflict side=left good L/R 0.90/0.00 bad L/R 0.90/0.00 food L/R 0.90/0.00\n"
                     "[E170 자극] variant=conflict side=right good L/R 0.00/0.90 bad L/R 0.00/0.90 food L/R 0.00/0.90\n")}


def run(ea=-1.20, ec=0.0, eb=-0.60, ed=0.60, badstim=False, regress=0.0, pushed7=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E170"))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        e162 = []
        for b in V.BRAINS:
            vals = {"base": eb, "bad": ed, "agree": ea, "conflict": ec}
            for wn in ("AB", "none"):
                for v in V.VARS:
                    if miss and b == 14 and wn == "none" and v == "conflict":
                        continue
                    mod = 0.01 + (vals[v] if wn == "AB" else 0.0)
                    st = STIM.get(v, "")
                    if badstim and b == 11 and v == "agree":
                        st = st.replace("bad L/R 0.90/0.00", "bad L/R 0.00/0.00")
                    pu = 7 if (pushed7 and wn == "AB" and b == 13 and v == "bad") else (8 if wn == "AB" else 0)
                    w(("logs", "E170", "ev_b%d_%s_%s.log" % (b, wn, v)), st + "[E146 변형] variant=%s vseed=0\n=> DECOMP mode=%s mod=%+.4f acc=50.0 off=-0.0077 pushed=%d kc_means[x]\n"
                      % (v, "all" if wn == "AB" else "none", mod, pu))
                    if v in ("base", "bad"):
                        e162.append("  e162 b%d %s %s: => mod %+.4f" % (b, wn, v, mod + (regress if (b == 12 and wn == "AB" and v == "base") else 0.0)))
        w(("E162.log",), "\n".join(e162) + "\n")
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("가산 합성 / 상쇄", {}, "합성 성공(H093) / 가산 상쇄"),
                       ("합성 없음(ea −0.60)", {"ea": -0.60}, "합성 없음(H093-null)"),
                       ("부분(ea −0.85)", {"ea": -0.85}, "부분"),
                       ("ρ 0.80 정확(−0.96)", {"ea": -0.96}, "합성 성공(H093)"),
                       ("충돌 비가산(+0.30)", {"ec": 0.30}, "합성 성공(H093) / 비가산"),
                       ("자극 구성 어긋남", {"badstim": True}, "보류(조작검증 실패)"),
                       ("회귀 어긋남 0.0002", {"regress": 0.0002}, "보류(조작검증 실패)"),
                       ("적재 pushed 7", {"pushed7": True}, "보류(조작검증 실패)"),
                       ("전제 bad +0.0999", {"ed": 0.0999}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want)
    ok_all &= good
    print("%-22s → %-40s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
