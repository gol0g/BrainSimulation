#!/usr/bin/env python3
"""verify_e180_independent.py 합성 시험(test_verify_e170 을 옮김 — 회귀 사례 자리에 동작점·진단·비포화): 합성 성공·없음·부분, 경계(ρ 0.80), 충돌 상쇄·비가산,
조작검증 실패(자극·동작점 줄 없음·값 다름·진단 한쪽·포화·적재·전제), 포화 경계 통과(하나 0.6599), 결측.
실행: python3 scripts/test_verify_e180.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e180_independent as V  # noqa: E402

STIM = {"agree": ("[E170 자극] variant=agree side=left good L/R 0.90/0.00 bad L/R 0.00/0.90 food L/R 0.90/0.90\n"
                  "[E170 자극] variant=agree side=right good L/R 0.00/0.90 bad L/R 0.90/0.00 food L/R 0.90/0.90\n"),
        "conflict": ("[E170 자극] variant=conflict side=left good L/R 0.90/0.00 bad L/R 0.90/0.00 food L/R 0.90/0.00\n"
                     "[E170 자극] variant=conflict side=right good L/R 0.00/0.90 bad L/R 0.00/0.90 food L/R 0.00/0.90\n")}
TL = "  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n"


def diag(r4, sides=("left", "right")):
    d = {"left": "[E171 평가 진단] variant=x side=left n=250 motor L/R %.6f/%.6f KC L/R 0.004603/0.001772\n" % (r4[0], r4[1]),
         "right": "[E171 평가 진단] variant=x side=right n=250 motor L/R %.6f/%.6f KC L/R 0.001635/0.004875\n" % (r4[2], r4[3])}
    return "".join(d[s] for s in sides)


def run(ea=-1.20, ec=0.0, eb=-0.60, ed=0.60, badstim=False, notonic=False, tonic_val=-4000.0, diag_one=False,
        rates_b10=(0.50, 0.46, 0.47, 0.52), pushed7=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E180"))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
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
                    tl = "" if (notonic and b == 12 and wn == "AB" and v == "base") else TL % (tonic_val if (b == 10 and wn == "none" and v == "bad") else -4000.0)
                    r4 = rates_b10 if (b == 10 and wn == "AB" and v == "agree") else (0.50, 0.46, 0.47, 0.52)
                    dg = diag(r4, ("left",) if (diag_one and b == 14 and wn == "AB" and v == "conflict") else ("left", "right"))
                    w(("logs", "E180", "ev_b%d_%s_%s.log" % (b, wn, v)), tl + st + "[E146 변형] variant=%s vseed=0\n" % v + dg
                      + "=> DECOMP mode=%s mod=%+.4f acc=50.0 off=-0.0077 pushed=%d kc_means[x]\n" % ("all" if wn == "AB" else "none", mod, pu))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("가산 합성 / 상쇄", {}, "합성 성공(H101) / 가산 상쇄"),
                       ("합성 없음(ea −0.60)", {"ea": -0.60}, "합성 없음(H101-null)"),
                       ("부분(ea −0.85)", {"ea": -0.85}, "부분"),
                       ("ρ 0.80 정확(−0.96)", {"ea": -0.96}, "합성 성공(H101)"),
                       ("충돌 비가산(+0.30)", {"ec": 0.30}, "합성 성공(H101) / 비가산"),
                       ("자극 구성 어긋남", {"badstim": True}, "보류(조작검증 실패)"),
                       ("동작점 줄 없음", {"notonic": True}, "보류(조작검증 실패)"),
                       ("동작점 값 −1000", {"tonic_val": -1000.0}, "보류(조작검증 실패)"),
                       ("진단 한쪽", {"diag_one": True}, "보류(조작검증 실패)"),
                       ("포화(4개 ≥ 0.66)", {"rates_b10": (0.6664, 0.6662, 0.6663, 0.6600)}, "보류(조작검증 실패)"),
                       ("포화 하나 0.6599 → 통과", {"rates_b10": (0.6664, 0.6662, 0.6663, 0.6599)}, "합성 성공(H101)"),
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
