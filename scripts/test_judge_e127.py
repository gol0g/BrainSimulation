#!/usr/bin/env python3
"""judge_e127 합성 시험(조건 1) + 실제 줄 형식 파싱 + E126 로드.
1차 8/10: 판정 정규식이 spec_share(…) 괄호 안 공백을 \\S* 로 받아 실제 줄 파싱 실패 — 판정 스크립트 정규식 결함, 수정 후 재시험."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e127 as J

KL = [("learn", w, t) for w in J.WIRES for t in J.TSEEDS]
KF = [("frozen", w, 600) for w in J.WIRES]


def rec(dL, dR, mok, tl=50.0, nl=50.0):
    return {"nL": 20, "dL": dL, "nR": 20, "dR": dR, "nB": 60, "dB": 0.0, "mok": mok, "mtot": 8, "share": 0.3, "tl": tl, "nl": nl}


def make(dL, dR, mok, reg_bad=()):
    R, E = {}, {}
    for i, k in enumerate(KL):
        R[k] = rec(dL[i], dR[i], mok[i], tl=50.0 + i, nl=40.0 + i); E[k] = (50.0 + i, 40.0 + i)
    for i, k in enumerate(KF):
        R[k] = rec(0.01, -0.01, 4, tl=30.0 + i, nl=20.0 + i); E[k] = (30.0 + i, 20.0 + i)
    for k in reg_bad:
        E[k] = (99.0, 99.0)
    return R, E


cases = []
def case(name, RE, want):
    c, r = J.judge(*RE)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-32s → %s" % ("PASS" if ok else "FAIL", name, v))

case("묻힘", make([0.5] * 16, [-0.5] * 16, [4] * 16), "묻힘")
case("묻힘 경계 12/16·margin 6", make([0.5] * 12 + [-0.1] * 4, [-0.5] * 16, [6] * 12 + [8] * 4), "묻힘")
case("신용 틀림 4/16", make([0.5] * 4 + [-0.2] * 12, [-0.5] * 16, [4] * 16), "신용 틀림")
case("신용 맞음·margin 좋음 → 보류", make([0.5] * 16, [-0.5] * 16, [8] * 16), "보류(혼재")
case("nan 부류는 맞음 아님", make([float("nan")] * 13 + [0.5] * 3, [-0.5] * 16, [4] * 16), "신용 틀림")
case("회귀 불일치", make([0.5] * 16, [-0.5] * 16, [4] * 16, reg_bad=(("learn", 13, 601),)), "조작검증")
R, E = make([0.5] * 16, [-0.5] * 16, [4] * 16); del R[("frozen", 12, 600)]
case("결측", (R, E), "결측")
ln = ("  cr learn w18 t600: => SDCREDIT mode=learn seed=18 trialseed=600 diff=cyclic | L전용 n=12 dSg=+0.0312 | R전용 n=9 dSg=-0.0150 | "
      "양쪽 n=80 dSg=+0.0010 | margin_sign_ok=5/8 | spec_share(|Σg_L−Σg_R| 중 전용 KC 몫) 평균 0.214 || "
      "=> SDLAB diff=cyclic rule=samediff mode=learn seed=18 trialseed=600 train_accL=84.0 train_accR=50.0 train_lbal=67.0 novel_accL=83.7 novel_accR=36.8 novel_lbal=60.3")
p = J.parse(ln)
ok = p is not None and p[0] == ("learn", 18, 600) and p[1]["dL"] == 0.0312 and p[1]["dR"] == -0.015 and p[1]["mok"] == 5 and p[1]["tl"] == 67.0 and p[1]["share"] == 0.214
cases.append(ok); print("%s 실제 줄 형식 파싱 → %s" % ("PASS" if ok else "FAIL", p[1] if p else None))
ln2 = ln.replace("dSg=+0.0312", "dSg=nan")
p2 = J.parse(ln2); ok = p2 is not None and p2[1]["dL"] != p2[1]["dL"]
cases.append(ok); print("%s nan dSg 파싱" % ("PASS" if ok else "FAIL"))
R, E = J.load()
ok = len(E) == 24 and E[("learn", 10, 600)] == (50.0, 53.0)
cases.append(ok); print("%s E126 24런 로드(w10 t600 = %s)" % ("PASS" if ok else "FAIL", E.get(("learn", 10, 600))))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
