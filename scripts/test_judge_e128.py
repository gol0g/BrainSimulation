#!/usr/bin/env python3
"""judge_e128 합성 시험(조건 1) + 실제 줄 형식 파싱 + E127 로드.
1차 8/9: E127 과 같은 정규식 결함(both_spike_share(…) 괄호 안 공백을 \\S* 로 받음) — 수정 후 재시험. 재발: 출력 문자열 괄호 라벨은 \\(.*?\\) 로 받는다."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e128 as J

KL = [("learn", w, t) for w in J.WIRES for t in J.TSEEDS]
KF = [("frozen", w, 600) for w in J.WIRES]


def make(tl, ftl, sf=0.5, ksf=0.8, red_fail=0):
    X, K = {}, {}
    for i, k in enumerate(KL):
        X[k] = {"sf": sf, "bss": 0.5, "tl": tl[i], "nl": 50.0 + i}; K[k] = ksf
    for i, k in enumerate(KF):
        X[k] = {"sf": (ksf + 0.1 if i < red_fail else sf), "bss": 0.5, "tl": ftl[i], "nl": 40.0 + i}; K[k] = ksf
    return X, K


cases = []
def case(name, XK, want):
    c, r = J.judge(*XK)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-28s → %s" % ("PASS" if ok else "FAIL", name, v))

case("획득 16/16", make([95.0] * 16, [50.0] * 8), "획득 —")
case("획득 경계 12/16·frozen 2/8", make([80.0] * 12 + [79.0] * 4, [81.0] * 2 + [50.0] * 6), "획득 —")
case("불획득 4/16", make([90.0] * 4 + [50.0] * 12, [50.0] * 8), "불획득")
case("혼재 8/16", make([90.0] * 8 + [50.0] * 8, [50.0] * 8), "보류(혼재")
case("희석 감소 6/8 → 조작검증", make([95.0] * 16, [50.0] * 8, red_fail=2), "조작검증")
case("희석 감소 7/8 → 통과", make([95.0] * 16, [50.0] * 8, red_fail=1), "획득 —")
X, K = make([95.0] * 16, [50.0] * 8); del X[("learn", 13, 601)]
case("결측", (X, K), "결측")
ln = ("  xh learn w18 t600: => SDCREDIT mode=learn seed=18 trialseed=600 diff=cyclic | L전용 n=102 dSg=+0.5 | R전용 n=105 dSg=-0.4 | 양쪽 n=148 dSg=+0.1 | margin_sign_ok=5/8 | spec_share(|Σg_L−Σg_R| 중 전용 KC 몫) 평균 0.300 || "
      "=> SDRATE mode=learn seed=18 trialseed=600 | kc_spikes_per_stim 평균 73.9 | active_kc_per_stim 평균 73.9 | both_spike_share(양쪽 KC 스파이크/전체) 0.650 | rate_margin_ok=5/8 || "
      "=> SDLAB diff=cyclic rule=samediff mode=learn seed=18 trialseed=600 train_accL=90.0 train_accR=80.0 train_lbal=85.0 novel_accL=50.0 novel_accR=50.0 novel_lbal=50.0")
m = J.T28.match(ln)
ok = m is not None and abs(J.sfrac(m.group(4)) - 148 / 355) < 1e-12 and m.group(5) == "0.650" and m.group(6) == "85.0"
cases.append(ok); print("%s 실제 줄 형식 파싱" % ("PASS" if ok else "FAIL"))
X, K = J.load()
ok = len(K) == 24 and abs(K[("frozen", 10, 600)] - 0) >= 0
cases.append(ok and all(0 < v < 1 for v in K.values())); print("%s E127 K50 shared_frac 24런 로드(frozen w10 %.3f)" % ("PASS" if cases[-1] else "FAIL", K.get(("frozen", 10, 600), float("nan"))))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
