#!/usr/bin/env python3
"""posthoc_e208_graded 합성 시험 — 정답을 손으로 계산한 반응 비율·가짜 RCKC/RCJ 로그로 집합 J·가중 J·코사인·자기 점검이 맞는지.
실행: python3 test_posthoc_e208_graded.py (scripts/ 에서)"""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_e208_graded as P  # noqa: E402


def write_log(path, cases, rcj, bad_n=False, drop=None):
    """cases[(side, cat)] = [set × 6], rcj[(side, pair)] = J_post 실수."""
    with open(path, "w", encoding="utf-8") as f:
        for (s, c), lst in sorted(cases.items()):
            for i, ids in enumerate(lst):
                if drop == (s, c) and i == 5:
                    continue
                n = len(ids) + (1 if bad_n and (s, c, i) == ("l", "A", 0) else 0)
                f.write("RCKC tag=post side=%s set=train cat=%s i=%d n=%d ids=%s\n" % (s, c, i, n, ",".join(str(k) for k in sorted(ids))))
                f.write("RCKC tag=post side=%s set=new cat=%s i=%d n=1 ids=999\n" % (s, c, i))   # 새 사례 — 읽지 않아야
                f.write("RCKC tag=pre side=%s set=train cat=%s i=%d n=1 ids=998\n" % (s, c, i))  # 첫 측정 — 읽지 않아야
        for (s, pr), j in rcj.items():
            f.write("RCJ side=%s pair=%s J_pre=0.0000 J_post=%.4f n_pre=1/1 n_post=1/1\n" % (s, pr, j))


def main():
    ok = tot = 0

    def chk(name, cond):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(("OK   " if cond else "실패 ") + name)

    # metrics 손계산
    px = {1: 1.0, 2: 0.5, 3: 1 / 6}; py = {1: 1.0, 3: 0.5, 4: 1 / 3}
    js, wj, cos = P.metrics(px, py)
    chk("집합 J(p ≥ 0.5) = 1/3", abs(js - 1 / 3) < 1e-12)
    chk("가중 J = (7/6)/(7/3) = 0.5", abs(wj - 0.5) < 1e-12)
    exp_cos = (1 + 1 / 12) / (math.sqrt(1 + 0.25 + 1 / 36) * math.sqrt(1 + 0.25 + 1 / 9))
    chk("코사인 = 0.82146", abs(cos - exp_cos) < 1e-12 and abs(cos - 0.82146) < 1e-5)
    chk("빈 벡터 → 0", P.metrics({}, {}) == (0.0, 0.0, 0.0))
    # pvec — 3/6 = 0.5 는 집합 안
    pv = P.pvec([{1, 2}, {1}, {1, 3}, {2}, set(), {2}])
    chk("pvec 비율", pv == {1: 0.5, 2: 0.5, 3: 1 / 6})
    chk("경계 3/6 은 집합 안", P.metrics(pv, pv)[0] == 1.0)
    # 가짜 로그 — 쪽·범주마다 6사례
    cases = {}
    for s in "lr":
        cases[(s, "A")] = [{1, 2, 3}, {1, 2}, {1, 2}, {1}, {4}, {1}]   # p: 1=5/6, 2=3/6, 3=1/6, 4=1/6
        cases[(s, "B")] = [{10}, {10}, {10}, {11}, {11}, {11}]           # p: 10=0.5, 11=0.5
        cases[(s, "C")] = [{1, 3}, {1, 3}, {1, 3}, {5}, {5}, {1}]        # p: 1=4/6, 3=3/6, 5=2/6
        cases[(s, "D")] = [{10, 12}, {12}, {12}, {12}, {10}, {10}]       # p: 10=3/6, 12=4/6
    pa = P.pvec(cases[("l", "A")]); pc = P.pvec(cases[("l", "C")]); pb = P.pvec(cases[("l", "B")]); pd = P.pvec(cases[("l", "D")])
    jac = P.metrics(pa, pc)[0]; jbd = P.metrics(pb, pd)[0]; jab = P.metrics(pa, pb)[0]
    chk("가짜 자료 집합 J — AC {1,2}∩{1,3} = 1/3 · BD {10,11}∩{10,12} = 1/3 · AB 0", (round(jac, 6), round(jbd, 6), jab) == (round(1 / 3, 6), round(1 / 3, 6), 0.0))
    rcj = {(s, pr): v for s in "lr" for pr, v in (("AC", jac), ("BD", jbd), ("AB", jab))}
    d = tempfile.mkdtemp(prefix="pg208_")
    pth = os.path.join(d, "m.log")
    write_log(pth, cases, rcj)
    st = P.brain_stats(pth)
    chk("자기 점검 통과(집합 J = RCJ J_post)", st["self_check"] is True)
    chk("S 집합 = 4 × 1/3 → 13333", st["S_set"] == int(round(4 * (1 / 3) * 1e4)))
    wac = P.metrics(pa, pc)[1]; wbd = P.metrics(pb, pd)[1]
    chk("S 가중 = 두 쪽 (가중 AC + 가중 BD)", st["S_w"] == int(round(2 * (wac + wbd) * 1e4)))
    chk("새 사례·첫 측정 줄은 안 읽음(id 998·999 없음)", 998 not in pa and 999 not in pa)
    rcj_bad = dict(rcj); rcj_bad[("r", "BD")] = 0.5
    write_log(pth, cases, rcj_bad)
    chk("RCJ 다르면 자기 점검 실패", P.brain_stats(pth)["self_check"] is False)

    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False
    write_log(pth, cases, rcj, bad_n=True)
    chk("n 불일치 → ValueError", raises(lambda: P.brain_stats(pth)))
    write_log(pth, cases, rcj, drop=("r", "C"))
    chk("사례 5개뿐 → ValueError", raises(lambda: P.brain_stats(pth)))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
