#!/usr/bin/env python3
"""posthoc_e207_core 합성 시험 — 정답을 아는 집합으로 공통 핵·핵 뺀 J·학습 KC 핵 몫, 그리고 RCKC 로그 → 집합(form_sets) 경로.
실행: python3 scripts/test_posthoc_e207_core.py"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_e207_core as C  # noqa: E402
import posthoc_kc_rewrite as PK  # noqa: E402


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %r (기대 %r)" % (name, got, want))

    core = set(range(0, 40))
    A, B = core | set(range(100, 120)), core | set(range(200, 220))
    Cc, D = core | set(range(100, 110)) | set(range(300, 310)), core | set(range(400, 420))
    st = C.core_stats({"A": A, "B": B, "C": Cc, "D": D})
    chk("공통 핵 크기", len(st["core"]), 40)
    chk("J(A,C) 핵 포함", st["J_AC"], 50 / 70)
    chk("J(A,C) 핵 뺌(범주 특이 10/30)", st["Jx_AC"], 10 / 30)
    chk("J(A,B) 핵 포함", st["J_AB"], 40 / 80)
    chk("J(A,B) 핵 뺌 = 0", st["Jx_AB"], 0.0)
    chk("J(B,D) 핵 뺌 = 0", st["Jx_BD"], 0.0)
    st0 = C.core_stats({"A": {1, 2}, "B": {3}, "C": {1}, "D": set()})
    chk("D 비면 핵 0", len(st0["core"]), 0)
    chk("빈 집합 J = 0", C.jac(set(), set()), 0.0)
    n = 500
    dR = np.zeros(n); dP = np.zeros(n); both = np.ones(n, dtype=bool)
    dR[[5, 10, 150]] = 80; dP[[5, 10, 150]] = 80; dP[[10, 150, 250]] = [-30, 0, 60]
    both[250] = True
    t, tc, c, cc = C.learn_in_core(dR, dP, both, core, 50.0)
    chk("R 맞춤 수(|d_R| ≥ 50)", t, 3)
    chk("R 맞춤 중 핵 안(5·10)", tc, 2)
    chk("P 바뀜 수(|d_P − d_R| ≥ 50 — 10·150·250)", c, 3)
    chk("P 바뀜 중 핵 안(10)", cc, 1)
    # RCKC 로그 → 집합: 노출 사례 4개 중 절반(2) 이상
    d = tempfile.mkdtemp(prefix="tpe207_")
    p = os.path.join(d, "m.log")
    with open(p, "w", encoding="utf-8") as f:
        rows = {"A": [{1, 2, 3}, {1, 2}, {1, 9}, {7}], "B": [{1}, {1, 4}, {4}, {5}], "C": [{1, 2}, {2}, set(), {8}], "D": [{1}, {1}, {6}, {6}]}
        for tag in ("pre", "post"):
            for c_, lst in rows.items():
                for i, ids in enumerate(lst):
                    f.write("RCKC tag=%s side=l set=train cat=%s i=%d n=%d ids=%s\n" % (tag, c_, i, len(ids), ",".join(str(x) for x in sorted(ids))))
                f.write("RCKC tag=%s side=l set=new cat=%s i=0 n=1 ids=999\n" % (tag, c_))
    sets = PK.form_sets(p, "pre")["l"]
    chk("집합 A(2회 이상)", sets["A"], {1, 2})
    chk("집합 B", sets["B"], {1, 4})
    chk("집합 C", sets["C"], {2})
    chk("집합 D", sets["D"], {1, 6})
    chk("새 사례(999) 제외", 999 in sets["A"], False)
    st2 = C.core_stats(sets)
    chk("로그 경로 공통 핵(A∩B∩C∩D = ∅)", st2["core"], set())
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
