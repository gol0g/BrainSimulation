#!/usr/bin/env python3
"""ctx_gate.rewire_within(E209 구획 되먹임 억제 재배선 판) 합성 시험 — 실행: python3 scripts/test_ctx_gate_rewire.py
 — 손으로 아는 작은 연결과 실제 크기(3000 × 400, p 0.05·0.08) 무작위 연결로:
in-degree 그대로·무리 밖 0·같은 무리 안 연결은 그대로·중복 없음·재현(같은 시드 같은 결과)·옮긴 수 = 원래 무리 밖 수·경계(후보 부족 → ValueError)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "genesis"))
from ctx_gate import rewire_within  # noqa: E402 — 모델이 쓰는 함수 그대로


def main():
    ok = tot = 0

    def chk(name, cond):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(("OK   " if cond else "실패 ") + name)

    # 작은 손 예: 앞 4개(0,1 = X · 2,3 = Y), 뒤 3개(0 = X · 1,2 = Y)
    gpre = np.array([True, True, False, False]); gpost = np.array([True, False, False])
    pre = np.array([0, 2, 3, 1, 2, 0]); post = np.array([0, 0, 1, 1, 2, 2])
    # 뒤 0(X): 앞 0(X 유지)·2(Y → X 후보 {1}) → {0,1}; 뒤 1(Y): 3(유지)·1(X → Y 후보 {2}) → {2,3}; 뒤 2(Y): 2(유지)·0(X → 후보 {3}) → {2,3}
    np_, nq, mv = rewire_within(pre, post, gpre, gpost, np.random.RandomState(1))
    got = sorted(zip(nq.tolist(), np_.tolist()))
    chk("손 예: 결과 쌍 = {(0,0),(0,1),(1,2),(1,3),(2,2),(2,3)}", got == [(0, 0), (0, 1), (1, 2), (1, 3), (2, 2), (2, 3)])
    chk("손 예: 옮긴 수 3", mv == 3)
    # 실제 크기 무작위
    rs0 = np.random.RandomState(7)
    for nm, npre, npost, p in (("KC→억제", 3000, 400, 0.05), ("억제→KC", 400, 3000, 0.08)):
        m = rs0.random_sample((npre, npost)) < p
        pre, post = np.nonzero(m)
        gpre = rs0.random_sample(npre) < 0.5
        gpost = np.zeros(npost, bool); gpost[rs0.permutation(npost)[:npost // 2]] = True
        if nm == "억제→KC":
            gpre = np.zeros(npre, bool); gpre[rs0.permutation(npre)[:npre // 2]] = True
            gpost = rs0.random_sample(npost) < 0.5
        a, b_, mv = rewire_within(pre, post, gpre, gpost, np.random.RandomState(11))
        a2, b2, mv2 = rewire_within(pre, post, gpre, gpost, np.random.RandomState(11))
        a3, _, _ = rewire_within(pre, post, gpre, gpost, np.random.RandomState(12))
        chk("%s: in-degree 그대로(뒤 뉴런마다)" % nm, np.array_equal(np.bincount(b_, minlength=npost), np.bincount(post, minlength=npost)))
        chk("%s: 무리 밖 0" % nm, int(np.sum(gpre[a] != gpost[b_])) == 0)
        chk("%s: 총 연결 수 그대로" % nm, a.size == pre.size)
        orig_in = set(zip(pre[gpre[pre] == gpost[post]].tolist(), post[gpre[pre] == gpost[post]].tolist()))
        chk("%s: 원래 같은 무리 안 연결은 모두 남음" % nm, orig_in <= set(zip(a.tolist(), b_.tolist())))
        chk("%s: 중복 쌍 없음" % nm, len(set(zip(a.tolist(), b_.tolist()))) == a.size)
        chk("%s: 옮긴 수 = 원래 무리 밖 수(%d)" % (nm, mv), mv == int(np.sum(gpre[pre] != gpost[post])))
        chk("%s: 재현(같은 시드 같음·다른 시드 다름)" % nm, np.array_equal(a, a2) and np.array_equal(b_, b2) and mv == mv2 and not np.array_equal(a, a3))
    # 후보 부족 → ValueError
    try:
        rewire_within(np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([True, False, False]), np.array([True]), np.random.RandomState(0))
        chk("후보 부족 → ValueError", False)
    except ValueError:
        chk("후보 부족 → ValueError", True)
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
