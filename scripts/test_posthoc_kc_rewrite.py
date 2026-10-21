#!/usr/bin/env python3
"""posthoc_kc_rewrite 합성 시험 — 정답을 아는 가짜 스냅숏·가중치로 KC 선호·다시 씀 몫이 맞게 나오는지.
경우: (1) P 가 R 맞춤 KC 를 그대로 뒤집음 → 다시 씀 1·뒤집힘 1·중앙 < 0·P 바뀐 KC 중 R 맞춤 1; (2) P 가 겹치지 않는 KC 만 맞춤 → 다시 씀 0·P 바뀐 KC 중 R 맞춤 0·중앙 1;
(3) P 가 R 맞춤 KC 를 절반만 지움(d_P = d_R/2, 차 = θ 보다 큼) → 다시 씀 1·뒤집힘 0·중앙 0.5; (4) 경계 θ 정확히 → 포함; (5) 한 motor 에만 시냅스가 있는 KC 는 제외.
실행: python3 scripts/test_posthoc_kc_rewrite.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_kc_rewrite as P  # noqa: E402

NK = 40
SYN = 4   # KC 마다 motor 쪽당 시냅스 수


def snap(missing_r=()):
    S = {}
    for sd in "lr":
        for m in "lr":
            pre = np.repeat(np.arange(NK), SYN)
            if m == "r" and missing_r:
                pre = pre[~np.isin(pre, list(missing_r))]
            S["kc_%s_to_motor_%s__pre" % (sd, m)] = pre.astype(np.uint32)
    return S


def weights(S, d):
    """KC 별 선호 d(길이 NK) → w_l = 150 + d/2, w_r = 150 − d/2 (합 300)."""
    W = {}
    for sd in "lr":
        for m in "lr":
            pre = S["kc_%s_to_motor_%s__pre" % (sd, m)].astype(int)
            W["kc_%s_to_motor_%s" % (sd, m)] = (150 + (0.5 if m == "l" else -0.5) * d[pre]).astype(np.float64)
    return W


def stats(dR, dP, theta=50.0, missing_r=()):
    S = snap(missing_r)
    R, Pw = weights(S, dR), weights(S, dP)
    out = {}
    for sd in "lr":
        a, bothA = P.kc_pref(R, S, sd, NK)
        b, bothB = P.kc_pref(Pw, S, sd, NK)
        out[sd] = P.rewrite_stats(a, b, bothA & bothB, theta)
    return P.merge(out)


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %s (기대 %s)" % (name, got, want))

    dR = np.zeros(NK); dR[:10] = 200.0; dR[10:20] = -200.0          # R 맞춤 20 KC(쪽마다)
    # (1) 그대로 뒤집음
    m = stats(dR, -dR)
    chk("(1) 뒤집음 — R 맞춤 수(두 쪽)", m["tuned"], 40)
    chk("(1) 다시 씀 몫", m["rewrite"], 1.0); chk("(1) 뒤집힘 몫", m["flip"], 1.0); chk("(1) 중앙", m["ratio_med"], -1.0); chk("(1) P 바뀐 KC 중 R 맞춤", m["changed_from_tuned"], 1.0)
    # (2) 겹치지 않는 KC 만
    dP = dR.copy(); dP[20:30] = 200.0; dP[30:40] = -200.0
    m = stats(dR, dP)
    chk("(2) 따로 — 다시 씀 몫", m["rewrite"], 0.0); chk("(2) P 바뀐 KC 수", m["changed"], 40); chk("(2) P 바뀐 KC 중 R 맞춤", m["changed_from_tuned"], 0.0); chk("(2) 중앙", m["ratio_med"], 1.0)
    # (3) 절반 지움
    m = stats(dR, dR / 2)
    chk("(3) 절반 — 다시 씀 몫", m["rewrite"], 1.0); chk("(3) 뒤집힘 몫", m["flip"], 0.0); chk("(3) 중앙", m["ratio_med"], 0.5)
    # (4) 경계: |d_R| = θ 정확히 → R 맞춤, |d_P − d_R| = θ 정확히 → 바뀜
    dRb = np.zeros(NK); dRb[0] = 50.0
    dPb = dRb.copy(); dPb[0] = 0.0
    m = stats(dRb, dPb)
    chk("(4) 경계 — R 맞춤", m["tuned"], 2); chk("(4) 경계 — 바뀜·겹침", (m["changed"], m["inter"]), (2, 2))
    dRc = np.zeros(NK); dRc[0] = 49.999
    m = stats(dRc, np.zeros(NK))
    chk("(4) 경계 아래 — R 맞춤 0", m["tuned"], 0)
    # (5) 한 motor 에만 시냅스가 있는 KC 제외(KC 0~4 는 motor_r 시냅스 없음)
    m = stats(dR, -dR, missing_r=range(5))
    chk("(5) 제외 — R 맞춤(쪽마다 20 − 5)", m["tuned"], 30)
    # (6) 형성 로그 집합 읽기 — 노출 6 중 3 이상(절반)인 KC, set=new 줄은 무시
    import tempfile
    fd, fp = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for i in range(6):
            ids = [1, 2] + ([3] if i < 3 else []) + ([4] if i < 2 else [])
            f.write("RCKC tag=pre side=l set=train cat=A i=%d n=%d ids=%s\n" % (i, len(ids), ",".join(map(str, ids))))
        f.write("RCKC tag=pre side=l set=new cat=A i=0 n=1 ids=99\n")
        f.write("RCKC tag=post side=l set=train cat=A i=0 n=1 ids=7\n")
    st = P.form_sets(fp, "pre"); os.remove(fp)
    chk("(6) 형성 집합 — 절반 이상(3/6) 포함·2/6 제외·new 무시", sorted(st["l"]["A"]), [1, 2, 3])
    # (7) 집합 관계 — R 맞춤 KC 0~9, P 바뀜 KC 5~14, A∪B = 0~7, C∪D = 4~11 → 겹침 4~7
    dR7 = np.zeros(NK); dR7[:10] = 200.0
    dP7 = dR7.copy(); dP7[5:15] = -200.0
    both7 = np.ones(NK, dtype=bool)
    s7 = P.set_stats(dR7, dP7, both7, {"A": set(range(0, 4)), "B": set(range(4, 8)), "C": set(range(4, 8)), "D": set(range(8, 12))}, 50.0)
    chk("(7) R 맞춤·A∪B 안", (s7["tuned"], s7["tuned_in_ab"]), (10, 8))
    chk("(7) 겹침 KC 수", s7["shared"], 4)
    chk("(7) 함께 바뀐 KC·그중 겹침·C∪D", (s7["inter"], s7["inter_in_shared"], s7["inter_in_cd"]), (5, 3, 5))
    chk("(7) P 바뀜·C∪D 안", (s7["changed"], s7["changed_in_cd"]), (10, 7))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
