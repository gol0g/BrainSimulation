"""E165: 망 안 형성(kcdevoja) 노출 일정 — 제시마다 (종류, 쪽, 쪽 약칭, 강도).

기본(cyclic, bad_mult 1, 강도 0.9 고정)은 E160·E161 의 고정 순환(good-L, bad-L, good-R, bad-R 반복)과 같다.
random: 종류·쪽별 제시 수(good 은 쪽마다 n, bad 는 쪽마다 n × bad_mult)를 먼저 맞추고 지역 난수로 섞는다(전역 난수 소비 없음).
강도: lo == hi 면 고정, 아니면 섞은 뒤 같은 지역 난수로 제시마다 U[lo, hi].
합성 정답 시험: scripts/test_exposure_schedule.py
"""
import numpy as np

CYC = (("good", "left", "l"), ("bad", "left", "l"), ("good", "right", "r"), ("bad", "right", "r"))
KEYS = ("good_l", "bad_l", "good_r", "bad_r")


def build(n, order="cyclic", bad_mult=1, int_lo=0.9, int_hi=0.9, seed=0):
    if order not in ("cyclic", "random"):
        raise ValueError("order 는 cyclic·random")
    if n <= 0 or bad_mult < 1:
        raise ValueError("n > 0, bad_mult >= 1")
    if not (0.0 <= int_lo <= int_hi <= 1.0):
        raise ValueError("0 <= int_lo <= int_hi <= 1")
    rs = np.random.RandomState(seed)
    if order == "cyclic":
        if bad_mult != 1:
            raise ValueError("cyclic 은 bad_mult 1 만(고정 순환)")
        seq = [CYC[i % 4] for i in range(4 * n)]
    else:
        base = []
        for c in CYC:
            base += [c] * (n * (bad_mult if c[0] == "bad" else 1))
        seq = [base[i] for i in rs.permutation(len(base))]
    if int_lo == int_hi:
        ints = [float(int_lo)] * len(seq)
    else:
        ints = [float(x) for x in rs.uniform(int_lo, int_hi, len(seq))]
    return [(ty, side, sd, it) for (ty, side, sd), it in zip(seq, ints)]


def summary(sched):
    """종류·쪽별 제시 수, 강도 최소·최대·평균, 순환 일치율(다음 제시가 고정 순환의 다음 칸인 비율 — 고정 순환 1.0, 균등 무작위 약 0.25)."""
    idx = {(c[0], c[2]): i for i, c in enumerate(CYC)}
    cnt = dict.fromkeys(KEYS, 0)
    for ty, _side, sd, _it in sched:
        cnt["%s_%s" % (ty, sd)] += 1
    ints = np.array([x[3] for x in sched], dtype=float)
    st = [idx[(x[0], x[2])] for x in sched]
    cm = float(np.mean([(st[k + 1] - st[k]) % 4 == 1 for k in range(len(st) - 1)])) if len(st) > 1 else float("nan")
    return dict(cnt, n=len(sched), int_min=float(ints.min()), int_max=float(ints.max()), int_mean=float(ints.mean()), cyc_match=cm)
