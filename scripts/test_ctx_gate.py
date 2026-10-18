"""ctx_gate 합성 시험(E204) — 배정·마스크·발화 세기. 사용: python3 scripts/test_ctx_gate.py"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "genesis"))
import ctx_gate as CG  # noqa: E402

ok = 0
tot = 0


def chk(name, cond):
    global ok, tot
    tot += 1
    ok += bool(cond)
    print(("OK   " if cond else "실패 ") + name)


# 1~3 재현성·쪽 순서
A = CG.build_all(19300 + 12355, 1000, 200, 200, 0.5, 0.10)
B = CG.build_all(19300 + 12355, 1000, 200, 200, 0.5, 0.10)
chk("같은 시드 → 같은 배정·마스크", all(np.array_equal(A[s][i], B[s][i]) for s in "lr" for i in range(3)))
C = CG.build_all(19300 + 12356, 1000, 200, 200, 0.5, 0.10)
chk("다른 시드 → 다른 배정", not np.array_equal(A["l"][0], C["l"][0]))
rs = np.random.RandomState(19300 + 12355)
l_ = CG.side_masks(rs, 1000, 200, 200, 0.5, 0.10)
r_ = CG.side_masks(rs, 1000, 200, 200, 0.5, 0.10)
chk("쪽 순서 l → r (build_all 과 같음)", np.array_equal(l_[0], A["l"][0]) and np.array_equal(r_[0], A["r"][0]) and np.array_equal(r_[2], A["r"][2]))
# 4~5 배정 비율
fx = A["l"][0].mean()
chk("X 무리 비율 0.5 근처(이항 ±4σ, n 1000): %.3f" % fx, abs(fx - 0.5) < 4 * np.sqrt(0.25 / 1000))
A3 = CG.build_all(7, 2000, 50, 60, 0.3, 0.2)
chk("비율 0.3 → %.3f" % A3["r"][0].mean(), abs(A3["r"][0].mean() - 0.3) < 4 * np.sqrt(0.21 / 2000))
# 6~9 마스크는 의도한 무리만, 밀도 ≈ p
isx, mx, my = A["l"]
chk("맥락 X 집단 → X 무리 연결 0", int(mx[:, isx].sum()) == 0)
chk("맥락 Y 집단 → Y 무리 연결 0", int(my[:, ~isx].sum()) == 0)
dx = mx[:, ~isx].mean(); dy = my[:, isx].mean()
chk("X 집단 → Y 무리 밀도 ≈ 0.10: %.4f" % dx, abs(dx - 0.10) < 0.01)
chk("Y 집단 → X 무리 밀도 ≈ 0.10: %.4f" % dy, abs(dy - 0.10) < 0.01)
chk("마스크 모양 (n_x, n_kc)·(n_y, n_kc)", A3["l"][1].shape == (50, 2000) and A3["l"][2].shape == (60, 2000))
# 10~13 발화 세기
isx2 = np.array([True, False, True, False, False])
chk("발화 세기 (X, Y)", CG.count_by_subset([0, 2, 2, 1, 4], isx2) == (3, 2))
chk("빈 발화 → (0, 0)", CG.count_by_subset([], isx2) == (0, 0))
try:
    CG.count_by_subset([5], isx2); bad = False
except ValueError:
    bad = True
chk("범위 밖 id 거부", bad)
chk("활성·비활성 X 맥락", CG.active_inactive((3, 1), (2, 5), "X") == (5, 6))
chk("활성·비활성 Y 맥락", CG.active_inactive((3, 1), (2, 5), "Y") == (6, 5))
# 15~18 인자 거부
for nm, args_ in (("w ≥ 0 거부", (0.0, 0.1, 0.5)), ("p = 0 거부", (-20.0, 0.0, 0.5)), ("frac = 1 거부", (-20.0, 0.1, 1.0)), ("frac = 0 거부", (-20.0, 0.1, 0.0))):
    try:
        CG.check_params(*args_); bad = False
    except ValueError:
        bad = True
    chk(nm, bad)
try:
    CG.check_params(-20.0, 0.1, 0.5); good = True
except ValueError:
    good = False
chk("정상 인자 통과", good)
print("합성 %d/%d" % (ok, tot))
sys.exit(0 if ok == tot else 1)
