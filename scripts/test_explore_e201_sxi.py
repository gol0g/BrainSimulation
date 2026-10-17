"""explore_e201_sxi.split·summarize 합성 시험 — Δ = gain·sign(SXI) + offset(구간별)을 알고 만든 KC 로 구간·부호 평균과 offset·gain 복원을 검사."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore_e201_sxi as X  # noqa: E402

ok = bad = 0


def chk(name, got, want, tol=1e-9):
    global ok, bad
    if (np.isnan(want) and np.isnan(got)) or abs(got - want) <= tol:
        ok += 1
    else:
        bad += 1
        print("실패", name, got, want)


# 쪽마다 KC 20개: |SXI| 구간 0~4 에 4개씩(교차 2·같은 쪽 2), 그중 하나는 비반응(제외돼야)
sx = np.array([0.05, -0.05, 0.07, -0.07, 0.15, -0.15, 0.12, -0.12, 0.25, -0.25, 0.22, -0.22,
               0.4, -0.4, 0.35, -0.35, 0.8, -0.8, 0.6, -0.6])
resp = np.ones(20, dtype=np.int8)
resp[2] = 0                                  # 구간 0 교차 KC 하나 비반응
OFF = [-10.0, -6.0, -3.0, -1.0, 0.0]
GAIN = [2.0, 8.0, 20.0, 40.0, 80.0]
d = np.zeros(20)
for k in range(20):
    i = [j for j, (lo, hi) in enumerate(X.BINS) if lo <= abs(sx[k]) < hi][0]
    d[k] = GAIN[i] * np.sign(sx[k]) + OFF[i]
d[2] = 999.0                                 # 비반응 KC 값은 섞이면 안 됨
Z = {"resp_l": resp, "resp_r": resp, "sxi_l": sx, "sxi_r": sx}
dl = {"l": d, "r": d}
rows = X.summarize(X.split(dl, Z))
for i in range(5):
    chk("구간%d offset" % i, rows[i]["offset"], OFF[i])
    chk("구간%d gain" % i, rows[i]["gain"], GAIN[i])
    chk("구간%d 교차 평균" % i, rows[i]["mean_c"], GAIN[i] + OFF[i])
chk("구간0 교차 수(비반응 제외·두 쪽)", rows[0]["n_c"], 2)
chk("구간0 같은 쪽 수", rows[0]["n_s"], 4)
chk("구간4 교차 수", rows[4]["n_c"], 4)
# SXI 0 인 KC 는 어느 부호에도 안 들어감
Z0 = {"resp_l": np.ones(2, np.int8), "resp_r": np.ones(2, np.int8), "sxi_l": np.array([0.0, 0.3]), "sxi_r": np.array([0.0, -0.3])}
r0 = X.summarize(X.split({"l": np.array([5.0, 7.0]), "r": np.array([5.0, -7.0])}, Z0))
chk("SXI 0 제외 — 구간0 교차 수", r0[0]["n_c"], 0)
chk("구간3 교차 평균", r0[3]["mean_c"], 7.0)
chk("구간3 같은 쪽 평균", r0[3]["mean_s"], -7.0)
chk("빈 구간 nan", r0[1]["mean_c"], float("nan"))
print("합성 %d/%d" % (ok, ok + bad))
sys.exit(0 if bad == 0 else 1)
