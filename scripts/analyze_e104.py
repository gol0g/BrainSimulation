#!/usr/bin/env python3
"""E104 런별 분석: 반전 시간 경과. 강도(옛 규칙 방향 연합): A = Σ_A전용 (g_l − g_r), B = Σ_B전용 (g_r − g_l). 양수 = 옛 쪽.
출력 한 줄 "E104SUM ...": 반전 시작 강도, 0 교차 시행(반전 후 몇 번째, 없으면 -1), 100시행 단위 강도 궤적,
초반(반전 후 0~400) 대 후반(마지막 400) 감소 속도, 반전 구간 greedy 옛 선택 비율(초반/후반)."""
import sys
import numpy as np

z = np.load(sys.argv[1])
gl = np.vstack([z["g0_l"][None, :], z["g_l"]]); gr = np.vstack([z["g0_r"][None, :], z["g_r"]])
stim, act, nl, nr = z["stim"], z["act"], z["nl"], z["nr"]
flip = int(z["flip_at"]); T = len(stim)
ka, kb = set(z["kc_a"].tolist()), set(z["kc_b"].tolist())
A = np.array(sorted(ka - kb), int); B = np.array(sorted(kb - ka), int)
sA = (gl[:, A] - gr[:, A]).sum(axis=1)   # 행 i = 시행 i-1 후 상태 (행 0 = 초기)
sB = (gr[:, B] - gl[:, B]).sum(axis=1)
o = {"flip": flip, "T": T}
for tag, s in (("A", sA), ("B", sB)):
    o["start_" + tag] = float(s[flip]); o["end_" + tag] = float(s[T])
    post = s[flip:]
    cr = np.where(post <= 0)[0]
    o["cross_" + tag] = int(cr[0]) if len(cr) else -1
    L = T - flip
    e0, e1 = s[flip], s[min(flip + 400, T)]
    l0, l1 = s[max(T - 400, flip)], s[T]
    o["rate_early_" + tag] = float((e0 - e1) / 4.0)   # 100시행당 감소
    o["rate_late_" + tag] = float((l0 - l1) / 4.0)
    o["traj_" + tag] = "/".join("%.0f" % s[min(flip + k, T)] for k in range(0, L + 1, 100))
# greedy 옛 선택: 반전 구간에서 출력 우세(nl vs nr)가 옛 규칙 쪽인 비율(탐색 여부와 무관한 네트워크 선호)
old = np.where(stim == "A", nl > nr, nr > nl)
rev = np.arange(T) >= flip
idx = np.where(rev)[0]
o["oldpref_early"] = float(old[idx[:400]].mean()) if len(idx) else float("nan")
o["oldpref_late"] = float(old[idx[-400:]].mean()) if len(idx) else float("nan")
print("E104SUM " + " ".join("%s=%s" % (k, ("%.6g" % v) if isinstance(v, float) else v) for k, v in o.items()))
