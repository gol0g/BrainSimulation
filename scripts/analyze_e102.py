#!/usr/bin/env python3
"""E102 런별 분석: 추적 npz → 사건별 Δg 요약(텍스트 한 줄씩). 판정은 judge_e102.py 가 한다.
사용: analyze_e102.py trace.npz  → stdout 에 "E102SUM key=value ..." 줄들."""
import sys
import numpy as np

z = np.load(sys.argv[1])
gl = np.vstack([z["g0_l"][None, :], z["g_l"]])   # (T+1, n_kc) — 행 t+1 = 시행 t 도파민 후
gr = np.vstack([z["g0_r"][None, :], z["g_r"]])
dl, dr = np.diff(gl, axis=0), np.diff(gr, axis=0)   # 시행 t 의 Δg (KC별)
el, er = z["e_l"], z["e_r"]
stim, act, rew = z["stim"], z["act"], z["reward"]
flip = int(z["flip_at"])
ka, kb = set(z["kc_a"].tolist()), set(z["kc_b"].tolist())
A = np.array(sorted(ka - kb), dtype=int)
B = np.array(sorted(kb - ka), dtype=int)
T = len(stim)
rev = np.arange(T) >= flip if flip > 0 else np.zeros(T, bool)
acq = ~rev


def ev(mask, D, grp):
    idx = np.where(mask)[0]
    if len(idx) == 0 or len(grp) == 0:
        return 0, float("nan"), float("nan")
    v = D[np.ix_(idx, grp)].sum(axis=1)
    return len(idx), float(v.mean()), float((D[np.ix_(idx, grp)] > 0).mean())


out = {"nA": len(A), "nB": len(B), "flip": flip, "T": T}
# 획득 구간 측정 도구 확인: A·L·보상 → A전용→L 증가해야
n, m, _ = ev(acq & (stim == "A") & (act == "L") & rew, dl, A); out.update(acqAL_n=n, acqAL_dg=m)
# 반전 구간 사건
for name, mask, D, grp in (
    ("PA", rev & (stim == "A") & (act == "L") & ~rew, dl, A),   # 옛 정답(L) 처벌 → 옛 연합 A→L
    ("RA", rev & (stim == "A") & (act == "R") & rew, dr, A),    # 새 정답(R) 보상 → 새 연합 A→R
    ("PA_new", rev & (stim == "A") & (act == "L") & ~rew, dr, A),  # 처벌 때 새 연합 A→R 변화
    ("RA_old", rev & (stim == "A") & (act == "R") & rew, dl, A),   # 보상 때 옛 연합 A→L 변화
    ("PB", rev & (stim == "B") & (act == "R") & ~rew, dr, B),
    ("RB", rev & (stim == "B") & (act == "L") & rew, dl, B),
):
    n, m, fpos = ev(mask, D, grp); out.update({name + "_n": n, name + "_dg": m, name + "_fpos": fpos})
# 처벌 직전 자격흔적 부호(A전용→L)
idx = np.where(rev & (stim == "A") & (act == "L") & ~rew)[0]
out["PA_e_mean"] = float(el[np.ix_(idx, A)].sum(axis=1).mean()) if len(idx) and len(A) else float("nan")
# 순 변화: [A전용 L−R] 반전 시작→끝, [B전용 R−L]
if flip > 0:
    s0, s1 = flip, T   # gl 행 인덱스: flip = 반전 직전 상태, T = 끝
    out["DA"] = float((gl[s1, A] - gr[s1, A]).sum() - (gl[s0, A] - gr[s0, A]).sum())
    out["DB"] = float((gr[s1, B] - gl[s1, B]).sum() - (gr[s0, B] - gl[s0, B]).sum())
    out["DA_start"] = float((gl[s0, A] - gr[s0, A]).sum())
    # 사건 종류별 옛 연합(A→L) 순기여 합
    for name, mask in (("all", rev), ("A_L_pun", rev & (stim == "A") & (act == "L") & ~rew),
                       ("A_R_rew", rev & (stim == "A") & (act == "R") & rew),
                       ("B_any", rev & (stim == "B"))):
        out["cAL_" + name] = float(dl[np.ix_(np.where(mask)[0], A)].sum()) if mask.any() and len(A) else 0.0
print("E102SUM " + " ".join("%s=%s" % (k, ("%.6g" % v) if isinstance(v, float) else v) for k, v in out.items()))
