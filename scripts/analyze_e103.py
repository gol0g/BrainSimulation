#!/usr/bin/env python3
"""E103 런별 분석: 추적 npz → 누수·끝 상태·판독 요약 한 줄("E103SUM ..."). 판정은 judge_e103.py.
누수: 반전 구간 A 시행·행동 R(새 정답)·보상 사건에서 (1) 행동 창 비선택 출력(L) 발화, (2) 옛 연합 Δg(A전용→L).
끝 상태: 반전 끝 A전용 (L−R) 합의 부호, 그리고 **공유 KC 포함** A 반응 KC 전체의 (L−R) 합 부호(판독 근사)."""
import sys
import numpy as np

z = np.load(sys.argv[1])
gl = np.vstack([z["g0_l"][None, :], z["g_l"]]); gr = np.vstack([z["g0_r"][None, :], z["g_r"]])
dl, dr = np.diff(gl, axis=0), np.diff(gr, axis=0)
stim, act, rew = z["stim"], z["act"], z["reward"]
spk = z["act_spk"]; flip = int(z["flip_at"]); T = len(stim)
ka, kb = set(z["kc_a"].tolist()), set(z["kc_b"].tolist())
A = np.array(sorted(ka - kb), int); B = np.array(sorted(kb - ka), int); S = np.array(sorted(ka & kb), int)
Aall = np.array(sorted(ka), int); Ball = np.array(sorted(kb), int)
rev = np.arange(T) >= flip
o = {"nA": len(A), "nB": len(B), "nShared": len(S)}


def m(x):
    return float(np.mean(x)) if len(x) else float("nan")


rn = np.where(rev & (stim == "A") & (act == "R") & rew)[0]   # 새 행동 보상
po = np.where(rev & (stim == "A") & (act == "L") & ~rew)[0]  # 옛 행동 처벌
o["RN_n"] = len(rn); o["PO_n"] = len(po)
o["RN_spkL"] = m(spk[rn, 0]) if len(spk) else float("nan")   # 새 행동(R) 창에서 옛 출력 L 발화
o["RN_spkR"] = m(spk[rn, 1]) if len(spk) else float("nan")
o["RN_dOld"] = m(dl[np.ix_(rn, A)].sum(axis=1)) if len(rn) else float("nan")   # 보상 때 옛 연합 Δg
o["RN_dNew"] = m(dr[np.ix_(rn, A)].sum(axis=1)) if len(rn) else float("nan")
o["PO_dOld"] = m(dl[np.ix_(po, A)].sum(axis=1)) if len(po) else float("nan")
if len(rn) > 3 and len(spk):
    x = spk[rn, 0].astype(float); y = dl[np.ix_(rn, A)].sum(axis=1)
    o["RN_corr_spkL_dOld"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
else:
    o["RN_corr_spkL_dOld"] = float("nan")
o["cOld_pun"] = float(dl[np.ix_(po, A)].sum()) if len(po) else 0.0
o["cOld_rewNew"] = float(dl[np.ix_(rn, A)].sum()) if len(rn) else 0.0
for tag, grp, sgn in (("A", A, 1), ("Aall", Aall, 1), ("B", B, -1), ("Ball", Ball, -1)):
    # 옛 규칙 방향 = A→L, B→R. (L−R)·sgn 양수 = 옛 쪽
    o["start_" + tag] = float(sgn * (gl[flip, grp] - gr[flip, grp]).sum()) if len(grp) else float("nan")
    o["end_" + tag] = float(sgn * (gl[T, grp] - gr[T, grp]).sum()) if len(grp) else float("nan")
print("E103SUM " + " ".join("%s=%s" % (k, ("%.6g" % v) if isinstance(v, float) else v) for k, v in o.items()))
