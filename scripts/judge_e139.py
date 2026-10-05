#!/usr/bin/env python3
"""E139 판정 — 기준 logs/E139/criteria_fixed.txt 의 **[수정] 기준**(2026-10-03, 본 표본 미관측 상태에서 경로 검사 근거로 대체).
뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다. 판정 수치는 추적 npz(traces/E139/tr_b*.npz, rows)에서 다시 계산하고 요약 줄과 대조한다.

단위(규약 P19): Δg = 시행 끝 g − 시행 시작 g 의 선택 KC 시냅스 합(교차·같은 쪽). A·B = 보상 시행 교차·같은 쪽 Δg 합, C·P = 처벌 시행.
e_da = 도파민 직전 자격흔적 합, e_end = 보상 창 끝 자격흔적 합. 독립 단위 = 뇌.
rows 열: 0 ep 1 t 2 side_r 3 explore 4 probe_r 5 v 6 ex_r 7 correct 8~11 dg(교차·같은쪽·비선택·무활동) 12 dg_전체 13~16 e_da
         17~20 dg_도파민전 21~24 e_end 25 보상창 motor_left 발화율 합 26 motor_right.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
POST = {10: -0.0838, 11: -0.0574, 12: -0.0428, 13: -0.0489, 14: -0.0442}     # E119 [사후]
REW = {10: 258, 11: 304, 12: 314, 13: 284, 14: 325}                           # E119 보상 수
TL = re.compile(r"^\s*e139 b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) \|\| => KCTRACE 시행 (\d+) 보상 (\d+) \| A ([-+0-9.]+) B ([-+0-9.]+) C ([-+0-9.]+) P ([-+0-9.]+)"
                r" .*?합일관성 최대 ([-+0-9.e]+) \| g연속 최대 ([-+0-9.e]+)")


def stats(rows):
    rw = rows[:, 7] == 1
    A, B = rows[rw, 8].sum(), rows[rw, 9].sum()
    C, P = rows[~rw, 8].sum(), rows[~rw, 9].sum()
    d_ = np.abs(rows[:, 8:12].sum(1) - rows[:, 12])
    cons_ok = bool(np.all((d_ <= 1e-6 * np.abs(rows[:, 12])) | (d_ <= 1e-3)))
    pre_ratio = float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12))
    eda = rows[rw, 14].sum() / rows[rw, 13].sum() if rows[rw, 13].sum() else float("nan")
    eend = rows[rw, 22].sum() / rows[rw, 21].sum() if rows[rw, 21].sum() else float("nan")
    blk = [float((rows[i * 100:(i + 1) * 100, 8] - rows[i * 100:(i + 1) * 100, 9]).sum()) for i in range(int(np.ceil(len(rows) / 100)))]
    return {"A": A, "B": B, "C": C, "P": P, "BA": (B / A) if A else float("nan"), "CP": (C / P) if P else float("nan"),
            "cons_ok": cons_ok, "pre_ratio": pre_ratio, "eda": eda, "eend": eend, "ml": float(rows[:, 25].mean()), "mr": float(rows[:, 26].mean()),
            "n": len(rows), "nrew": int(rw.sum()), "blk": blk, "dD": (A + C) - (B + P)}


def judge(T, S, W):
    """T: 요약 줄, S: npz 재계산 stats, W: 뇌 → (다른 시냅스 수, 최대 |차|, KC→motor 시냅스 수) — E119 가중치 대비."""
    miss = [b for b in BRAINS if b not in T or b not in S or b not in W]
    if miss:
        return ["[측정 확인] 결측 뇌 %s — **판정 보류, 수치 미출력**" % miss], None
    checks, ok = [], True
    v1 = [b for b in BRAINS if W[b][0] <= 0.003 * W[b][2] and abs(T[b]["nrew"] - REW[b]) <= 3 and abs(T[b]["post"] - POST[b]) <= 0.002 + 1e-9]   # 수정 2: 최대|차| 제외
    checks.append("[V1' 비간섭(잡음 바닥)] E119 대비 다른 시냅스 ≤0.3%%·보상 ±3·[사후] ±0.002(최대|차|는 보고만): %d/5 (다른 시냅스 %s, 보상 %s) %s"
                  % (len(v1), " ".join(str(W[b][0]) for b in BRAINS), " ".join(str(T[b]["nrew"]) for b in BRAINS), "통과" if len(v1) == 5 else "실패"))
    ok &= len(v1) == 5
    v2 = [b for b in BRAINS if S[b]["cons_ok"]]
    checks.append("[V2 합 일관성] %d/5 %s" % (len(v2), "통과" if len(v2) == 5 else "실패")); ok &= len(v2) == 5
    v3 = [b for b in BRAINS if S[b]["n"] == 500 and T[b]["gap"] == 0.0]
    checks.append("[V3 추적 완결] 시행 500·g 연속 0: %d/5 %s" % (len(v3), "통과" if len(v3) == 5 else "실패")); ok &= len(v3) == 5
    v4 = [b for b in BRAINS if S[b]["pre_ratio"] <= 1e-3]
    checks.append("[V4 단계 분리] 도파민 전 변화 ≤1e-3 × 시행 변화: %d/5 (%s) %s"
                  % (len(v4), " ".join("%.1e" % S[b]["pre_ratio"] for b in BRAINS), "통과" if len(v4) == 5 else "실패")); ok &= len(v4) == 5
    agree = [b for b in BRAINS if all(abs(S[b][k] - T[b][k]) <= max(0.06, 1e-6 * abs(S[b][k])) for k in ("A", "B", "C", "P"))]
    checks.append("[대조] 요약 줄 A·B·C·P = npz 재계산: %d/5 %s" % (len(agree), "통과" if len(agree) == 5 else "실패")); ok &= len(agree) == 5
    rw_ = [b for b in BRAINS if S[b]["BA"] >= 0.5 and S[b]["CP"] >= 0.5 and S[b]["eend"] >= 0.5 and S[b]["eda"] <= 0 and S[b]["ml"] > 0 and S[b]["mr"] > 0]
    dec = [b for b in BRAINS if S[b]["BA"] <= 0.2 and S[b]["CP"] <= 0.2]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif len(rw_) >= 4:
        verdict = "H062-rw — 보상 창이 비선택 자격흔적을 만들어 두 도파민 부호 모두에서 교차·같은 쪽이 함께 움직인다"
    elif len(dec) >= 4:
        verdict = "H062-dec — 결정 단계 흔적이 지배(변화가 실행 행동을 따름)"
    else:
        verdict = "보류"
    return checks, {"rw": rw_, "dec": dec, "ok": ok, "verdict": verdict}


def report(checks, res, T=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        s = S[b]
        print("b%d 보상 %d/%d | A %+.0f B %+.0f C %+.0f P %+.0f | B/A %.3f C/P %.3f | e_da 같은쪽/교차 %+.3f e_end %+.3f | 보상창 motor 좌 %.4f 우 %.4f | 블록 ΔD %s"
              % (b, s["nrew"], s["n"], s["A"], s["B"], s["C"], s["P"], s["BA"], s["CP"], s["eda"], s["eend"], s["ml"], s["mr"],
                 " ".join("%+.0f" % x for x in s["blk"])))
    print("H062-rw 조건 %d/5, H062-dec 조건 %d/5" % (len(res["rw"]), len(res["dec"])))
    print("판정: %s" % res["verdict"])


def load(check_weights=True):
    T, S, W = {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E139.log"), encoding="utf-8"):
            m = TL.match(ln)
            if m:
                g = m.groups()
                T[int(g[0])] = {"pre": float(g[1]), "post": float(g[2]), "n": int(g[3]), "nrew": int(g[4]), "A": float(g[5]), "B": float(g[6]),
                                "C": float(g[7]), "P": float(g[8]), "cons": float(g[9]), "gap": float(g[10])}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E139", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = stats(np.load(f)["rows"])
        if check_weights:
            fa = os.path.join(EXP, "traces", "E139", "w_b%d.npz" % b); fb = os.path.join(EXP, "traces", "E119", "w_rw0_b%d.npz" % b)
            if os.path.exists(fa) and os.path.exists(fb):
                a, c = np.load(fa), np.load(fb)
                if sorted(a.files) == sorted(c.files):
                    nd = sum(int((a[k] != c[k]).sum()) for k in a.files)
                    mx = max(float(np.abs(a[k] - c[k]).max()) for k in a.files)
                    nkm = sum(int(a[k].size) for k in a.files if k.startswith("kc_") and "_to_motor_" in k)
                    W[b] = (nd, mx, nkm)
    return T, S, W


if __name__ == "__main__":
    T, S, W = load()
    c, r = judge(T, S, W)
    report(c, r, T, S)
    sys.exit(0)
