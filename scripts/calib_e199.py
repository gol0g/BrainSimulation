#!/usr/bin/env python3
"""E199 학습률 보정(사전 규칙 — logs/E199/criteria_fixed.txt): 표본 밖 뇌 15 를 η ∈ {0.15, 0.6, 2.4} 로 각 3,000시행 학습(경쟁형 규칙)한 가중치에서
뇌 15 지수(traces/E197/pathcheck/sxi_b15.npz — E196 경로 검사 −4900 측정)의 강한 선택 반응 KC(|SXI| ≥ 0.5) 중 |Δ| ≥ 250 비율을 세고,
그 비율 ≥ 0.5 인 가장 작은 η 를 고른다(없으면 2.4). 평가(행동)는 보지 않는다. 각 η 의 학습 로그는 정규화 적용 3,000회·hard=1·eta 일치·KC 별 합 300 을 확인 — 어기면 보정 실패(선택 없음).
입력: logs/E199/calib/train_b15_eta{η}.log, traces/E199/calib/w_b15_eta{η}.npz, 스냅숏 traces/E176/snap/conn_b15.npz. 출력: logs/E199/calib_eta.txt(선택)·표준 출력(η 별 줄).
실행: python3 scripts/calib_e199.py [루트]"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backend", "genesis"))
import judge_e198 as J8  # noqa: E402
import judge_e199 as J9  # noqa: E402
import kc_comp  # noqa: E402

ETAS = ("0.15", "0.6", "2.4")
THR_D, THR_F = 250.0, 0.5


def main(root="."):
    E = os.path.join(root, "research", "experiments")
    sn = os.path.join(E, "traces", "E176", "snap", "conn_b15.npz")
    Z = np.load(os.path.join(E, "traces", "E197", "pathcheck", "sxi_b15.npz"))
    res, bad = {}, []
    for et in ETAS:
        lg = os.path.join(E, "logs", "E199", "calib", "train_b15_eta%s.log" % et)
        wf = os.path.join(E, "traces", "E199", "calib", "w_b15_eta%s.npz" % et)
        try:
            cp = J9.read_comp(lg)
            W, S = np.load(wf), np.load(sn)
        except (OSError, ValueError, KeyError) as ex:
            bad.append("η %s 결측 %s" % (et, type(ex).__name__))
            continue
        cm = cp["comp"]
        if not (cp["hard"] and cm is not None and len(cm) == 3 and cm[0] == 3000 and cm[1] <= 1e-3 and cm[2] == 1
                and cp["eta"] is not None and abs(cp["eta"] - float(et)) <= 1e-9):
            bad.append("η %s 규칙 줄 %s %s %s" % (et, cp["hard"], cm, cp["eta"]))
            continue
        dmax = 0.0
        for kn in "lr":
            a, b = "kc_%s_to_motor_l" % kn, "kc_%s_to_motor_r" % kn
            s, both = kc_comp.kc_sums(S[a + "__pre"], W[a], S[b + "__pre"], W[b], 3000)
            dmax = max(dmax, float(np.max(np.abs(s[both] - 300.0))))
        if dmax > 0.01:
            bad.append("η %s KC 합 최대 |S−300| %.4g" % (et, dmax))
            continue
        D, _ = J8.deltas(wf, sn)
        st = wk = st_sat = wk_sat = 0
        for sd in "lr":
            rp = Z["resp_" + sd].astype(bool); sx = np.abs(np.round(Z["sxi_" + sd], 9))
            m_st = rp & (sx >= 0.5); m_wk = rp & (sx < 0.5)
            st += int(m_st.sum()); wk += int(m_wk.sum())
            st_sat += int(np.sum(m_st & (np.abs(D[sd]) >= THR_D))); wk_sat += int(np.sum(m_wk & (np.abs(D[sd]) >= THR_D)))
        res[et] = (st_sat / max(st, 1), st_sat, st, wk_sat, wk)
        print("[E199 보정] η %s | 강한 선택 반응 KC |Δ| ≥ %.0f %d/%d(%.3f) · 약한 선택 %d/%d · 정규화 %d회·뒤 최대 %.2e" % (et, THR_D, st_sat, st, res[et][0], wk_sat, wk, cm[0], cm[1]))
    if bad:
        print("[E199 보정] 실패: " + "; ".join(bad))
        print("보정: 선택 없음(보정 실패)")
        return 0
    pick = next((et for et in ETAS if res[et][0] >= THR_F), ETAS[-1])
    why = "비율 ≥ %.1f 인 가장 작은 η" % THR_F if res[pick][0] >= THR_F else "기준 충족 η 없음 → 가장 큰 η"
    line = "[E199 보정] 선택 eta=%s (사전 규칙: %s; 강한 선택 반응 KC |Δ| ≥ %.0f 비율 %s)" % (pick, why, THR_D, " · ".join("%s %.3f" % (e, res[e][0]) for e in ETAS))
    with open(os.path.join(E, "logs", "E199", "calib_eta.txt"), "w", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
