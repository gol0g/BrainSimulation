#!/usr/bin/env python3
"""E202 판정·독립 대조 합성 시험용 입력 생성기 — 정답(분해·기전·판정)을 알고 만든 뇌 5개 분량의 파일 트리.
KC 3000개/쪽, 경로마다 KC 하나당 시냅스 2개. 무리: 0~199 약한 교차(SXI +0.05)·200~399 약한 같은 쪽(−0.05)·400~499 강한 교차(+0.7)·
500~599 강한 같은 쪽(−0.7)·600~699 중간(+0.3, 반응)·나머지 비반응. 학습 Δ: 약한 무리 = D_obs(offset 만), 강한 ±200, 중간 +50, 비반응 0."""
import os

import numpy as np

CLASSES = ("S+", "S-", "C+", "C-")
KEY = {"S+": "Sp", "S-": "Sm", "C+": "Cp", "C-": "Cm"}
POPS = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")
NK = 3000
SCEN = {
    # 이름: (W 범주별 도파민 합, m 범주별 단위 도파민당 x)
    "M1": ({"S+": 10.0, "S-": -6.0, "C+": 6.0, "C-": -10.0}, {"S+": -2.0, "S-": -2.0, "C+": 2.0, "C-": 2.0}),
    "M2": ({"S+": 8.0, "S-": -2.0, "C+": 8.0, "C-": -2.0}, {"S+": -3.0, "S-": -3.0, "C+": 1.0, "C-": 1.0}),
    "MIX": ({"S+": 10.0, "S-": -2.0, "C+": 6.0, "C-": -2.0}, {"S+": -3.0, "S-": -3.0, "C+": 1.0, "C-": 1.0}),
}
# 정답: (D_obs, D_freq, D_mag, 기전)
ANSWER = {"M1": (-16.0, -16.0, 0.0, "M1"), "M2": (-12.0, 0.0, -12.0, "M2"), "MIX": (-20.0, -8.0, -12.0, "혼합")}


def _sxi():
    sx = np.zeros(NK)
    resp = np.zeros(NK, dtype=np.int8)
    sx[0:200], sx[200:400], sx[400:500], sx[500:600], sx[600:700] = 0.05, -0.05, 0.7, -0.7, 0.3
    resp[0:700] = 1
    return sx, resp


def _delta(dobs, weak_scale=1.0):
    d = np.zeros(NK)
    d[0:400] = dobs * weak_scale
    d[400:500], d[500:600], d[600:700] = 200.0, -200.0, 50.0
    return d


def write_brain(base, b, scen, weak_scale=1.0, flip200=False, n_rows=3000, drop=None, log_drop=None, own=1.0):
    """own: 약한 무리 x 중 자기 쪽 자극 시행 몫(정정 1 — 기본 1.0, 0.6 이면 분류 오염 시나리오)."""
    ex = os.path.join(base, "research", "experiments")
    for sub in ("traces/E202", "traces/E200", "traces/E196", "traces/E176/snap", "logs/E202"):
        os.makedirs(os.path.join(ex, sub), exist_ok=True)
    W, M = SCEN[scen]
    X = {c: W[c] * M[c] for c in CLASSES}
    dobs = sum(X.values())
    sx, resp = _sxi()
    pre = np.repeat(np.arange(NK), 2)
    np.savez(os.path.join(ex, "traces", "E176", "snap", "conn_b%d.npz" % b), **{nm + "__pre": pre.astype(np.int32) for nm in POPS})
    np.savez(os.path.join(ex, "traces", "E196", "sxi_b%d.npz" % b), resp_l=resp, resp_r=resp, sxi_l=sx, sxi_r=sx,
             tot_l=np.ones(NK, dtype=np.int32), tot_r=np.ones(NK, dtype=np.int32))

    def weights(d):
        w = {}
        for kn in "lr":
            oc = "r" if kn == "l" else "l"
            w["kc_%s_to_motor_%s" % (kn, oc)] = (150.0 + d[pre] / 2).astype(np.float32)   # 교차
            w["kc_%s_to_motor_%s" % (kn, kn)] = (150.0 - d[pre] / 2).astype(np.float32)   # 같은 쪽
        return w

    d_meas = _delta(dobs)
    d_learn = _delta(dobs, weak_scale)
    np.savez(os.path.join(ex, "traces", "E202", "w_b%d.npz" % b), **weights(d_learn))
    d200 = d_learn.copy()
    if flip200:
        d200[400:700] = -d200[400:700]
    np.savez(os.path.join(ex, "traces", "E200", "w_b%d.npz" % b), **weights(d200))
    # 계측: 약한 무리 KC 는 범주마다 X_c(상수), 나머지는 총 Δ 를 S+ 에
    dg = {}
    for c in CLASSES:
        for sd in "lr":
            x = np.zeros(NK)
            x[0:400] = X[c]
            if c == "S+":
                x[400:700] = d_meas[400:700]
            dg["x_%s_%s" % (KEY[c], sd)] = x
            dg["ec_%s_%s" % (KEY[c], sd)] = np.full(NK, 0.5)
            dg["es_%s_%s" % (KEY[c], sd)] = np.full(NK, 0.7)
    for sd in "lr":
        dg["tot_x_" + sd] = sum(dg["x_%s_%s" % (KEY[c], sd)] for c in CLASSES)
    # 정정 1: 자극 쪽별 분할 — 약한 무리는 own 몫을 자기 쪽 자극 칸에, 나머지를 반대쪽에. 다른 KC 는 전부 자기 쪽.
    for q in ("x", "ec", "es"):
        for c in CLASSES:
            for sd in "lr":
                tot = dg["%s_%s_%s" % (q, KEY[c], sd)]
                fr_own = np.ones(NK)
                if q == "x":
                    fr_own[0:400] = own
                for st in "lr":
                    dg["%ss_%s_%s_%s" % (q, KEY[c], st, sd)] = tot * (fr_own if st == sd else (1.0 - fr_own))
    # 정정 2: 도파민 성분 xd(정규화 전) — 합성에서는 x 의 0.9 배(정규화가 차를 10% 바꾼 꼴)
    for c in CLASSES:
        for sd in "lr":
            dg["xd_%s_%s" % (KEY[c], sd)] = 0.9 * dg["x_%s_%s" % (KEY[c], sd)]
            for st in "lr":
                dg["xds_%s_%s_%s" % (KEY[c], st, sd)] = 0.9 * dg["xs_%s_%s_%s" % (KEY[c], st, sd)]
    for sd in "lr":
        dg["tot_xd_" + sd] = 0.9 * dg["tot_x_" + sd]
    dg["Ws"] = np.asarray([[W[c] / 2, W[c] / 2] for c in CLASSES])
    N = np.array([900, 600, 700, 800])
    if n_rows != 3000:
        N = np.array([900, 600, 700, n_rows - 2200])
    rows = np.zeros((n_rows, 9))
    rows[:, 8] = np.repeat(np.arange(4), N)[:n_rows] if N.sum() == n_rows else 0
    dg["Ns"] = np.stack([N // 2, N - N // 2], axis=1)
    dg.update({"rows": rows, "row_cols": np.asarray(("trial", "stim_r", "ex_r", "correct", "same", "ctx_x", "cat_b", "da", "cls")),
               "classes": np.asarray(CLASSES), "W": np.asarray([W[c] for c in CLASSES]), "N": N})
    if drop != "dg":
        np.savez(os.path.join(ex, "traces", "E202", "dg_b%d.npz" % b), **dg)
    lines = ["[맥락 범주 과제] 시행 3000 중 맥락 X 1500 · Y 1500 · 정답 같은 쪽 1500 · 교차 1500",
             "[E199 경쟁 정규화] 적용 3000회 · 정규화 전 max|S−목표| 평균 0.1 최대 0.2 · 정규화 뒤 최대 1.0e-09 · hard=1",
             "[E200 도파민 통계] 시행 3000 · 평균 DA +0.0010 · 평균 |DA| 0.6000 · 기준선 1 · r̄ 끝 0.3000 · τ 100 · r̄₀ 0.25",
             "[E202 계측] 시행 1 범주 S+ · DA +0.7500 · 자극 left 실행 left",
             "[E202 계측] 시행 2 범주 C- · DA -0.7600 · 자극 right 실행 left",
             "[E202 계측] 시행 3 범주 S- · DA -0.7500 · 자극 left 실행 left",
             "[E202 계측] 시행 3000 · 범주 S+ 900 S- 600 C+ 700 C- 800 · 도파민 합 · 저장 x"]
    if log_drop is not None:
        lines = [ln for ln in lines if not ln.startswith(log_drop)]
    with open(os.path.join(ex, "logs", "E202", "train_b%d.log" % b), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_tree(base, scens, per_brain=None):
    """scens: 뇌 10~14 의 시나리오 목록. per_brain: {뇌(정수): {write_brain 키워드}} — 정수 키라 ** 로 풀지 않고 사전 그대로 받는다."""
    per_brain = per_brain or {}
    for i, b in enumerate((10, 11, 12, 13, 14)):
        write_brain(base, b, scens[i], **per_brain.get(b, {}))
