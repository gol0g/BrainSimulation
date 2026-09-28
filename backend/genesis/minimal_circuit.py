#!/usr/bin/env python3
"""최소 회로 — 보상이 행동을 바꾸는가 (E086).

## 왜 여기까지 내려오는가
E085에서 전체 모델(28,323 뉴런)의 R-STDP가 **가중치는 크게 바꾸는데**(변화율 85.6%,
교차 경로 std 0→0.87) **행동은 +0.0007밖에 안 바꾼다**는 것이 검증된 측정 위에서 확인됐다.
E069부터 16건을 병목 좁히기에 썼지만, 쫓던 효과 자체가 측정 바닥 아래였다.

전체 모델에서 원인을 분리하는 대신, **같은 뉴런·같은 가소성 코드**로 만든 최소 회로에서
"보상이 행동을 바꾼다"를 먼저 성립시킨다. 성립하지 않으면 학습 규칙 자체의 문제이고,
성립하면 그 회로를 기준점으로 삼아 무엇을 더할 때 깨지는지 추적할 수 있다.

## 회로
    자극 A ─┐                    ┌─> 행동 L
            ├─> [KC 스파스 층] ──┤
    자극 B ─┘                    └─> 행동 R

- 감각 2집단(A/B) → KC(스파스 확장) → 출력 2집단(L/R). **KC→출력만 R-STDP.**
- 규칙: A일 때 L, B일 때 R 이 정답.
- **선천 배선 없음** — 정답 경로가 미리 깔려 있지 않다.
  (C47의 "개념 5층이 전부 선천이었다"가 여기서 재발하지 않게 한다.)
- **탐색**: 확률 epsilon 으로 무작위 행동을 실제로 실행한다(자격흔적에 남도록).

## 성공 기준 (사전 선언 — E086)
1. **학습**: 정답률이 무작위(50%)보다 유의하게 높아진다.
2. **무보상 대조**: 보상을 끊으면 50% 근처에 머문다.
3. **수반성 대조(yoked)**: 행동과 무관하게 **같은 빈도**로 보상하면 학습되지 않는다.
   보상의 양이 아니라 **행동-보상 수반성**이 원인임을 보인다.
4. **규칙 반전**: A→R, B→L 로 뒤집으면 재학습한다.
5. **동결 유지**: 학습을 멈춘 뒤에도 성능이 유지된다.
"""
import argparse
import io
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pygenn import (GeNNModel, create_neuron_model, init_postsynaptic,
                    init_sparse_connectivity, init_var, init_weight_update)
from rstdp_model import DEFAULT_PARAMS, make_rstdp_model

PATTERNS = {}
BASE = [0.0]          # 기대 보상(러닝 평균). 리스트로 둬서 함수 안에서 갱신한다.
# E102: 시행별 사건 추적(--trace-file). None 이면 아무것도 읽지 않는다 — 동역학 불변(기기에서 읽기만 한다).
TRACE = None


def _pre_sums(s, var, n_pre):
    """시냅스 변수(var)를 **전시냅스(KC)별로 합산**한다. 추적 전용 — 값을 쓰지 않는다."""
    s.vars[var].pull_from_device()
    v = np.asarray(s.vars[var].values, dtype=np.float64).ravel()
    return np.bincount(TRACE["pre"][id(s)], weights=v, minlength=n_pre)
RULE = {"A": "L", "B": "R"}
FLIP = {"A": "R", "B": "L"}


def build(args):
    """전체 모델과 **같은 rstdp_model**을 쓴다. 최소 회로용 학습 규칙을 새로 만들지 않는다."""
    m = GeNNModel("float", "minimal_circuit")
    m.dt = 1.0
    # ★INV-A1. 2026-09-20 사고: 이 줄이 없어서 **연결 추첨이 매 실행 달라졌다**.
    # 실측: 같은 조건 3회에 eval = 54.0 / 46.0 / 100.0, frozen과 noreward의 연결 개수가
    # n=7987 vs 8078 로 달랐다(가중치는 양쪽 다 |Δ|=0). E086 25런 전체가 이 위에 있었고,
    # "seed2가 100% 학습"도 배선 추첨 운이었다.
    # 사전등록에는 `[x] INV-A1 genn_seed 고정`으로 체크해 두었다 — **체크만 하고 확인하지 않았다.**
    # E081의 프로브 불변식 위반과 같은 유형이다(규약 P15).
    m.seed = 12345 + args.seed      # forager_brain.py:1596 과 같은 방식

    lif_p = {"C": 1.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
             "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0}
    lif_v = {"V": -65.0, "RefracTime": 0.0}

    # 감각은 **한 집단**이고, 자극 A/B는 그 위의 서로 다른 패턴이다.
    # 2026-09-19 실측: 자극이 감각 집단 **전체**를 켜면 양쪽에 연결된 KC가 무조건 둘 다에
    # 반응한다 → 자카드 99.5%, KC 798/800 발화. 스파스 확장이 아무것도 분리하지 못했다.
    # 실제 버섯체에서 냄새는 사구체의 **부분집합**을 켠다. 그 구조로 바꾼다.
    sens_lif = create_neuron_model(
        "SensLIF",
        params=["C", "TauM", "Vrest", "Vreset", "Vthresh", "TauRefrac"],
        vars=[("V", "scalar"), ("RefracTime", "scalar"), ("I_input", "scalar")],
        sim_code="""
        if (RefracTime <= 0.0) {
            const scalar alpha = I_input * (TauM / C) + Vrest;
            V = alpha - (exp(-dt / TauM) * (alpha - V));
        } else {
            RefracTime -= dt;
        }
        """,
        threshold_condition_code="RefracTime <= 0.0 && V >= Vthresh",
        reset_code="V = Vreset; RefracTime = TauRefrac;")
    sens_p = {k: v for k, v in lif_p.items() if k != "Ioffset"}
    pops = {}
    pops["sens"] = m.add_neuron_population(
        "sens", args.n_sens, sens_lif, sens_p,
        {"V": -65.0, "RefracTime": 0.0, "I_input": 0.0})
    pops["kc"] = m.add_neuron_population("kc", args.n_kc, "LIF", lif_p, lif_v)
    pops["out_l"] = m.add_neuron_population("out_l", args.n_out, "LIF", lif_p, lif_v)
    pops["out_r"] = m.add_neuron_population("out_r", args.n_out, "LIF", lif_p, lif_v)
    for p in pops.values():
        p.spike_recording_enabled = True
    # 고른 행동을 출력에 주입하려면 Ioffset을 동적으로 바꿀 수 있어야 한다.
    pops["out_l"].set_param_dynamic("Ioffset")
    pops["out_r"].set_param_dynamic("Ioffset")
    # 감각 → KC: 고정 스파스. 학습하지 않는다(버섯체: KC 입력은 무작위 고정).
    m.add_synapse_population(
        "sens_kc", "SPARSE", pops["sens"], pops["kc"],
        init_weight_update("StaticPulse", {},
                           {"g": init_var("Constant", {"constant": args.sens_kc_w})}),
        init_postsynaptic("ExpCurr", {"tau": 5.0}),
        init_sparse_connectivity("FixedProbability", {"prob": args.sens_kc_p}))

    # KC 전역 억제 — 스파스 코딩을 강제한다(실제 버섯체의 APL 피드백).
    # 이것이 없으면 KC가 800개 중 798개 발화해 스파스 확장이 무의미해진다(실측).
    if args.kc_inh > 0:
        inh = m.add_neuron_population("kc_inh", args.n_inh, "LIF", lif_p, lif_v)
        inh.spike_recording_enabled = True
        pops["kc_inh"] = inh
        m.add_synapse_population(
            "kc_to_inh", "SPARSE", pops["kc"], inh,
            init_weight_update("StaticPulse", {},
                               {"g": init_var("Constant", {"constant": args.kc_inh_drive})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": 0.5}))
        m.add_synapse_population(
            "inh_to_kc", "SPARSE", inh, pops["kc"],
            init_weight_update("StaticPulse", {},
                               {"g": init_var("Constant", {"constant": -args.kc_inh})}),
            init_postsynaptic("ExpCurr", {"tau": 10.0}),
            init_sparse_connectivity("FixedProbability", {"prob": 0.5}))

    # KC → 출력: **유일한 학습 경로**.
    kp = dict(DEFAULT_PARAMS)
    kp["w_max"] = args.w_max
    if args.tau_e is not None:
        kp["tau_e"] = args.tau_e
    # frozen: 학습률 0. **배선만으로 나오는 정답률**을 잰다.
    # 첫 구간부터 70%가 나왔다 — 선천 배선이 정답을 만들고 있는지 먼저 갈라야 한다(C47 전례).
    kp["eta"] = 0.0 if args.mode == "frozen" else args.eta
    wu = make_rstdp_model()
    syn = {}
    for nm in ("l", "r"):
        syn[nm] = m.add_synapse_population(
            "kc_out_%s" % nm, "SPARSE", pops["kc"], pops["out_" + nm],
            init_weight_update(wu, kp,
                               {"g": init_var("Constant", {"constant": args.kc_out_w}), "e": 0.0},
                               {"preTrace": 0.0}, {"postTrace": 0.0}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": args.kc_out_p}))
        # `dopamine`은 일반 param으로 선언돼 있으므로 **동적 지정**해야 매 스텝 바꿀 수 있다.
        # rstdp_model.py 주석에 적혀 있는 절차인데 빠뜨려 `IndexError: unordered_map::at`가 났다.
        syn[nm].set_wu_param_dynamic("dopamine")

    # 출력 상호 억제 — 승자가 하나 나오게. 학습하지 않는다.
    for src, dst, nm in (("out_l", "out_r", "lr"), ("out_r", "out_l", "rl")):
        m.add_synapse_population(
            "out_wta_%s" % nm, "SPARSE", pops[src], pops[dst],
            init_weight_update("StaticPulse", {},
                               {"g": init_var("Constant", {"constant": -args.wta})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": 0.5}))

    m.build()
    m.load(num_recording_timesteps=max(args.steps, args.da_steps))
    return m, pops, syn


def make_patterns(args):
    """자극 A/B = 감각 집단 위의 서로 다른 **부분집합**. 겹침은 args.overlap 으로 준다."""
    rs = np.random.RandomState(1000 + args.seed)
    n, k = args.n_sens, int(args.n_sens * args.pattern_frac)
    shared = rs.choice(n, int(k * args.overlap), replace=False)
    rest = np.setdiff1d(np.arange(n), shared)
    a_only = rs.choice(rest, k - len(shared), replace=False)
    rest2 = np.setdiff1d(rest, a_only)
    b_only = rs.choice(rest2, k - len(shared), replace=False)
    pa = np.zeros(n); pa[shared] = 1; pa[a_only] = 1
    pb = np.zeros(n); pb[shared] = 1; pb[b_only] = 1
    pats = {"A": pa, "B": pb}
    if getattr(args, "n_stim", 2) == 4:
        # E107: C/D 는 A/B **다음에** 같은 난수열에서 뽑는다 → A/B 는 2자극 런과 동일(1단계 쌍둥이 성립).
        # C/D 는 감각 집단 전체에서 뽑아 A/B 와 무작위로 겹친다(KC 표현 공유 → 간섭 여지).
        shared2 = rs.choice(n, int(k * args.overlap), replace=False)
        rest3 = np.setdiff1d(np.arange(n), shared2)
        c_only = rs.choice(rest3, k - len(shared2), replace=False)
        rest4 = np.setdiff1d(rest3, c_only)
        d_only = rs.choice(rest4, k - len(shared2), replace=False)
        pc = np.zeros(n); pc[shared2] = 1; pc[c_only] = 1
        pd = np.zeros(n); pd[shared2] = 1; pd[d_only] = 1
        pats["C"], pats["D"] = pc, pd
    return pats


def exemplar_levels(args):
    return [float(x) for x in args.distort_test.split(",") if x.strip()]


def make_exemplars(args):
    """E122: 범주 사례. 원형(PATTERNS A/B)의 활성 비트 m개를 원형에 없는 비트 m개로 바꾼다(m = round(d × 활성 수)).
    추가 비트는 원형 밖 전체(다른 범주 비트 포함)에서 뽑는다. 훈련·평가 사례와 원형은 서로 모두 다르다(집합 수준).
    별도 난수열(RandomState 2000+seed) — 배선·시행 난수열을 건드리지 않는다."""
    rs = np.random.RandomState(2000 + args.seed)
    levels = exemplar_levels(args)
    out = {}
    for cat in ("A", "B"):
        p = PATTERNS[cat]
        on = np.flatnonzero(p > 0)
        off = np.flatnonzero(p == 0)
        seen = {tuple(on.tolist())}

        def draw(d, _p=p, _on=on, _off=off, _seen=seen):
            m = int(round(d * len(_on)))
            if m < 1:
                raise SystemExit("왜곡 %.3f 은 교체 비트 0개 — 원형과 같다" % d)
            for _ in range(10000):
                q = _p.copy()
                q[rs.choice(_on, m, replace=False)] = 0
                q[rs.choice(_off, m, replace=False)] = 1
                key = tuple(np.flatnonzero(q).tolist())
                if key not in _seen:
                    _seen.add(key)
                    return q
            raise SystemExit("사례 생성 실패(중복만 나옴): d=%.3f" % d)
        out[cat] = {"train": [draw(args.distort_train) for _ in range(args.exemplars)],
                    "test": {d: [draw(d) for _ in range(args.n_test_ex)] for d in levels}}
    return out


def make_samediff(args):
    """E124: 같음/다름 관계 과제. 감각 집단을 두 반쪽(0~h-1, h~2h-1)으로 나누고 항목 코드(반쪽 안 활성 k개)를
    양쪽에 놓는다. 같은 항목 쌍 = 'S', 다른 항목 쌍 = 'D'. 항목 0..T-1 은 훈련, T..N-1 은 평가 전용(처음 보는 항목).
    별도 난수열(RandomState 4000+seed)."""
    rs = np.random.RandomState(4000 + args.seed)
    h = args.n_sens // 2
    k = int(round(args.sd_frac * h))
    codes, seen = [], set()
    while len(codes) < args.sd_items:
        c = tuple(sorted(rs.choice(h, k, replace=False).tolist()))
        if c not in seen:
            seen.add(c); codes.append(np.array(c))
    pats = {}
    for i in range(args.sd_items):
        for j in range(args.sd_items):
            q = np.zeros(args.n_sens)
            q[codes[i]] = 1
            q[h + codes[j]] = 1
            key = ("S%d" % i) if i == j else ("D%d_%d" % (i, j))
            pats[key] = (q, "L" if i == j else "R")
    return codes, pats


def samediff_halves(args, codes):
    """E124 경로 검사용: 항목 i 를 한쪽 반쪽에만 놓은 패턴(훈련·평가에 안 쓴다)."""
    h = args.n_sens // 2
    out = {}
    for i, c in enumerate(codes):
        q1 = np.zeros(args.n_sens); q1[c] = 1
        q2 = np.zeros(args.n_sens); q2[h + c] = 1
        out["H1_%d" % i], out["H2_%d" % i] = q1, q2
    return out


def register_exemplar(key, arr, cat):
    PATTERNS[key] = arr
    RULE[key] = RULE[cat]


def apply_stim(pops, args, stim):
    v = pops["sens"].vars["I_input"]
    arr = np.zeros(args.n_sens, dtype=np.float32)
    if stim is not None:
        arr[:] = PATTERNS[stim] * args.stim_i
    v.values = arr
    v.push_to_device()


def set_dopamine(syn, val):
    for s in syn.values():
        s.set_dynamic_param_value("dopamine", float(val))


def read_g(s):
    s.pull_connectivity_from_device()
    s.vars["g"].pull_from_device()
    v = s.vars["g"].values
    if v is None:
        raise RuntimeError("가중치를 읽지 못했다")
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size == 0:
        raise RuntimeError("가중치가 비었다")
    return a


def run_trial(m, pops, syn, stim, args, rng, rewarded_fn):
    """한 시행: 자극 제시 → 출력 스파이크 집계 → 행동 결정 → 보상 전달."""
    # 시행 사이 무자극 구간 — 앞 시행의 억제가 남아 다음 시행을 누르지 않게.
    apply_stim(pops, args, None)
    set_dopamine(syn, 0.0)
    for _ in range(args.gap_steps):
        m.step_time()
    m.pull_recording_buffers_from_device()

    apply_stim(pops, args, stim)
    set_dopamine(syn, 0.0)

    for _ in range(args.steps):
        m.step_time()
    m.pull_recording_buffers_from_device()
    nl = len(pops["out_l"].spike_recording_data[0][1])
    nr = len(pops["out_r"].spike_recording_data[0][1])

    if args.mode == "supervised":
        # 정답만 강제하면 **항상 보상** → 도파민이 계속 +1 → 전 시냅스가 w_max 로 포화한다.
        # 2026-09-25 실측: 가중치 0.5→12~20, 여러 곳이 19.999=w_max. s4는 A·B 둘 다 out_L 쪽 +16.
        # rstdp_model.py:60 에 같은 실패가 기록돼 있다. **정답/오답을 반반 강제**해 양쪽 부호를 넣는다.
        act = RULE[stim] if rng.random() < args.sup_correct else ("R" if RULE[stim] == "L" else "L")
    elif rng.random() < args.epsilon:
        act = "L" if rng.random() < 0.5 else "R"
    elif nl != nr:
        act = "L" if nl > nr else "R"
    else:
        act = "L" if rng.random() < 0.5 else "R"

    # ★신용 할당의 핵심: **고른 행동을 자격흔적에 남긴다.**
    # 2026-09-19 실측: 이것 없이 돌리니 정답률이 배선 기준선 87%에서 16~22%로 **떨어졌다**.
    # 탐색으로 고른 행동은 네트워크가 만든 활동과 다른데, 흔적에는 네트워크 활동이 남는다.
    # 그 상태에서 정답을 보상하면 **네트워크가 원래 하려던 오답이 강화된다**.
    # C52가 같은 교훈이다("motor에 강제한 행동이 상류 자격흔적에 담기지 않는다").
    if args.act_drive > 0:
        pops["out_l"].set_dynamic_param_value("Ioffset", args.act_drive if act == "L" else 0.0)
        pops["out_r"].set_dynamic_param_value("Ioffset", args.act_drive if act == "R" else 0.0)
        for _ in range(args.act_steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        if TRACE is not None:
            # E103: 행동 창에서 **비선택 출력도 발화하는가** — 발화하면 그쪽 자격흔적도 양으로 남아 보상이 옛 연합까지 강화한다.
            TRACE["act_spk"].append((len(pops["out_l"].spike_recording_data[0][1]),
                                     len(pops["out_r"].spike_recording_data[0][1])))
        pops["out_l"].set_dynamic_param_value("Ioffset", 0.0)
        pops["out_r"].set_dynamic_param_value("Ioffset", 0.0)

    # 보상 지연: 행동 창이 끝난 뒤 **아무것도 안 하고** 기다린다.
    # act_steps 를 늘리면 창과 지연이 함께 늘어 원인을 못 가른다(K41). 여기서 지연만 바꾼다.
    if args.delay_steps > 0:
        apply_stim(pops, args, None)
        for _ in range(args.delay_steps):
            m.step_time()
        m.pull_recording_buffers_from_device()

    give = rewarded_fn(stim, act)
    if TRACE is not None:
        # 도파민 직전의 자격흔적 — 이 값의 부호가 처벌 시 옛 연합이 깎이는지 강화되는지를 정한다.
        TRACE["e_l"].append(_pre_sums(syn["l"], "e", TRACE["n_pre"]))
        TRACE["e_r"].append(_pre_sums(syn["r"], "e", TRACE["n_pre"]))
    # 정답에 보상만 주면 가중치가 올라가기만 해 한쪽이 상한으로 폭주한다
    # (실측: kc_out_l 평균 0.500→20.000 = w_max, std 0.0026). 오답에 음의 도파민이 필요하다.
    #
    # 단 **무보상 대조(noreward)는 벌도 주지 않는다**. 2026-09-19 사고: `give`가 항상 False라
    # 매 시행 음의 도파민이 들어가 "학습 없음"이 아니라 "전부 벌"이 됐다. 대조가 아니었다.
    if args.mode == "noreward":
        set_dopamine(syn, 0.0)
    elif args.baseline > 0:
        # **보상 예측 오차**를 쓴다. 보상 자체를 쓰면 기댓값이 0이 아니라 가중치가 표류한다.
        #   비대칭(+1.0/-0.5), 정답률 50% → 기댓값 +0.25  (실측: 한쪽 출력이 30→300 스파이크로 폭주)
        #   대칭(+1.0/-1.0), 정답률 55% → 기댓값 +0.10  (폭주가 느려질 뿐 멈추지 않음)
        # 기대 보상을 빼면 기댓값이 자동으로 0이 된다 — Frémaux et al. 의 요지.
        r = 1.0 if give else -1.0
        BASE[0] += args.baseline * (r - BASE[0])
        set_dopamine(syn, args.da * (r - BASE[0]))
    else:
        set_dopamine(syn, args.da if give else -args.da_neg)
    apply_stim(pops, args, None)
    for _ in range(args.da_steps):
        m.step_time()
    m.pull_recording_buffers_from_device()
    set_dopamine(syn, 0.0)
    if TRACE is not None:
        TRACE["g_l"].append(_pre_sums(syn["l"], "g", TRACE["n_pre"]))
        TRACE["g_r"].append(_pre_sums(syn["r"], "g", TRACE["n_pre"]))
        TRACE["ev"].append((stim, act, bool(give), nl, nr))
    return act, nl, nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-file", type=str, default=None,
                    help="shuffled 모드용. 같은 시드·난수열의 learn 이 남긴 보상 계열(0/1 한 줄씩).")
    ap.add_argument("--dump-rewards", type=str, default=None,
                    help="learn 모드에서 시행별 보상 여부(0/1)를 이 파일에 기록한다.")
    ap.add_argument("--trial-seed", type=int, default=None,
                    help="E089: **시행 난수열만** 따로 고정한다. 배선(--seed)은 그대로 두고 "
                         "행동·자극 순서만 바꿔, 학습 결과가 난수열에 얼마나 좌우되는지 잰다. "
                         "미지정이면 --seed 를 쓴다(기존 동작).")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--da-steps", type=int, default=20)
    ap.add_argument("--n-sens", type=int, default=100)
    ap.add_argument("--n-kc", type=int, default=800)
    ap.add_argument("--n-out", type=int, default=50)
    ap.add_argument("--sens-kc-w", type=float, default=2.0)
    ap.add_argument("--sens-kc-p", type=float, default=0.05)
    ap.add_argument("--kc-out-w", type=float, default=0.5)
    ap.add_argument("--kc-out-p", type=float, default=0.2)
    ap.add_argument("--wta", type=float, default=4.0)
    ap.add_argument("--stim-i", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--w-max", type=float, default=20.0)
    ap.add_argument("--tau-e", type=float, default=None,
                    help="E093: 자격흔적 시정수(기본 200). 동시활동에서 도파민까지 35~65스텝 걸린다. "
                         "짧으면 신용이 안 남고 길면 앞 시행 흔적이 섞인다.")
    ap.add_argument("--da", type=float, default=1.0)
    ap.add_argument("--da-neg", type=float, default=0.5,
                    help="오답 시 음의 도파민. 0이면 가중치가 올라가기만 해 한쪽이 상한으로 폭주한다.")
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--act-drive", type=float, default=6.0,
                    help="고른 행동을 해당 출력 집단에 주입해 자격흔적에 남긴다. 0이면 끔 — "
                         "끄면 탐색 행동이 흔적에 안 남아 학습이 반대로 간다(실측 87%%→16%%).")
    ap.add_argument("--act-steps", type=int, default=15)
    ap.add_argument("--delay-steps", type=int, default=0,
                    help="E097: 행동 주입이 끝난 뒤 도파민까지의 **무자극 대기**. "
                         "act_steps 는 행동 창 길이와 보상 지연을 **동시에** 바꾼다(K41 교락). "
                         "이 인자로 지연만 따로 늘려 둘을 분리한다.")
    ap.add_argument("--sup-correct", type=float, default=0.5,
                    help="supervised 모드에서 **정답 행동을 강제할 확률**. 0.5면 정답/오답 반반이라 "
                         "보상과 벌이 균형을 이룬다. 1.0이면 항상 보상 → w_max 포화(실측).")
    ap.add_argument("--eval-trials", type=int, default=100,
                    help="훈련 후 **탐색 없이** 정답률을 재는 시행 수. 훈련 중 정답률은 "
                         "탐색에 오염되므로 학습 성과를 나타내지 못한다.")
    ap.add_argument("--baseline", type=float, default=0.0,
                    help="보상 예측 오차의 기준선 학습률(0이면 끔). 기대 보상을 빼서 "
                         "도파민 기댓값을 0으로 만든다. 0이면 가중치가 한쪽으로 표류한다.")
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--n-inh", type=int, default=40)
    ap.add_argument("--kc-inh", type=float, default=6.0,
                    help="KC 전역 억제 강도(APL 모사). 0이면 끔 — 끄면 KC가 거의 전부 발화한다.")
    ap.add_argument("--kc-inh-drive", type=float, default=0.5)
    ap.add_argument("--pattern-frac", type=float, default=0.3,
                    help="자극 하나가 켜는 감각 뉴런 비율.")
    ap.add_argument("--overlap", type=float, default=0.2,
                    help="A와 B 패턴이 공유하는 비율. 0이면 완전 분리된 자극.")
    ap.add_argument("--gap-steps", type=int, default=200,
                    help="자극 사이 무자극 구간. 전역 억제가 가라앉을 시간을 준다.")
    ap.add_argument("--probe-reverse", action="store_true",
                    help="B를 먼저 재서 순서 효과를 확인한다.")
    ap.add_argument("--n-stim", type=int, default=2, choices=(2, 4),
                    help="E107: 자극 수. 4면 C/D 추가(A/B 는 2자극과 동일하게 뽑힘)")
    ap.add_argument("--phase2-trials", type=int, default=0,
                    help="E107: 1단계(A/B, --trials) 뒤 C/D 만 제시하는 2단계 시행 수. 0이면 이전 동작")
    ap.add_argument("--phase2-rule", default="same", choices=("same", "cross"),
                    help="E107: 2단계 규칙. same = C→L, D→R / cross = C→R, D→L")
    ap.add_argument("--flip-at", type=int, default=None,
                    help="E104: reversal 모드에서 규칙을 뒤집는 시행 번호(기본: 전체의 절반 — 이전 동작)")
    ap.add_argument("--trace-file", type=str, default=None,
                    help="E102: 시행별 (자극·행동·보상, 도파민 직전 자격흔적, 도파민 후 가중치)를 KC별 합으로 .npz 저장. 읽기 전용")
    ap.add_argument("--probe-kc", action="store_true",
                    help="학습 전에 **KC가 A와 B를 분리하는지** 먼저 잰다. 분리가 없으면 "
                         "스파스 확장이 아무 일도 하지 않으므로 학습 실험 자체가 무의미하다.")
    ap.add_argument("--mode", default="learn",
                    choices=["learn", "noreward", "frozen", "yoked", "reversal", "supervised",
                             "shuffled"],
                    help="learn=정상 / noreward=도파민 자체를 안 줌 / "
                         "frozen=학습률 0(배선만의 성능) / "
                         "yoked=행동무관 동일빈도 보상(수반성 대조) / reversal=중간에 규칙 반전")
    ap.add_argument("--yoked-rate", type=float, default=None,
                    help="yoked 모드의 보상 빈도. 지정하지 않으면 learn 조건의 실측 정답률을 넣어야 한다.")
    ap.add_argument("--exemplars", type=int, default=0,
                    help="E122: >0 이면 원형 A/B 대신 범주당 이 수의 변형 사례로 훈련한다(원형은 훈련에 안 나옴)")
    ap.add_argument("--distort-train", type=float, default=0.2, help="E122: 훈련 사례 왜곡(교체 비트 비율)")
    ap.add_argument("--distort-test", type=str, default="0.1,0.2,0.3,0.4", help="E122: 평가 사례 왜곡 수준(쉼표)")
    ap.add_argument("--n-test-ex", type=int, default=20, help="E122: 수준·범주당 평가 사례 수(훈련과 겹치지 않음)")
    ap.add_argument("--samediff", action="store_true",
                    help="E124: 같음/다름 관계 과제(같은 항목 쌍→L, 다른 쌍→R). 평가는 처음 보는 항목으로")
    ap.add_argument("--sd-items", type=int, default=8, help="E124: 항목 수(앞 sd-train-items 개만 훈련)")
    ap.add_argument("--sd-train-items", type=int, default=4)
    ap.add_argument("--sd-frac", type=float, default=0.3, help="E124: 반쪽 안 항목 코드 활성 비율")
    ap.add_argument("--probe-sd", action="store_true",
                    help="E124 경로 검사: 학습 전 같음 쌍 S_i 와 반쪽 단독 H1_i·H2_i 의 KC 집합 — 결합(AND) KC 비율 측정 후 종료")
    ap.add_argument("--probe-ex", action="store_true",
                    help="E122 경로 검사: 학습 전 원형·사례의 KC 집합 자카드만 재고 종료")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    rng = random.Random(a.seed if a.trial_seed is None else a.trial_seed)

    global PATTERNS
    PATTERNS = make_patterns(a)
    EX = None
    SD = None
    if a.samediff:
        if a.exemplars > 0 or a.mode not in ("learn", "frozen", "noreward") or a.n_stim != 2 or a.phase2_trials:
            raise SystemExit("--samediff 는 learn/frozen/noreward, 2자극, 1단계, 사례 모드 없이만 지원한다")
        if a.n_sens % 2:
            raise SystemExit("--samediff 는 n_sens 가 짝수여야 한다")
        SD, _sdp = make_samediff(a)
        for key, (q, r) in _sdp.items():
            PATTERNS[key] = q
            RULE[key] = r
        _h = a.n_sens // 2
        _ov = [len(set(SD[i].tolist()) & set(SD[j].tolist())) for i in range(a.sd_items) for j in range(i + 1, a.sd_items)]
        _nv = [len(set(SD[i].tolist()) & set(SD[j].tolist())) for i in range(a.sd_train_items) for j in range(a.sd_train_items, a.sd_items)]
        print("[관계] 항목 %d개(훈련 %d, 새 %d), 반쪽 %d 중 활성 %d, 항목 간 겹침 %d~%d(평균 %.1f), 새↔훈련 겹침 %d~%d, 패턴 %d개(S %d, D %d)"
              % (a.sd_items, a.sd_train_items, a.sd_items - a.sd_train_items, _h, len(SD[0]), min(_ov), max(_ov), np.mean(_ov),
                 min(_nv), max(_nv), len(_sdp), sum(k.startswith("S") for k in _sdp), sum(k.startswith("D") for k in _sdp)))
    if a.exemplars > 0:
        if a.mode not in ("learn", "frozen", "noreward", "shuffled") or a.n_stim != 2 or a.phase2_trials:
            raise SystemExit("--exemplars 는 learn/frozen/noreward/shuffled, 2자극, 1단계만 지원한다")
        EX = make_exemplars(a)
        for cat in ("A", "B"):
            for i, q in enumerate(EX[cat]["train"]):
                register_exemplar("%s#%d" % (cat, i), q, cat)
            for d, lst in EX[cat]["test"].items():
                for i, q in enumerate(lst):
                    register_exemplar("%s@%.2f#%d" % (cat, d, i), q, cat)
        # 경로 검사 출력(P19: 정의 포함) — 교체 비트 수·원형과의 겹침·다른 범주 원형과의 겹침
        for cat, oth in (("A", "B"), ("B", "A")):
            P, O = PATTERNS[cat], PATTERNS[oth]
            def _st(lst):
                ov = [int((q * P).sum()) for q in lst]; ob = [int((q * O).sum()) for q in lst]; n1 = [int(q.sum()) for q in lst]
                return "활성 %d~%d, 원형겹침 %d~%d, 타원형겹침 %d~%d" % (min(n1), max(n1), min(ov), max(ov), min(ob), max(ob))
            print("[사례] %s 훈련 %d개(d=%.2f): %s" % (cat, a.exemplars, a.distort_train, _st(EX[cat]["train"])))
            for d, lst in EX[cat]["test"].items():
                print("[사례] %s 평가 d=%.2f %d개: %s" % (cat, d, len(lst), _st(lst)))
        _all = [tuple(np.flatnonzero(q).tolist()) for c in ("A", "B") for q in EX[c]["train"] + sum(EX[c]["test"].values(), [])]
        _pro = {tuple(np.flatnonzero(PATTERNS[c]).tolist()) for c in ("A", "B")}
        print("[사례] 전체 %d개, 서로 다른 집합 %d개, 원형과 같은 것 %d개(0이어야 함)"
              % (len(_all), len(set(_all)), sum(x in _pro for x in _all)))
    if a.phase2_trials > 0:
        if a.n_stim != 4:
            raise SystemExit("--phase2-trials 는 --n-stim 4 가 필요하다")
        if a.mode not in ("learn", "frozen", "noreward"):
            raise SystemExit("2단계는 learn/frozen/noreward 모드만 지원한다(reversal·shuffled 미지원)")
    if a.n_stim == 4:
        RULE["C"], RULE["D"] = ("L", "R") if a.phase2_rule == "same" else ("R", "L")
    # shuffled 용 보상 계열: learn 과 **같은 난수열**로 만든 뒤 섞는다.
    # 실제 learn 의 보상률을 미리 알 수 없으므로, 같은 시드의 rng 로 뽑은 행동열로
    # 기대 보상률을 추정하는 대신 **직전 E090/E091 실측 보상률(약 50%)**을 쓰지 않는다.
    # 대신 learn 을 먼저 돌려 얻은 보상 계열을 파일로 받는다(--reward-file).
    SHUFFLED = []
    if a.mode == "shuffled":
        if not a.reward_file or not os.path.exists(a.reward_file):
            raise SystemExit("shuffled 모드는 --reward-file 이 필요하다 "
                             "(같은 시드·난수열의 learn 이 남긴 보상 계열)")
        seq = [x.strip() == "1" for x in io.open(a.reward_file, encoding="utf-8") if x.strip()]
        if len(seq) < a.trials:
            raise SystemExit("보상 계열이 짧다: %d < %d" % (len(seq), a.trials))
        seq = seq[:a.trials]
        shuf = random.Random(7000 + (a.trial_seed if a.trial_seed is not None else a.seed))
        shuf.shuffle(seq)
        SHUFFLED = seq

    m, pops, syn = build(a)

    if a.probe_kc:
        # 전제 확인: A와 B가 서로 다른 KC 집합을 켜는가.
        # 버섯체의 스파스 확장은 **자극마다 다른 소수 집합**이 켜져야 의미가 있다.
        sets = {}
        print("  패턴 겹침: A와 B가 공유하는 감각 뉴런 %d/%d"
              % (int((PATTERNS["A"] * PATTERNS["B"]).sum()), int(PATTERNS["A"].sum())))
        # 자극 사이에 **무자극 구간**을 둔다. 전역 억제(tau 10)가 남아 있으면
        # 뒤에 잰 자극이 눌려 "분리"처럼 보인다 — 1차 측정에서 A=98 / B=8 로 나왔다.
        # 순서를 바꿔서도 재서 순서 효과를 확인한다.
        order = ("B", "A") if a.probe_reverse else ("A", "B")
        for stim in order:
            apply_stim(pops, a, None)
            for _ in range(a.gap_steps):
                m.step_time()
            m.pull_recording_buffers_from_device()
            apply_stim(pops, a, stim)
            set_dopamine(syn, 0.0)
            for _ in range(a.steps):
                m.step_time()
            m.pull_recording_buffers_from_device()
            ids = np.asarray(pops["kc"].spike_recording_data[0][1], dtype=int)
            sets[stim] = set(ids.tolist())
            print("  자극 %s: KC 발화 %d/%d (%.1f%%), 출력 L=%d R=%d"
                  % (stim, len(sets[stim]), a.n_kc, len(sets[stim]) / a.n_kc * 100,
                     len(pops["out_l"].spike_recording_data[0][1]),
                     len(pops["out_r"].spike_recording_data[0][1])))
        both = sets["A"] & sets["B"]
        union = sets["A"] | sets["B"]
        jac = len(both) / len(union) * 100 if union else float("nan")
        print("  겹침 %d개 | **자카드 %.1f%%** (0%%=완전분리, 100%%=구분없음)" % (len(both), jac))
        print("=> KCSEP seed=%d a=%d b=%d both=%d jaccard=%.1f"
              % (a.seed, len(sets["A"]), len(sets["B"]), len(both), jac))
        return

    def present(key):
        """무자극 gap → 자극 steps. 도파민 0(학습 없음). (out_L 스파이크, out_R 스파이크, KC 집합)"""
        apply_stim(pops, a, None)
        set_dopamine(syn, 0.0)
        for _ in range(a.gap_steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        apply_stim(pops, a, key)
        for _ in range(a.steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        return (len(pops["out_l"].spike_recording_data[0][1]), len(pops["out_r"].spike_recording_data[0][1]),
                set(np.asarray(pops["kc"].spike_recording_data[0][1], dtype=int).tolist()))

    if a.probe_sd:
        if SD is None:
            raise SystemExit("--probe-sd 는 --samediff 가 필요하다")
        for key, q in samediff_halves(a, SD).items():
            PATTERNS[key] = q
        rows = []
        for i in range(a.sd_train_items):
            kS = present("S%d" % i)[2]; k1 = present("H1_%d" % i)[2]; k2 = present("H2_%d" % i)[2]
            j = (i + 1) % a.sd_train_items
            kD = present("D%d_%d" % (i, j))[2]
            union = k1 | k2
            conj = kS - union                    # 양쪽이 함께일 때만 켜지는 KC(결합)
            rows.append((len(kS), len(k1), len(k2), len(conj), len(kS & union),
                         len(kS & kD) / len(kS | kD) * 100 if (kS | kD) else float("nan")))
        r = np.array(rows, dtype=float)
        print("=> SDKC seed=%d sens_kc_p=%.3f sens_kc_w=%.2f | KC(S) %.1f KC(H1) %.1f KC(H2) %.1f | 결합전용 %.1f (%.1f%% of KC(S)) | S∩(H1∪H2) %.1f | jaccard(S_i,D_i,i+1) %.1f"
              % (a.seed, a.sens_kc_p, a.sens_kc_w, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(),
                 100 * r[:, 3].sum() / max(r[:, 0].sum(), 1), r[:, 4].mean(), r[:, 5].mean()))
        apply_stim(pops, a, None)
        return

    if a.probe_ex:
        if EX is None:
            raise SystemExit("--probe-ex 는 --exemplars 가 필요하다")
        def jac(x, y):
            return len(x & y) / len(x | y) * 100 if (x | y) else float("nan")
        kA, kB = present("A")[2], present("B")[2]
        print("=> EXKC seed=%d proto_A_kc=%d proto_B_kc=%d jaccard_AB=%.1f" % (a.seed, len(kA), len(kB), jac(kA, kB)))
        for d in exemplar_levels(a):
            jA, jB = [], []
            for i in range(min(10, a.n_test_ex)):
                ks = present("A@%.2f#%d" % (d, i))[2]
                jA.append(jac(ks, kA)); jB.append(jac(ks, kB))
            print("=> EXKC d=%.2f A사례(10): jaccard_to_protoA 평균 %.1f | jaccard_to_protoB 평균 %.1f" % (d, np.mean(jA), np.mean(jB)))
        apply_stim(pops, a, None)
        return

    w0 = {k: read_g(s).copy() for k, s in syn.items()}
    # E104: 반전 시점. 기본은 이전과 같이 전체 시행의 절반(동작 불변).
    FLIP_AT = a.flip_at if a.flip_at is not None else a.trials // 2
    if a.trace_file:
        global TRACE
        TRACE = {"n_pre": a.n_kc, "pre": {}, "e_l": [], "e_r": [], "g_l": [], "g_r": [], "ev": [], "act_spk": []}
        for k, sy in syn.items():
            sy.pull_connectivity_from_device()
            TRACE["pre"][id(sy)] = np.asarray(sy.get_sparse_pre_inds(), dtype=np.int64)
        TRACE["g0_l"] = _pre_sums(syn["l"], "g", a.n_kc)
        TRACE["g0_r"] = _pre_sums(syn["r"], "g", a.n_kc)

    EX_RNG = random.Random(3000 + (a.seed if a.trial_seed is None else a.trial_seed))
    hist = []
    ok = 0
    n_rewarded = 0
    REWARD_LOG = []
    for t in range(a.trials + a.phase2_trials):
        rule = FLIP if (a.mode == "reversal" and t >= FLIP_AT) else RULE
        if t < a.trials:
            stim = "A" if rng.random() < 0.5 else "B"
            if EX is not None:
                # E122: 범주 순서는 K50과 같은 rng, 사례 선택은 별도 난수열(시행 난수열 불변)
                stim = "%s#%d" % (stim, EX_RNG.randrange(a.exemplars))
            elif SD is not None:
                # E124: A → 같음, B → 다름(범주 순서는 K50과 같은 rng). 항목은 별도 난수열, 훈련 항목만.
                _i = EX_RNG.randrange(a.sd_train_items)
                if stim == "A":
                    stim = "S%d" % _i
                else:
                    _j = EX_RNG.randrange(a.sd_train_items - 1)
                    stim = "D%d_%d" % (_i, _j if _j < _i else _j + 1)
        else:
            stim = "C" if rng.random() < 0.5 else "D"   # E107 2단계: 난수 소비는 1단계와 같은 방식

        if a.mode == "shuffled":
            # E092: 보상의 **총량·시계열을 그대로 두고 수반성만 제거**한다.
            # learn 과 같은 난수열로 미리 뽑아둔 보상 계열을 시행에 무작위 재배정한다.
            # yoked(고정 확률)와 다르다 — 보상률이 learn 과 같아야 조작이 유효하다.
            def rf(s_, act_, _t=t):
                return SHUFFLED[_t]
        elif a.mode == "supervised":
            # E087: 탐색·선택 피드백을 **전부 제거**한다. 행동을 강제하되 정답/오답을 반반 섞고,
            # 실제 정답 여부로 보상/벌을 준다. 남는 질문은 하나다 —
            # 보상+동시활동이 **올바른 시냅스를 강화하는가**.
            def rf(s_, act_, _rule=rule):
                return _rule[s_] == act_
        elif a.mode in ("noreward", "frozen"):
            def rf(s_, act_):
                return False
        elif a.mode == "yoked":
            rate = a.yoked_rate if a.yoked_rate is not None else 0.5

            def rf(s_, act_, _r=rate):
                return rng.random() < _r
        else:
            def rf(s_, act_, _rule=rule):
                return _rule[s_] == act_

        act, nl, nr = run_trial(m, pops, syn, stim, a, rng, rf)
        if rf(stim, act) if a.mode in ("learn", "reversal") else False:
            pass
        _got = rf(stim, act)
        n_rewarded += 1 if _got else 0
        REWARD_LOG.append(1 if _got else 0)
        ok += 1 if rule[stim] == act else 0

        if (t + 1) % a.block == 0:
            acc = ok / a.block * 100.0
            hist.append(acc)
            print("  시행 %4d~%4d: 정답률 %5.1f%%" % (t + 2 - a.block, t + 1, acc))
            ok = 0

    # ★평가 단계: 탐색 없이(greedy), 학습 없이 정답률을 잰다.
    # 2026-09-20 사고: 정답률을 **탐색하는 중에** 재고 있었다. 학습에는 탐색이 필요한데
    # epsilon=1.0 이면 행동이 100% 무작위라 학습된 매핑이 행동으로 나올 수가 없다.
    # 실측: epsilon=1.0 seed2 에서 A→out_L 296 vs 42, B→out_R 199 vs 47 로 **매핑은 학습됐는데**
    # 정답률은 51.5%였다. 훈련과 평가를 분리한다(전체 모델은 원래 분리돼 있다).
    eval_ok = 0
    eval_orig = 0
    eval_tie = 0
    eval_rng = random.Random(9000 + a.seed)
    eval_by = {"A": [0, 0], "B": [0, 0]}   # [정답, 전체]
    for i in range(a.eval_trials):
        # 자극 순서를 **무작위**로. 번갈아 제시하면 앞 시행의 이월과 완벽히 교락된다.
        # 2026-09-20 실측: 번갈아 제시에서 정답률이 정확히 0% 또는 50%만 나왔다.
        # 0%는 출력이 **이전 자극**의 답을 내고 있다는 뜻이다(A시행에 R, B시행에 L).
        # 무작위 순서면 이월이 남아 있어도 체계적 오답이 되지 않아 그 크기를 볼 수 있다.
        stim = "A" if eval_rng.random() < 0.5 else "B"
        apply_stim(pops, a, None)
        set_dopamine(syn, 0.0)
        for _ in range(a.gap_steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        apply_stim(pops, a, stim)
        for _ in range(a.steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        _nl = len(pops["out_l"].spike_recording_data[0][1])
        _nr = len(pops["out_r"].spike_recording_data[0][1])
        # 동점은 **무응답이 아니라 찍기**다. 오답으로 세면 frozen 정답률이 0%로 나와
        # "배선이 완벽히 반대"처럼 보인다(2026-09-20 실측 eval=0.0%). 훈련 때와 같이 무작위로 고른다.
        if _nl > _nr:
            act = "L"
        elif _nr > _nl:
            act = "R"
        else:
            eval_tie += 1
            act = "L" if rng.random() < 0.5 else "R"
        # 2026-09-25 수정(외부 검토): reversal 모드는 훈련 후반에 FLIP 을 쓰는데
        # 평가는 항상 RULE 로 채점했다 → **역전을 학습해도 옛 정답으로 점수를 매긴다.**
        # 최종 활성 규칙과 원래 규칙 둘 다 집계한다.
        _final = FLIP if a.mode == "reversal" else RULE
        eval_by[stim][1] += 1
        if _final[stim] == act:
            eval_ok += 1
            eval_by[stim][0] += 1
        if RULE[stim] == act:
            eval_orig += 1
    eval_acc = eval_ok / a.eval_trials * 100.0
    apply_stim(pops, a, None)
    if SD is not None:
        # E124: 관계 평가 — 훈련 항목 쌍과 처음 보는 항목 쌍. 같음/다름 반반, 균형 정답률 = (같음 정답률 + 다름 정답률)/2.
        _xr = random.Random(9300 + a.seed)
        res = {}
        for setname, lo, hi in (("train", 0, a.sd_train_items), ("novel", a.sd_train_items, a.sd_items)):
            c = {"S": [0, 0], "D": [0, 0]}; _tie = 0
            for i in range(a.eval_trials):
                same = _xr.random() < 0.5
                _i = _xr.randrange(lo, hi)
                if same:
                    key = "S%d" % _i
                else:
                    _j = _xr.randrange(lo, hi - 1)
                    _j = _j if _j < _i else _j + 1
                    key = "D%d_%d" % (_i, _j)
                _nl, _nr, _ = present(key)
                if _nl == _nr:
                    _tie += 1
                act = "L" if _nl > _nr else ("R" if _nr > _nl else ("L" if _xr.random() < 0.5 else "R"))
                kk = "S" if same else "D"
                c[kk][1] += 1
                c[kk][0] += 1 if RULE[key] == act else 0
            sa = c["S"][0] / c["S"][1] * 100 if c["S"][1] else float("nan")
            da = c["D"][0] / c["D"][1] * 100 if c["D"][1] else float("nan")
            res[setname] = (sa, da, (sa + da) / 2, _tie, c["S"][1], c["D"][1])
        apply_stim(pops, a, None)
        print("=> SDGEN mode=%s seed=%d trialseed=%s train_same=%.1f train_diff=%.1f train_bal=%.1f novel_same=%.1f novel_diff=%.1f novel_bal=%.1f | ties train:%d novel:%d | n_same/diff train %d/%d novel %d/%d"
              % (a.mode, a.seed, a.seed if a.trial_seed is None else a.trial_seed,
                 res["train"][0], res["train"][1], res["train"][2], res["novel"][0], res["novel"][1], res["novel"][2],
                 res["train"][3], res["novel"][3], res["train"][4], res["train"][5], res["novel"][4], res["novel"][5]))
    if EX is not None:
        # E122: 미학습 사례 평가 — 원형 평가(위, 원형은 훈련에 안 나옴) 뒤, 별도 난수열. 탐색·학습 없음.
        _xr = random.Random(9200 + a.seed)
        res = {}
        for lv in ["train"] + exemplar_levels(a):
            _ok = 0; _tie = 0
            for i in range(a.eval_trials):
                cat = "A" if _xr.random() < 0.5 else "B"
                if lv == "train":
                    key = "%s#%d" % (cat, _xr.randrange(a.exemplars))
                else:
                    key = "%s@%.2f#%d" % (cat, lv, _xr.randrange(a.n_test_ex))
                _nl, _nr, _ = present(key)
                if _nl == _nr:
                    _tie += 1
                act = "L" if _nl > _nr else ("R" if _nr > _nl else ("L" if _xr.random() < 0.5 else "R"))
                _ok += 1 if RULE[cat] == act else 0
            res[lv] = (_ok / a.eval_trials * 100.0, _tie)
        apply_stim(pops, a, None)
        print("=> EXGEN mode=%s seed=%d trialseed=%s proto=%.1f train=%.1f %s | ties %s"
              % (a.mode, a.seed, a.seed if a.trial_seed is None else a.trial_seed, eval_acc, res["train"][0],
                 " ".join("d%.2f=%.1f" % (lv, res[lv][0]) for lv in exemplar_levels(a)),
                 " ".join("%s:%d" % (("train" if lv == "train" else "d%.2f" % lv), res[lv][1]) for lv in res)))
    eval_cd = None
    if a.n_stim == 4:
        # E107: C/D 평가 — A/B 평가가 끝난 뒤, 별도 난수열(A/B 결과 불변). 학습 없음·탐색 없음.
        _r2 = random.Random(9100 + a.seed); _ok2 = 0
        for i in range(a.eval_trials):
            stim = "C" if _r2.random() < 0.5 else "D"
            apply_stim(pops, a, None); set_dopamine(syn, 0.0)
            for _ in range(a.gap_steps):
                m.step_time()
            m.pull_recording_buffers_from_device()
            apply_stim(pops, a, stim)
            for _ in range(a.steps):
                m.step_time()
            m.pull_recording_buffers_from_device()
            _nl = len(pops["out_l"].spike_recording_data[0][1]); _nr = len(pops["out_r"].spike_recording_data[0][1])
            act = "L" if _nl > _nr else ("R" if _nr > _nl else ("L" if _r2.random() < 0.5 else "R"))
            _ok2 += 1 if RULE[stim] == act else 0
        eval_cd = _ok2 / a.eval_trials * 100.0
        apply_stim(pops, a, None)
        print("=== 평가 C/D (규칙 %s) : 정답률 %.1f%% ===" % (a.phase2_rule, eval_cd))
    print("")
    print("=== 평가 (탐색 없음, 학습 없음, 무작위 순서 %d시행) : 정답률 %.1f%% | 동점 %d회 ==="
          % (a.eval_trials, eval_acc, eval_tie))
    if a.mode == "reversal":
        print("   (reversal) 최종 규칙 기준 %.1f%% | 원래 규칙 기준 %.1f%%"
              % (eval_acc, eval_orig / a.eval_trials * 100.0))
    for _k in ("A", "B"):
        _c, _n = eval_by[_k]
        print("   자극 %s: %d/%d (%.1f%%)" % (_k, _c, _n, (_c / _n * 100) if _n else float("nan")))

    # 학습 후 출력이 **자극에 따라 갈리는가**. 정답률만 보면 "왜 안 되는지"를 못 본다.
    # 2026-09-20 관측: seed2가 배선 14% → 학습 52%. 편향은 지웠는데 50%를 못 넘었다.
    # 학습이 매핑을 만들지 못하고 가중치를 대칭으로 밀기만 하는지 여기서 가른다.
    print("")
    print("=== 학습 후 자극별 출력 (선택 이전의 원 신호) ===")
    for stim in ("A", "B"):
        apply_stim(pops, a, None)
        for _ in range(a.gap_steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        apply_stim(pops, a, stim)
        set_dopamine(syn, 0.0)
        for _ in range(a.steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        _nl = len(pops["out_l"].spike_recording_data[0][1])
        _nr = len(pops["out_r"].spike_recording_data[0][1])
        _kc = len(set(np.asarray(pops["kc"].spike_recording_data[0][1], dtype=int).tolist()))
        print("  자극 %s → 정답 %s | out_L=%4d  out_R=%4d  차이 %+5d | KC %d개"
              % (stim, RULE[stim], _nl, _nr, _nl - _nr, _kc))
    apply_stim(pops, a, None)

    # ★가중치 구조: **A에 반응하는 KC**의 out_L 가중치가 out_R 가중치보다 커졌는가.
    # 정답률만 보면 "왜 방향이 안 맞는지"를 못 본다(K27).
    kc_sets = {}
    for stim in ("A", "B"):
        apply_stim(pops, a, None)
        set_dopamine(syn, 0.0)
        for _ in range(a.gap_steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        apply_stim(pops, a, stim)
        for _ in range(a.steps):
            m.step_time()
        m.pull_recording_buffers_from_device()
        kc_sets[stim] = set(np.asarray(pops["kc"].spike_recording_data[0][1], dtype=int).tolist())
    apply_stim(pops, a, None)
    only_a = kc_sets["A"] - kc_sets["B"]
    only_b = kc_sets["B"] - kc_sets["A"]
    print("")
    print("=== 가중치 구조 (A전용 KC %d개, B전용 KC %d개) ===" % (len(only_a), len(only_b)))
    pre = {}
    for k, sy in syn.items():
        sy.pull_connectivity_from_device()
        pre[k] = np.asarray(sy.get_sparse_pre_inds())
    for label, kcset, want in (("A전용", only_a, "l"), ("B전용", only_b, "r")):
        if not kcset:
            print("  %s KC 없음" % label); continue
        idx = np.fromiter(kcset, dtype=int)
        wl = read_g(syn["l"])[np.isin(pre["l"], idx)]
        wr = read_g(syn["r"])[np.isin(pre["r"], idx)]
        mark = "**정답쪽 우세**" if ((wl.mean() > wr.mean()) == (want == "l")) else "**반대**"
        print("  %s KC → out_L 평균 %.4f | out_R 평균 %.4f | 차이 %+.4f (정답은 out_%s) %s"
              % (label, wl.mean(), wr.mean(), wl.mean() - wr.mean(), want.upper(), mark))

    if TRACE is not None:
        np.savez_compressed(
            a.trace_file, g0_l=TRACE["g0_l"], g0_r=TRACE["g0_r"],
            e_l=np.array(TRACE["e_l"]), e_r=np.array(TRACE["e_r"]),
            g_l=np.array(TRACE["g_l"]), g_r=np.array(TRACE["g_r"]),
            stim=np.array([x[0] for x in TRACE["ev"]]), act=np.array([x[1] for x in TRACE["ev"]]),
            reward=np.array([x[2] for x in TRACE["ev"]]), nl=np.array([x[3] for x in TRACE["ev"]]),
            nr=np.array([x[4] for x in TRACE["ev"]]),
            act_spk=np.array(TRACE["act_spk"]) if TRACE["act_spk"] else np.zeros((0, 2)),
            kc_a=np.fromiter(kc_sets["A"], dtype=np.int64), kc_b=np.fromiter(kc_sets["B"], dtype=np.int64),
            flip_at=(FLIP_AT if a.mode == "reversal" else -1))
        print("[추적] %d시행 저장 → %s" % (len(TRACE["ev"]), a.trace_file))

    if a.dump_rewards:
        _txt = chr(10).join(str(x) for x in REWARD_LOG) + chr(10)
        io.open(a.dump_rewards, "w", encoding="utf-8").write(_txt)

    w1 = {k: read_g(s) for k, s in syn.items()}
    print("\n=== 가중치 변화 ===")
    for k in sorted(syn):
        d = np.abs(w1[k] - w0[k])
        print("  kc_out_%s: n=%d |Δ|평균 %.5f 변화율 %5.1f%% | 평균 %.3f→%.3f | std %.4f→%.4f"
              % (k, d.size, d.mean(), (d > 1e-9).mean() * 100,
                 w0[k].mean(), w1[k].mean(), w0[k].std(), w1[k].std()))

    first, last = (hist[0], hist[-1]) if hist else (float("nan"), float("nan"))
    print("\n=== 요약 (mode=%s seed=%d) ===" % (a.mode, a.seed))
    print("  첫 구간 %.1f%% → 마지막 구간 %.1f%% | 변화 %+.1f%%p | 보상률 %.1f%%"
          % (first, last, last - first, n_rewarded / (a.trials + a.phase2_trials) * 100.0))
    print("=> MINCIRC mode=%s seed=%d first=%.1f last=%.1f delta=%+.1f reward=%.1f **eval=%.1f** tie=%d trialseed=%s"
          % (a.mode, a.seed, first, last, last - first,
             n_rewarded / (a.trials + a.phase2_trials) * 100.0, eval_acc, eval_tie,
             a.seed if a.trial_seed is None else a.trial_seed)
          + ("" if eval_cd is None else " evalCD=%.1f phase2=%d rule2=%s" % (eval_cd, a.phase2_trials, a.phase2_rule)))


if __name__ == "__main__":
    main()
