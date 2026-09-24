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
    return {"A": pa, "B": pb}


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

    if rng.random() < args.epsilon:
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
        pops["out_l"].set_dynamic_param_value("Ioffset", 0.0)
        pops["out_r"].set_dynamic_param_value("Ioffset", 0.0)

    give = rewarded_fn(stim, act)
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
    return act, nl, nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
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
    ap.add_argument("--da", type=float, default=1.0)
    ap.add_argument("--da-neg", type=float, default=0.5,
                    help="오답 시 음의 도파민. 0이면 가중치가 올라가기만 해 한쪽이 상한으로 폭주한다.")
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--act-drive", type=float, default=6.0,
                    help="고른 행동을 해당 출력 집단에 주입해 자격흔적에 남긴다. 0이면 끔 — "
                         "끄면 탐색 행동이 흔적에 안 남아 학습이 반대로 간다(실측 87%%→16%%).")
    ap.add_argument("--act-steps", type=int, default=15)
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
    ap.add_argument("--probe-kc", action="store_true",
                    help="학습 전에 **KC가 A와 B를 분리하는지** 먼저 잰다. 분리가 없으면 "
                         "스파스 확장이 아무 일도 하지 않으므로 학습 실험 자체가 무의미하다.")
    ap.add_argument("--mode", default="learn",
                    choices=["learn", "noreward", "frozen", "yoked", "reversal"],
                    help="learn=정상 / noreward=도파민 자체를 안 줌 / "
                         "frozen=학습률 0(배선만의 성능) / "
                         "yoked=행동무관 동일빈도 보상(수반성 대조) / reversal=중간에 규칙 반전")
    ap.add_argument("--yoked-rate", type=float, default=None,
                    help="yoked 모드의 보상 빈도. 지정하지 않으면 learn 조건의 실측 정답률을 넣어야 한다.")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    rng = random.Random(a.seed)

    global PATTERNS
    PATTERNS = make_patterns(a)
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

    w0 = {k: read_g(s).copy() for k, s in syn.items()}

    hist = []
    ok = 0
    n_rewarded = 0
    for t in range(a.trials):
        rule = FLIP if (a.mode == "reversal" and t >= a.trials // 2) else RULE
        stim = "A" if rng.random() < 0.5 else "B"

        if a.mode in ("noreward", "frozen"):
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
        n_rewarded += 1 if rf(stim, act) else 0
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
        eval_by[stim][1] += 1
        if RULE[stim] == act:
            eval_ok += 1
            eval_by[stim][0] += 1
    eval_acc = eval_ok / a.eval_trials * 100.0
    apply_stim(pops, a, None)
    print("")
    print("=== 평가 (탐색 없음, 학습 없음, 무작위 순서 %d시행) : 정답률 %.1f%% | 동점 %d회 ==="
          % (a.eval_trials, eval_acc, eval_tie))
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
          % (first, last, last - first, n_rewarded / a.trials * 100.0))
    print("=> MINCIRC mode=%s seed=%d first=%.1f last=%.1f delta=%+.1f reward=%.1f **eval=%.1f** tie=%d"
          % (a.mode, a.seed, first, last, last - first,
             n_rewarded / a.trials * 100.0, eval_acc, eval_tie))


if __name__ == "__main__":
    main()
