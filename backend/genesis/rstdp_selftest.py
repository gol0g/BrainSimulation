#!/usr/bin/env python3
"""R-STDP 모델 단독 검증 (C36 self-test) — 배선 전에 모델 자체가 작동하는지 확인.

이 세션에서 반복된 실패("코드에 있다"를 "작동한다"로 취급)를 피하기 위해, 뇌에 배선하기 전에
최소 모델로 다음을 확인한다:
  1. 컴파일·실행되는가
  2. 도파민이 0이면 가중치가 **안 변하는가** (게이팅 작동)
  3. 도파민을 주면 **활동한 시냅스만** 변하는가 → **std가 0에서 증가**해야 함
     (기존 `w[:] += eta*trace` 방식은 std가 영원히 0이었다 = 신용 할당 부재의 서명)
"""
import sys, os
import numpy as np
from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_weight_update

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rstdp_model import make_rstdp_model, DEFAULT_PARAMS

model = GeNNModel("float", "rstdp_selftest")
model.dt = 1.0
model.seed = 12345

N = 100
# pre: 절반만 활동시킬 것이므로 전류 주입형 LIF
lif_p = {"C": 1.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -50.0,
         "Ioffset": 0.0, "TauRefrac": 2.0}
lif_v = {"V": -65.0, "RefracTime": 0.0}

pre = model.add_neuron_population("pre", N, "LIF", lif_p, lif_v)
post = model.add_neuron_population("post", N, "LIF", lif_p, lif_v)
pre.spike_recording_enabled = True
post.spike_recording_enabled = True

wu = make_rstdp_model()
syn = model.add_synapse_population(
    "s", "SPARSE", pre, post,
    init_weight_update(wu, DEFAULT_PARAMS,
                       {"g": 5.0, "e": 0.0}, {"preTrace": 0.0}, {"postTrace": 0.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity("FixedProbability", {"prob": 0.1}))

# dopamine을 동적 파라미터로 지정(빌드 전) — 매 스텝 파이썬에서 값을 바꿀 수 있게 된다
syn.set_wu_param_dynamic("dopamine")

model.build()
model.load(num_recording_timesteps=10)


def snap():
    syn.pull_connectivity_from_device()
    syn.vars["g"].pull_from_device()
    v = np.array(syn.vars["g"].values, dtype=np.float64)
    return v.copy()


def drive(steps, da, active_half=True):
    """pre 뉴런의 절반만 강하게 구동 → 활동한 시냅스만 신용을 받아야 한다."""
    syn.set_dynamic_param_value("dopamine", da)
    for t in range(steps):
        pre.vars["V"].pull_from_device()
        v = pre.vars["V"].view
        v[:N // 2] = -45.0                       # 앞쪽 절반만 발화 유도(임계 초과)
        if not active_half:
            v[N // 2:] = -45.0
        pre.vars["V"].push_to_device()
        model.step_time()


w0 = snap()
print("[0] 초기 g: 평균 %.4f, std %.6f, n=%d" % (w0.mean(), w0.std(), w0.size))

# (1) 도파민 0 — 변하면 안 됨
drive(300, 0.0)
w1 = snap()
print("[1] 도파민 0 후: 평균 %.4f, std %.6f | |Δ|=%.6f → %s"
      % (w1.mean(), w1.std(), np.abs(w1 - w0).mean(),
         "게이팅 OK(불변)" if np.abs(w1 - w0).mean() < 1e-6 else "**게이팅 실패(도파민 없이 변함)**"))

# (2) 도파민 1.0 — 활동한 시냅스만 변해야 하고 std가 올라야 함
drive(300, 1.0)
w2 = snap()
d = np.abs(w2 - w1)
print("[2] 도파민 1.0 후: 평균 %.4f, std %.6f | |Δ|평균=%.6f, 변화시냅스 %.1f%%"
      % (w2.mean(), w2.std(), d.mean(), (d > 1e-9).mean() * 100))
print("=> 판정: %s" % (
    "**신용 할당 작동** (std 0→%.4f, 일부 시냅스만 변화)" % w2.std()
    if (w2.std() > 1e-4 and (d > 1e-9).mean() < 0.999)
    else "실패 — std가 안 오르거나 전 시냅스 균일 변화(= 기존 전역 스칼라와 동일)"))
