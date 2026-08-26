#!/usr/bin/env python3
"""
WM 게이트 회로 프로브 (C21) — DESIGN_RECOVERY 333행의 "미구현" 항목.

기록: "순서 기억 = 입력 게이팅 필요. 4월 코드에 기계 존재(wm_gate, dopamine_to_wm_gate_weight,
wm_update_gate). 러너가 이걸 engage 안 함. → dopamine-gated WM 쓰기가 seq-wm의 진짜 재구현."
→ 나는 대신 gate_wm_input(시냅스 가중치 직접 스케일)이라는 우회를 만들어 썼고, 원 회로는 검증 안 함.

여기서 측정: dopamine → wm_update_gate → wm_thalamic → working_memory 경로가 실제로 작동하나.
- 도파민 방출 시 wm_update_gate 발화가 오르나(게이트 열림)
- 그때 wm_thalamic → WM 쓰기가 일어나 A-패턴이 유지되나
작동하면 = 우회 없이 뇌 자체 게이팅으로 seq-wm 재구현 가능. 미작동이면 = 회로는 있으나 죽어 있음.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

def rate(b, name):
    p = getattr(b, name, None)
    if p is None:
        return None
    try:
        return len(p.spike_recording_data[0][0])
    except Exception:
        return None

cfg = ForagerBrainConfig()
b = ForagerBrain(cfg)
env = ForagerGym(ForagerConfig())
obs = env.reset()
for _ in range(20):
    a, i = b.process(obs); obs, _, d, _ = env.step((a,))
    if d: obs = env.reset()

POPS = ["dopamine_neurons", "wm_update_gate", "wm_thalamic", "working_memory"]

def measure(label, dopamine, steps=4):
    acc = {p: [] for p in POPS}
    for _ in range(steps):
        if dopamine:
            b.release_dopamine(reward_magnitude=1.0, primary_reward=True)
        b.process(obs)
        if not dopamine:
            b.decay_dopamine()
        for p in POPS:
            r = rate(b, p)
            if r is not None:
                acc[p].append(r)
    out = {p: (float(np.mean(v)) if v else float("nan")) for p, v in acc.items()}
    print(f"GATE {label}: " + " | ".join(f"{p}={out[p]:.0f}" for p in POPS))
    return out

no_da = measure("dopamine-OFF", False)
da = measure("dopamine-ON", True)
print("GATEDIFF: " + " | ".join(
    f"{p}={da[p]-no_da[p]:+.0f}" for p in POPS))
g = da["wm_update_gate"] - no_da["wm_update_gate"]
t = da["wm_thalamic"] - no_da["wm_thalamic"]
w = da["working_memory"] - no_da["working_memory"]
print(f"VERDICT: gate열림={'YES' if g > 20 else 'NO'} | thalamic전달={'YES' if t > 20 else 'NO'} "
      f"| WM쓰기={'YES' if w > 20 else 'NO'}")
