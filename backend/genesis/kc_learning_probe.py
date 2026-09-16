#!/usr/bin/env python3
"""KC 학습이 왜 안 일어나는지 직접 측정 (H012 검증 전 필수).

H012는 "도파민이 KC에 도달하지 않는다"고 주장했으나, 코드를 보니
`_update_rstdp_weights()` 안에 KC 학습 블록이 **이미 존재**하고 게이트 조건도 참이다.
결론 전에 어디서 막히는지 측정한다(이 프로젝트에서 조작 무효를 5회 겪었다).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

import argparse as _ap2
_p2 = _ap2.ArgumentParser(); _p2.add_argument("--kc-rstdp", action="store_true")
# INV-A4/A5는 **기본값으로 박는다**(규약 P15). 설정 기본값 0.0은 D1 포화 상태이고,
# 이 프로브가 그 상태로 E079 (a)를 측정했다 — 2026-09-16 적발.
_p2.add_argument("--d1-inhib", type=float, default=-400.0)
_p2.add_argument("--direct-inhib", type=float, default=-100.0)
_a2, _ = _p2.parse_known_args()
cfg = ForagerBrainConfig()
cfg.d1_inhibition = _a2.d1_inhib
cfg.direct_inhibition = _a2.direct_inhib
if _a2.kc_rstdp:
    cfg.kc_rstdp = True
    print("[E079] kc_rstdp=True (시냅스별 자격흔적)")
b = ForagerBrain(cfg)
env = ForagerGym(ForagerConfig())
obs = env.reset()
for _ in range(20):
    a, _ = b.process(obs)
    obs, _, d, _ = env.step((a,))
    if d: obs = env.reset()

print("=== 게이트 조건 ===")
print("  sparse_expansion_enabled =", cfg.sparse_expansion_enabled)
print("  hasattr(kc_to_d1_l) =", hasattr(b, "kc_to_d1_l"))
print("  _context_hard_gate_active =", getattr(b, "_context_hard_gate_active", "없음"))

def snap(nm):
    s = getattr(b, nm, None)
    if s is None: return None
    try:
        s.pull_connectivity_from_device(); s.vars["g"].pull_from_device()
        v = s.vars["g"].values
        if v is None or (hasattr(v,"size") and v.size==0): v = s.vars["g"].view
        return np.array(v, dtype=np.float64).copy()
    except Exception as e:
        print("  조회실패 %s: %s" % (nm, e)); return None

NAMES = ["kc_to_d1_l", "kc_to_d1_r", "food_to_d1_l"]
print("\n=== KC 발화 확인 (학습엔 pre 활동이 필요) ===")
for pop in ["kc_left", "kc_right", "d1_left"]:
    p = getattr(b, pop, None)
    if p is None: print("  %-10s 없음" % pop); continue
    try:
        n = len(p.spike_recording_data[0][0])
        print("  %-10s 스파이크 %d" % (pop, n))
    except Exception as e:
        print("  %-10s 기록없음(%s)" % (pop, type(e).__name__))

before = {n: snap(n) for n in NAMES}
print("\n=== 보상 100회 + decay_dopamine 호출 ===")
rew = 0
for _ in range(100):
    b.release_dopamine(reward_magnitude=1.0, primary_reward=True)
    b.process(obs)
    b.decay_dopamine()      # KC 학습은 이 안의 _update_rstdp_weights()에서 일어난다
    rew += 1
after = {n: snap(n) for n in NAMES}

print("\n%-14s %10s %10s %10s" % ("시냅스", "|Δ|평균", "변화율", "std(후)"))
print("-"*48)
for n in NAMES:
    if before[n] is None or after[n] is None or before[n].shape != after[n].shape:
        print("  %-12s 비교불가" % n); continue
    d = np.abs(after[n]-before[n])
    print("%-14s %10.6f %9.1f%% %10.4f" % (n, d.mean(), (d>1e-9).mean()*100, after[n].std()))

r = getattr(b, "_last_rstdp_results", None)
print("\n=== _update_rstdp_weights 반환값에 KC 항목이 있나 ===")
if isinstance(r, dict):
    kc_keys = [k for k in r if "kc" in k.lower()]
    print("  전체 %d개 키 / KC 관련: %s" % (len(r), kc_keys if kc_keys else "**없음**"))
else:
    print("  결과 없음:", type(r))
