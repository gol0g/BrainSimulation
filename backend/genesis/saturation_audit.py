#!/usr/bin/env python3
"""
전 집단 포화 감사 (C14) — WM·sts_social에서 두 번 발견된 포화 병리가 다른 집단에도 있나.

방법: 감각 입력 없음(rest) vs 강한 입력(food/pain/social 등) 조건에서 각 집단의 스파이크를 비교.
- rest에서 이미 높고 입력에 변화 없음 → 포화(병리, WM/sts_social과 동형)
- rest 낮고 입력에 증가 → 정상
전 집단을 훑어 숨은 포화를 찾는다. 개념이 잘 되는 회로도 포화면 더 좋아질 여지가 있음.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

b = ForagerBrain(ForagerBrainConfig())
env = ForagerGym(ForagerConfig())
obs = env.reset()
for _ in range(20):
    a, i = b.process(obs)
    obs, _, d, _ = env.step((a,))
    if d:
        obs = env.reset()

nh = env.config.n_rays // 2
def make_obs(mode):
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    for key in ("food_rays_left", "food_rays_right", "good_food_rays_left", "good_food_rays_right",
                "bad_food_rays_left", "bad_food_rays_right", "pain_rays_left", "pain_rays_right",
                "predator_rays_left", "predator_rays_right", "agent_rays_left", "agent_rays_right",
                "wall_rays_left", "wall_rays_right"):
        o[key] = np.zeros(nh)
    for key in ("food_sound_high", "food_sound_low", "sound_food_left", "sound_food_right",
                "sound_danger_left", "sound_danger_right", "danger_signal", "social_proximity",
                "npc_call_food_left", "npc_call_food_right", "agent_sound_left", "agent_sound_right"):
        o[key] = 0.0
    if mode == "strong":
        for key in ("food_rays_left", "good_food_rays_left", "pain_rays_right",
                    "predator_rays_right", "agent_rays_left"):
            o[key] = np.ones(nh) * 0.9
        for key in ("food_sound_high", "sound_food_left", "sound_danger_right",
                    "danger_signal", "social_proximity", "npc_call_food_left", "agent_sound_left"):
            o[key] = 0.9
    return o

# 감사 대상: spike recording 켜진 주요 집단
NAMES = [n for n in dir(b) if not n.startswith("_")]
pops = []
for n in NAMES:
    try:
        p = getattr(b, n)
        if hasattr(p, "spike_recording_data") and hasattr(p, "vars"):
            pops.append(n)
    except Exception:
        pass

def measure(mode, steps=4):
    o = make_obs(mode)
    acc = {}
    for _ in range(steps):
        b.process(o)
        for n in pops:
            try:
                c = len(getattr(b, n).spike_recording_data[0][0])
                acc.setdefault(n, []).append(c)
            except Exception:
                pass
    return {k: float(np.mean(v)) for k, v in acc.items()}

rest = measure("rest")
strong = measure("strong")

print("\n=== 포화 감사 (C14) ===")
print(f"{'population':>28} {'rest':>9} {'strong':>9} {'Δ':>9}  판정")
sat, ok, dead = [], [], []
for n in sorted(set(rest) & set(strong)):
    r, s = rest[n], strong[n]
    d = s - r
    if r > 100 and abs(d) < max(20.0, r * 0.05):
        verdict = "★포화(무반응)"; sat.append(n)
    elif r < 1 and s < 1:
        verdict = "무활동"; dead.append(n)
    else:
        verdict = "정상"; ok.append(n)
    if verdict != "정상":
        print(f"{n:>28} {r:>9.0f} {s:>9.0f} {d:>+9.0f}  {verdict}")
print(f"\n요약: 포화 {len(sat)}개 / 무활동 {len(dead)}개 / 정상 {len(ok)}개 (총 {len(rest)})")
if sat:
    print("포화 집단:", ", ".join(sat))
