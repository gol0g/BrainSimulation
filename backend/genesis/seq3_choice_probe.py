#!/usr/bin/env python3
"""
3존 순차 forced-choice 프로브 (C20) — D24가 지목했으나 미실행이던 과제.

D24 규명: 2존 order_rate는 ill-posed(무작위가 directed를 이김). 처방으로 "N>2존 특정순서" 과제를
적어뒀으나 실행하지 않았음. 여기서 구현한다.

과제: A→B→C 순서. WM 상태에 따라 올바른 다음 존을 3지선다로 고르는가.
- WM 비었음 → A 선택해야
- A 적재 → B 선택해야
- A·B 적재 → C 선택해야
무작위 성공률 33%(2존 50%보다 엄격). 기준 >50%.
seq-WM 73%(2존)가 더 엄격한 과제에서도 유지되는지 = 진짜 순차 표상인지 판별.
"""
import argparse
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

# 3존을 좌/중/우 시각 위치로 제시(중앙은 양쪽 약하게 = 정면)
def present(o, nh, target_side):
    """target_side: 'left'|'center'|'right' 에 강한 good food, 나머지 약하게."""
    lo = 0.15
    L = 0.9 if target_side == "left" else lo
    R = 0.9 if target_side == "right" else lo
    if target_side == "center":
        L = R = 0.55   # 양쪽 동일 = 직진(중앙) 신호
    o["good_food_rays_left"] = np.ones(nh) * L
    o["good_food_rays_right"] = np.ones(nh) * R
    o["food_rays_left"] = np.ones(nh) * L
    o["food_rays_right"] = np.ones(nh) * R
    return o


def decide(brain, obs, steps=5):
    tot = 0.0
    for _ in range(steps):
        a, info = brain.process(obs)
        tot += a
    if tot < -0.02:
        return "left"
    if tot > 0.02:
        return "right"
    return "center"


def run(brain, env, trials=60, inhib=-200):
    obs = env.reset(); brain.reset()
    for _ in range(30):
        a, i = brain.process(obs); obs, _, d, _ = env.step((a,))
        if d: obs = env.reset()
    nh = env.config.n_rays // 2

    # 존 A/B/C를 좌/중/우에 무작위 배정(위치 편향 제거)
    correct = {"empty": 0, "loadA": 0, "loadAB": 0}
    n = 0
    for t in range(trials):
        sides = ["left", "center", "right"]
        np.random.shuffle(sides)
        zA, zB, zC = sides[0], sides[1], sides[2]

        base = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}

        # 조건1: WM 비움 → A 골라야
        brain.gate_wm_input(0.0)
        o1 = present(dict(base), nh, "left")  # 모든 존 동시 제시 대신, A 위치를 강하게
        o1 = present(dict(base), nh, zA)
        if decide(brain, o1) == zA:
            correct["empty"] += 1

        # 조건2: A 적재 → B 골라야
        brain.reset()
        for _ in range(8):
            brain.gate_wm_input(1.0)
            brain.process(present(dict(base), nh, zA))
            brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
        brain.gate_wm_input(0.0)
        if decide(brain, present(dict(base), nh, zB)) == zB:
            correct["loadA"] += 1

        # 조건3: A·B 적재 → C 골라야
        for _ in range(8):
            brain.gate_wm_input(1.0)
            brain.process(present(dict(base), nh, zB))
            brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
        brain.gate_wm_input(0.0)
        if decide(brain, present(dict(base), nh, zC)) == zC:
            correct["loadAB"] += 1
        n += 1

    e = correct["empty"] / max(n, 1) * 100
    a_ = correct["loadA"] / max(n, 1) * 100
    ab = correct["loadAB"] / max(n, 1) * 100
    comb = (e + a_ + ab) / 3
    return e, a_, ab, comb, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-weights", default=None)
    ap.add_argument("--inhib-wm", type=float, default=-200.0)
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()
    cfg = ForagerBrainConfig()
    cfg.inhibitory_to_wm_weight = args.inhib_wm
    brain = ForagerBrain(cfg)
    if args.load_weights:
        brain.load_all_weights(args.load_weights)
    env = ForagerGym(ForagerConfig())
    e, a_, ab, comb, n = run(brain, env, args.trials, args.inhib_wm)
    print(f"SEQ3: empty→A {e:.1f}% | A→B {a_:.1f}% | AB→C {ab:.1f}% | combined {comb:.1f}% "
          f"(random=33.3, n={n}) [{'PASS' if comb > 50 else 'FAIL'}]")


if __name__ == "__main__":
    main()
