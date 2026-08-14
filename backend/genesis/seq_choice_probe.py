#!/usr/bin/env python3
"""
순차 forced-choice 프로브 (D25) — 개념 forced-choice 방법론을 순서에 적용.

D24: order_rate/2-zone 과제가 ill-posed(무작위≥directed). 유효 검증 필요.
이 프로브: 중립 상태서 A/B를 egocentric 감각으로 동시 제시하고, WM 상태에 따라 올바른 다음 존을
선택하는가 측정. WM 비었으면(A 미방문) → A로, WM에 A 적재되면 → B로. 무작위=50%, 기준>60%.
= WM 래치가 순차 선택을 구동하나. order_rate(무작위로 풀림) 우회한 순수 순차-의사결정 측정.
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def _side_rays(nh, left_val, right_val):
    return np.ones(nh) * left_val, np.ones(nh) * right_val


def run_choice(brain, env, n_trials=100, inhib=-200):
    obs = env.reset()
    brain.reset()
    for _ in range(30):
        a, info = brain.process(obs)
        obs, _, d, _ = env.step((a,))
        if d:
            obs = env.reset()

    nh = env.config.n_rays // 2
    correct_empty = 0   # WM 빔 → A 선택
    correct_loaded = 0  # WM에 A 적재 → B 선택
    n = 0
    for t in range(n_trials):
        a_side = "left" if np.random.random() > 0.5 else "right"  # A가 어느 쪽
        # --- 조건 1: WM 비움 → A로 가야 ---
        brain.gate_wm_input(0.0)
        base = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        al, ar = (0.9, 0.2) if a_side == "left" else (0.2, 0.9)   # A 쪽 강, B 쪽 약(둘 다 good food)
        base["good_food_rays_left"], base["good_food_rays_right"] = _side_rays(nh, al, ar)
        base["food_rays_left"], base["food_rays_right"] = _side_rays(nh, al, ar)
        tot = 0.0
        for _ in range(5):
            ang, info = brain.process(base)
            tot += ang
        # A가 왼쪽이면 왼쪽(음수)으로 가야 정답
        if (a_side == "left" and tot < -0.02) or (a_side == "right" and tot > 0.02):
            correct_empty += 1

        # --- 조건 2: A를 WM에 적재 → B로 가야 ---
        brain.reset();
        for _ in range(10):  # 워밍업
            a2, info = brain.process(obs)
            obs2, _, d, _ = env.step((a2,));
            if d: obs = env.reset()
        # A 위치로 이동 + 게이트 열어 A 적재
        env.agent_x = (0.3 if a_side == "left" else 0.7) * env.config.width
        env.agent_y = 0.3 * env.config.height
        oA = env._get_observation()
        for _ in range(8):
            brain.gate_wm_input(1.0)
            ang, info = brain.process(oA)
            brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
        # 이제 A/B 동시 제시, 게이트 닫음 → B로 가야(A 적재됐으니)
        brain.gate_wm_input(0.0)
        b_side = "right" if a_side == "left" else "left"
        base2 = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        bl, br = (0.9, 0.2) if b_side == "left" else (0.2, 0.9)
        base2["good_food_rays_left"], base2["good_food_rays_right"] = _side_rays(nh, bl, br)
        base2["food_rays_left"], base2["food_rays_right"] = _side_rays(nh, bl, br)
        tot2 = 0.0
        for _ in range(5):
            ang, info = brain.process(base2)
            tot2 += ang
        if (b_side == "left" and tot2 < -0.02) or (b_side == "right" and tot2 > 0.02):
            correct_loaded += 1
        n += 1

    return correct_empty / max(n, 1) * 100, correct_loaded / max(n, 1) * 100, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-weights", default=None)
    ap.add_argument("--inhib-wm", type=float, default=-200.0)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    cfg = ForagerBrainConfig()
    cfg.inhibitory_to_wm_weight = args.inhib_wm
    brain = ForagerBrain(cfg)
    if args.load_weights:
        brain.load_all_weights(args.load_weights)
    env = ForagerGym(ForagerConfig())
    ce, cl, n = run_choice(brain, env, n_trials=args.trials)
    print(f"SEQ-CHOICE: WM-empty→A {ce:.1f}% | WM-loaded→B {cl:.1f}% | combined {(ce+cl)/2:.1f}% "
          f"(random=50, n={n}) [{'PASS' if (ce+cl)/2 > 60 else 'FAIL'}]")


if __name__ == "__main__":
    main()
