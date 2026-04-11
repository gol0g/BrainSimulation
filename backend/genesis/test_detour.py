#!/usr/bin/env python3
"""
M3 Detour Test v4 — Revaluation SWR vs No Replay

2 conditions × N seeds. 간소화 버전.
Revaluation replay = reverse value backup through transition graph.

Usage:
    python test_detour.py --seeds 5
"""

import argparse
import numpy as np
import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def run_single_seed(seed, learning_eps, condition):
    """단일 seed × 단일 조건"""
    np.random.seed(seed)

    brain_config = ForagerBrainConfig()
    env_config = ForagerConfig()
    env_config.latent_switch_enabled = False  # 수동 제어
    env = ForagerGym(env_config, render_mode="none")
    brain = ForagerBrain(brain_config)

    # === Phase A: Learning (stable env) ===
    for ep in range(learning_eps):
        obs = env.reset()
        done = False
        while not done:
            angle, info = brain.process(obs)
            obs, reward, done, step_info = env.step((angle,))
        if brain_config.swr_replay_enabled:
            brain.replay_swr()

    old_zones = env.rich_zones.copy() if hasattr(env, 'rich_zones') and env.rich_zones else []

    # === Latent Switch: Rich Zone 이동 ===
    new_zones = []
    if old_zones:
        margin = env_config.rich_zone_radius + 50
        for _ in old_zones:
            for _ in range(200):
                cx = np.random.uniform(margin, env_config.width - margin)
                cy = np.random.uniform(margin, env_config.height - margin)
                far = all(np.sqrt((cx-oz[0])**2 + (cy-oz[1])**2) > env_config.rich_zone_radius * 4
                         for oz in old_zones)
                if far:
                    new_zones.append((cx, cy))
                    break

    # Food를 새 rich zone으로 이동
    if new_zones:
        env.rich_zones = new_zones
        for i, food in enumerate(env.foods):
            if np.random.random() < 0.5:
                zx, zy = new_zones[np.random.randint(len(new_zones))]
                r = env_config.rich_zone_radius * 0.8
                nx = zx + np.random.uniform(-r, r)
                ny = zy + np.random.uniform(-r, r)
                nx = np.clip(nx, 20, env_config.width - 20)
                ny = np.clip(ny, 20, env_config.height - 20)
                env.foods[i] = (nx, ny, food[2])

    # === 1 episode in new env (discover new food → transition buffer 채우기) ===
    obs = env.reset()
    if new_zones:
        env.rich_zones = new_zones
        # foods도 새 위치로 재배치
        for i, food in enumerate(env.foods):
            if np.random.random() < 0.5 and new_zones:
                zx, zy = new_zones[np.random.randint(len(new_zones))]
                r = env_config.rich_zone_radius * 0.8
                nx = zx + np.random.uniform(-r, r)
                ny = zy + np.random.uniform(-r, r)
                nx = np.clip(nx, 20, env_config.width - 20)
                ny = np.clip(ny, 20, env_config.height - 20)
                env.foods[i] = (nx, ny, food[2])
    done = False
    while not done:
        angle, info = brain.process(obs)
        obs, reward, done, step_info = env.step((angle,))

    # === Replay Window ===
    if condition == "revaluation":
        brain.replay_swr()  # includes reverse value backup
    # else: no_replay — skip

    # === Test: First 100 steps ===
    obs = env.reset()
    if new_zones:
        env.rich_zones = new_zones
        for i, food in enumerate(env.foods):
            if np.random.random() < 0.5 and new_zones:
                zx, zy = new_zones[np.random.randint(len(new_zones))]
                r = env_config.rich_zone_radius * 0.8
                nx = zx + np.random.uniform(-r, r)
                ny = zy + np.random.uniform(-r, r)
                nx = np.clip(nx, 20, env_config.width - 20)
                ny = np.clip(ny, 20, env_config.height - 20)
                env.foods[i] = (nx, ny, food[2])

    time_in_new = 0
    time_in_old = 0
    first_food = None
    rz_r = env_config.rich_zone_radius

    for step in range(100):
        angle, info = brain.process(obs)
        obs, reward, done, step_info = env.step((angle,))
        ax, ay = env.agent_x, env.agent_y

        for oz in old_zones:
            if np.sqrt((ax-oz[0])**2 + (ay-oz[1])**2) < rz_r:
                time_in_old += 1
        for nz in new_zones:
            if np.sqrt((ax-nz[0])**2 + (ay-nz[1])**2) < rz_r:
                time_in_new += 1
        if first_food is None and env.total_food_eaten > 0:
            first_food = step
        if done:
            break

    return {
        "new_zone_pct": time_in_new / 100,
        "old_zone_pct": time_in_old / 100,
        "first_food": first_food,
        "food_100": env.total_food_eaten,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--learning-episodes", type=int, default=10)
    args = parser.parse_args()

    conditions = ["revaluation", "no_replay"]
    results = {c: [] for c in conditions}

    print("=" * 60)
    print(f"  M3 DETOUR v4 — Revaluation vs No Replay")
    print(f"  {args.seeds} seeds × 2 conditions")
    print("=" * 60)

    for seed in range(args.seeds):
        for cond in conditions:
            r = run_single_seed(seed, args.learning_episodes, cond)
            results[cond].append(r)
        print(f"  seed {seed+1}/{args.seeds}: "
              f"reval new={results['revaluation'][-1]['new_zone_pct']:.0%} "
              f"no_rep new={results['no_replay'][-1]['new_zone_pct']:.0%}")

    print(f"\n{'='*60}")
    print("  RESULTS (first 100 steps)")
    print(f"{'='*60}")

    for cond in conditions:
        d = results[cond]
        new_z = np.mean([x["new_zone_pct"] for x in d])
        old_z = np.mean([x["old_zone_pct"] for x in d])
        food = np.mean([x["food_100"] for x in d])
        ff = [x["first_food"] for x in d if x["first_food"] is not None]
        ff_avg = np.mean(ff) if ff else float('inf')
        print(f"\n  {cond.upper()}:")
        print(f"    New zone: {new_z:.1%}")
        print(f"    Old zone: {old_z:.1%}")
        print(f"    Food(100): {food:.1f}")
        print(f"    First food: {ff_avg:.0f} steps")

    reval_new = np.mean([x["new_zone_pct"] for x in results["revaluation"]])
    no_new = np.mean([x["new_zone_pct"] for x in results["no_replay"]])
    diff = (reval_new - no_new) * 100

    print(f"\n  DIFF: revaluation - no_replay = {diff:+.1f}pp toward new zone")
    print(f"  VERDICT: {'REVALUATION HELPS' if diff > 5 else 'NOT CONCLUSIVE' if diff > 0 else 'NO EFFECT'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
