#!/usr/bin/env python3
"""
M3 Detour Test v3 — 4-Condition Replay Content Control

4 conditions:
1. no_replay: no SWR after switch
2. old_replay: SWR with OLD buffer (stale memories — should be harmful)
3. new_replay: SWR with NEW experiences only (should help)
4. mixed_replay: SWR with recency-weighted buffer (biologically realistic)

Measures first-100-step heading bias to old vs new rich zones.

Usage:
    python test_detour.py --learning-episodes 10 --test-episodes 1 --seeds 20
"""

import argparse
import numpy as np
import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def measure_first_n_steps(brain, env, old_zones, new_zones, n_steps=100):
    """첫 N스텝 heading bias 측정 — old zone vs new zone 방향"""
    obs = env.reset()
    # Rich zone 복원
    if new_zones:
        env.rich_zones = new_zones

    time_toward_old = 0
    time_toward_new = 0
    time_in_old = 0
    time_in_new = 0
    first_food_step = None
    rz_radius = env.config.rich_zone_radius

    for step in range(n_steps):
        angle, info = brain.process(obs)
        old_x, old_y = env.agent_x, env.agent_y
        obs, reward, done, step_info = env.step((angle,))
        new_x, new_y = env.agent_x, env.agent_y

        if first_food_step is None and env.total_food_eaten > 0:
            first_food_step = step

        # Zone occupancy
        for oz in old_zones:
            if np.sqrt((new_x - oz[0])**2 + (new_y - oz[1])**2) < rz_radius:
                time_in_old += 1
        for nz in new_zones:
            if np.sqrt((new_x - nz[0])**2 + (new_y - nz[1])**2) < rz_radius:
                time_in_new += 1

        # Heading: did the agent move TOWARD old or new zone?
        dx, dy = new_x - old_x, new_y - old_y
        if abs(dx) + abs(dy) > 0.1:  # moved
            for oz in old_zones:
                dist_before = np.sqrt((old_x - oz[0])**2 + (old_y - oz[1])**2)
                dist_after = np.sqrt((new_x - oz[0])**2 + (new_y - oz[1])**2)
                if dist_after < dist_before:
                    time_toward_old += 1
            for nz in new_zones:
                dist_before = np.sqrt((old_x - nz[0])**2 + (old_y - nz[1])**2)
                dist_after = np.sqrt((new_x - nz[0])**2 + (new_y - nz[1])**2)
                if dist_after < dist_before:
                    time_toward_new += 1

        if done:
            break

    return {
        "old_zone_pct": time_in_old / max(step + 1, 1),
        "new_zone_pct": time_in_new / max(step + 1, 1),
        "toward_old_pct": time_toward_old / max(step + 1, 1),
        "toward_new_pct": time_toward_new / max(step + 1, 1),
        "first_food": first_food_step,
        "steps_completed": step + 1,
    }


def run_single_seed(seed, learning_eps, condition, env_config):
    """단일 seed로 한 조건 실행"""
    np.random.seed(seed)

    brain_config = ForagerBrainConfig()
    env_config_copy = copy.deepcopy(env_config)
    env_config_copy.latent_switch_enabled = False
    env = ForagerGym(env_config_copy, render_mode="none")
    brain = ForagerBrain(brain_config)

    # === Phase A: Learning ===
    for ep in range(learning_eps):
        obs = env.reset()
        done = False
        while not done:
            angle, info = brain.process(obs)
            obs, reward, done, step_info = env.step((angle,))
        if brain_config.swr_replay_enabled:
            brain.replay_swr()

    old_zones = env.rich_zones.copy() if hasattr(env, 'rich_zones') and env.rich_zones else []
    old_buffer = list(brain.experience_buffer)  # 학습 후 buffer 저장

    # === Latent Switch ===
    new_zones = []
    if old_zones:
        margin = env_config_copy.rich_zone_radius + 50
        for _ in old_zones:
            for _ in range(200):
                cx = np.random.uniform(margin, env_config_copy.width - margin)
                cy = np.random.uniform(margin, env_config_copy.height - margin)
                far = all(np.sqrt((cx-oz[0])**2 + (cy-oz[1])**2) > env_config_copy.rich_zone_radius * 4
                         for oz in old_zones)
                if far:
                    new_zones.append((cx, cy))
                    break
        if new_zones:
            env.rich_zones = new_zones
            for i, food in enumerate(env.foods):
                if np.random.random() < 0.5 and new_zones:
                    zx, zy = new_zones[np.random.randint(len(new_zones))]
                    r = env_config_copy.rich_zone_radius * 0.8
                    nx = zx + np.random.uniform(-r, r)
                    ny = zy + np.random.uniform(-r, r)
                    nx = np.clip(nx, 20, env_config_copy.width - 20)
                    ny = np.clip(ny, 20, env_config_copy.height - 20)
                    env.foods[i] = (nx, ny, food[2])

    # === Brief new-env exploration (1 episode to populate buffer with new experiences) ===
    obs = env.reset()
    if new_zones:
        env.rich_zones = new_zones
    done = False
    while not done:
        angle, info = brain.process(obs)
        obs, reward, done, step_info = env.step((angle,))
    new_buffer = list(brain.experience_buffer)  # new experiences added

    # === Replay Window (condition-dependent) ===
    if condition == "no_replay":
        pass  # no replay
    elif condition == "old_replay":
        brain.experience_buffer = old_buffer  # restore old buffer
        brain.replay_swr()
    elif condition == "new_replay":
        # Keep only experiences from after the switch
        new_only = [e for e in new_buffer if e not in old_buffer]
        if not new_only:
            new_only = new_buffer[-5:]  # fallback
        brain.experience_buffer = new_only
        brain.replay_swr()
    elif condition == "mixed_replay":
        # Recency weighted: keep all but bias toward recent
        brain.experience_buffer = new_buffer  # includes both old and new
        brain.replay_swr()

    # === Test: First 100 steps in changed env ===
    result = measure_first_n_steps(brain, env, old_zones, new_zones, n_steps=100)
    result["condition"] = condition
    result["seed"] = seed
    return result


def run_detour_test(learning_eps=10, n_seeds=20):
    """4조건 × N seeds 실험"""
    env_config = ForagerConfig()
    conditions = ["no_replay", "old_replay", "new_replay", "mixed_replay"]
    all_results = {c: [] for c in conditions}

    print("=" * 70)
    print("  M3 DETOUR TEST v3 — 4-Condition Replay Content Control")
    print(f"  {learning_eps} learning eps × {n_seeds} seeds × 4 conditions")
    print("=" * 70)

    for seed in range(n_seeds):
        for cond in conditions:
            result = run_single_seed(seed, learning_eps, cond, env_config)
            all_results[cond].append(result)
        if (seed + 1) % 5 == 0:
            print(f"  seed {seed+1}/{n_seeds} complete")

    # === Results ===
    print(f"\n{'='*70}")
    print("  RESULTS (first 100 steps after latent switch)")
    print(f"{'='*70}")

    for cond in conditions:
        data = all_results[cond]
        toward_new = np.mean([d["toward_new_pct"] for d in data])
        toward_old = np.mean([d["toward_old_pct"] for d in data])
        in_new = np.mean([d["new_zone_pct"] for d in data])
        in_old = np.mean([d["old_zone_pct"] for d in data])
        food_steps = [d["first_food"] for d in data if d["first_food"] is not None]
        avg_food = np.mean(food_steps) if food_steps else float('inf')

        print(f"\n  {cond.upper()}:")
        print(f"    Toward new zone: {toward_new:.1%}")
        print(f"    Toward old zone: {toward_old:.1%}")
        print(f"    In new zone:     {in_new:.1%}")
        print(f"    In old zone:     {in_old:.1%}")
        print(f"    First food:      {avg_food:.0f} steps ({len(food_steps)}/{len(data)} found)")

    # === Comparison vs no_replay baseline ===
    print(f"\n{'='*70}")
    print("  COMPARISON vs NO_REPLAY baseline")
    print(f"{'='*70}")

    baseline_new = np.mean([d["toward_new_pct"] for d in all_results["no_replay"]])
    for cond in ["old_replay", "new_replay", "mixed_replay"]:
        data = all_results[cond]
        toward_new = np.mean([d["toward_new_pct"] for d in data])
        diff = (toward_new - baseline_new) * 100
        print(f"  {cond}: toward_new {diff:+.1f}pp vs no_replay")

    # Verdict
    new_vs_no = np.mean([d["toward_new_pct"] for d in all_results["new_replay"]]) - baseline_new
    old_vs_no = np.mean([d["toward_new_pct"] for d in all_results["old_replay"]]) - baseline_new

    print(f"\n  VERDICT:")
    if new_vs_no > 0.05:
        print(f"    ✓ New replay helps (+{new_vs_no*100:.1f}pp toward new zone)")
    else:
        print(f"    ✗ New replay doesn't help ({new_vs_no*100:+.1f}pp)")
    if old_vs_no < -0.02:
        print(f"    ✓ Old replay hurts ({old_vs_no*100:+.1f}pp — maladaptive consolidation confirmed)")
    else:
        print(f"    — Old replay neutral ({old_vs_no*100:+.1f}pp)")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 Detour Test v3")
    parser.add_argument("--learning-episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    results = run_detour_test(
        learning_eps=args.learning_episodes,
        n_seeds=args.seeds
    )
