#!/usr/bin/env python3
"""
M3 Detour/Reversal Test — Replay-Driven Replanning 측정

환경이 바뀐 직후 첫 시도에서 에이전트가 적응하는지 측정.
Replay ON vs OFF 비교로 replay의 인과적 기여 증명.

Protocol:
1. Phase A (학습): 안정 환경에서 N 에피소드 학습
2. Latent switch: Pain zone + 장애물 재배치 (에이전트 모르게)
3. Replay window: ON 조건은 SWR replay 실행, OFF는 건너뜀
4. Phase B (테스트): 변경된 환경에서 첫 에피소드 성능 측정

측정 지표:
- 첫 시도 생존 시간 (steps)
- Pain zone 진입 횟수 (이전 위치에 갇히는가?)
- 음식 획득 속도 (first food latency)

Usage:
    python test_detour.py --learning-episodes 10 --test-episodes 5
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def run_detour_test(learning_eps=10, test_eps=5):
    """Replay ON vs OFF detour 비교 실험"""

    results = {"replay_on": [], "replay_off": []}

    for condition in ["replay_on", "replay_off"]:
        print(f"\n{'='*60}")
        print(f"  CONDITION: {condition.upper()}")
        print(f"{'='*60}")

        brain_config = ForagerBrainConfig()
        env_config = ForagerConfig()
        # Latent switch 비활성화 (수동 제어)
        env_config.latent_switch_enabled = False
        env = ForagerGym(env_config, render_mode="none")
        brain = ForagerBrain(brain_config)

        # === Phase A: 안정 환경에서 학습 ===
        print(f"\n  Phase A: Learning ({learning_eps} episodes, stable env)...")
        for ep in range(learning_eps):
            obs = env.reset()
            done = False
            while not done:
                angle, info = brain.process(obs)
                obs, reward, done, step_info = env.step((angle,))

            # Replay between episodes
            if brain_config.swr_replay_enabled and brain_config.hippocampus_enabled:
                brain.replay_swr()

            survived = "✓" if env.steps >= env_config.max_steps - 10 else "✗"
            if (ep + 1) % 5 == 0:
                print(f"    ep {ep+1}: steps={env.steps}, food={env.total_food_eaten} {survived}")

        # 학습 후 Pain zone 위치 기록
        old_pain_zones = env.pain_zones.copy() if env.pain_zones else []
        old_obstacles = env.obstacles.copy() if env.obstacles else []

            # === Latent Switch: Rich Zone 이동 (핵심!) ===
        # Pain zone이 아닌 FOOD 위치를 바꿔야 replay의 hippocampal 기여 측정 가능
        # Pain 회피는 반사(Push-Pull)로 충분 — replay 없이도 작동
        # 음식 위치 변경은 hippocampal food memory 업데이트가 필요
        old_rich_zones = env.rich_zones.copy() if hasattr(env, 'rich_zones') and env.rich_zones else []
        new_rich = []  # 기본값
        print(f"\n  Latent Switch: relocating RICH ZONES (food locations)...")
        print(f"    Old rich zones: {[(int(z[0]),int(z[1])) for z in old_rich_zones]}")

        # Rich zone을 기존 위치에서 멀리 이동
        if hasattr(env, 'rich_zones') and env.rich_zones:
            new_rich = []
            margin = env_config.rich_zone_radius + 50
            for _ in env.rich_zones:
                for _ in range(200):
                    cx = np.random.uniform(margin, env_config.width - margin)
                    cy = np.random.uniform(margin, env_config.height - margin)
                    far_enough = all(
                        np.sqrt((cx - oz[0])**2 + (cy - oz[1])**2) > env_config.rich_zone_radius * 4
                        for oz in old_rich_zones
                    )
                    if far_enough:
                        new_rich.append((cx, cy))
                        break
            if new_rich:
                env.rich_zones = new_rich
                # 음식도 새 rich zone으로 재배치
                for i, food in enumerate(env.foods):
                    if np.random.random() < 0.5:
                        zx, zy = new_rich[np.random.randint(len(new_rich))]
                        r = env_config.rich_zone_radius * 0.8
                        nx = zx + np.random.uniform(-r, r)
                        ny = zy + np.random.uniform(-r, r)
                        nx = np.clip(nx, 20, env_config.width - 20)
                        ny = np.clip(ny, 20, env_config.height - 20)
                        env.foods[i] = (nx, ny, food[2])
                print(f"    New rich zones: {[(int(z[0]),int(z[1])) for z in new_rich]}")

        # === Replay Window ===
        if condition == "replay_on":
            print(f"\n  Replay Window: executing SWR replay...")
            if brain_config.swr_replay_enabled and brain_config.hippocampus_enabled:
                replay_info = brain.replay_swr()
                if replay_info:
                    print(f"    Replayed {replay_info.get('replayed_count', 0)} experiences")
        else:
            print(f"\n  Replay Window: SKIPPED (control condition)")

        # === Phase B: 변경된 환경에서 테스트 ===
        print(f"\n  Phase B: Testing ({test_eps} episodes, changed env)...")
        test_results = []

        for ep in range(test_eps):
            # Reset하되 Rich zone은 변경된 위치 유지
            obs = env.reset()
            # Rich zone을 변경된 위치로 복원 (reset이 초기화할 수 있으므로)
            if hasattr(env, 'rich_zones') and new_rich:
                env.rich_zones = new_rich

            done = False
            first_food_step = None
            time_in_old_zone = 0
            time_in_new_zone = 0
            total_steps_test = 0

            while not done:
                angle, info = brain.process(obs)
                obs, reward, done, step_info = env.step((angle,))

                # First food latency
                if first_food_step is None and env.total_food_eaten > 0:
                    first_food_step = env.steps

                # Time in old vs new rich zone
                ax, ay = env.agent_x, env.agent_y
                for oz in old_rich_zones:
                    if np.sqrt((ax - oz[0])**2 + (ay - oz[1])**2) < env_config.rich_zone_radius:
                        time_in_old_zone += 1
                if hasattr(env, 'rich_zones'):
                    for nz in env.rich_zones:
                        if np.sqrt((ax - nz[0])**2 + (ay - nz[1])**2) < env_config.rich_zone_radius:
                            time_in_new_zone += 1
                total_steps_test += 1

            steps = env.steps
            food = env.total_food_eaten
            survived = steps >= env_config.max_steps - 10
            old_ratio = time_in_old_zone / max(total_steps_test, 1)
            new_ratio = time_in_new_zone / max(total_steps_test, 1)

            test_results.append({
                "steps": steps,
                "food": food,
                "survived": survived,
                "first_food_step": first_food_step or steps,
                "old_zone_ratio": old_ratio,
                "new_zone_ratio": new_ratio,
                "death_cause": step_info.get("death_cause", "unknown"),
            })

            status = "✓" if survived else f"✗({step_info.get('death_cause', '?')})"
            print(f"    test {ep+1}: steps={steps}, food={food}, "
                  f"old_zone={old_ratio:.1%}, new_zone={new_ratio:.1%}, "
                  f"first_food={first_food_step or 'never'} {status}")

            # Replay between test episodes too
            if condition == "replay_on" and brain_config.swr_replay_enabled:
                brain.replay_swr()

        results[condition] = test_results

    # === 결과 비교 ===
    print(f"\n{'='*60}")
    print(f"  DETOUR TEST RESULTS")
    print(f"{'='*60}")

    for cond in ["replay_on", "replay_off"]:
        data = results[cond]
        avg_steps = np.mean([d["steps"] for d in data])
        avg_food = np.mean([d["food"] for d in data])
        survival = sum(1 for d in data if d["survived"]) / len(data)
        avg_old_zone = np.mean([d["old_zone_ratio"] for d in data])
        avg_new_zone = np.mean([d["new_zone_ratio"] for d in data])
        avg_first_food = np.mean([d["first_food_step"] for d in data])

        print(f"\n  {cond.upper()}:")
        print(f"    Survival:        {survival:.0%}")
        print(f"    Avg food:        {avg_food:.1f}")
        print(f"    Old zone time:   {avg_old_zone:.1%} (should be LOW if adapted)")
        print(f"    New zone time:   {avg_new_zone:.1%} (should be HIGH if adapted)")
        print(f"    First food step: {avg_first_food:.0f}")

    # 차이 계산
    on_old = np.mean([d["old_zone_ratio"] for d in results["replay_on"]])
    off_old = np.mean([d["old_zone_ratio"] for d in results["replay_off"]])
    on_new = np.mean([d["new_zone_ratio"] for d in results["replay_on"]])
    off_new = np.mean([d["new_zone_ratio"] for d in results["replay_off"]])
    on_food = np.mean([d["food"] for d in results["replay_on"]])
    off_food = np.mean([d["food"] for d in results["replay_off"]])
    on_food_lat = np.mean([d["first_food_step"] for d in results["replay_on"]])
    off_food_lat = np.mean([d["first_food_step"] for d in results["replay_off"]])

    print(f"\n  COMPARISON (replay_on - replay_off):")
    print(f"    Old zone time:   {on_old - off_old:+.1%} (negative = replay helps forget old)")
    print(f"    New zone time:   {on_new - off_new:+.1%} (positive = replay finds new)")
    print(f"    Food count:      {on_food - off_food:+.1f}")
    print(f"    First food:      {on_food_lat - off_food_lat:+.0f} steps")

    # Pass/Fail — replay should reduce old zone time OR increase new zone time
    replay_helps = (off_old - on_old) > 0.01 or (on_new - off_new) > 0.01 or (on_food - off_food) > 2
    print(f"\n  VERDICT: {'REPLAY HELPS' if replay_helps else 'REPLAY EFFECT NOT DETECTED'}")
    print(f"{'='*60}")

    return replay_helps, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 Detour/Reversal Test")
    parser.add_argument("--learning-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=5)
    args = parser.parse_args()

    passed, results = run_detour_test(
        learning_eps=args.learning_episodes,
        test_eps=args.test_episodes
    )
    sys.exit(0 if passed else 1)
