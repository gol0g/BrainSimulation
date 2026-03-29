#!/usr/bin/env python3
"""
개념 형성 평가 스크립트 (C0)

훈련된 모델을 로드하여 개념 관련 테스트를 실행.
훈련과 분리된 별도 평가 — 가중치 변경 없음.

Usage:
    python evaluate_concepts.py --load-weights brain_kc3000_b8.npz --test call_semantics
    python evaluate_concepts.py --load-weights brain_kc3000_b8.npz --test all
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def test_call_semantics(brain, n_trials=100):
    """
    Test 3: Call Semantics — 시각 단서 없이 NPC food call만으로 접근하는가

    시각 단서(food_rays)를 0으로 마스킹하고, NPC food call만 남긴 상태에서
    에이전트가 call 방향으로 이동하는 비율 측정.

    기준: >60% (random=50%)
    """
    env_config = ForagerConfig()
    env = ForagerGym(env_config)
    obs = env.reset()

    # 1회 워밍업 (50스텝)
    for _ in range(50):
        angle, info = brain.process(obs)
        obs, _, done, _ = env.step((angle,))
        if done:
            obs = env.reset()

    correct = 0
    total = 0

    for trial in range(n_trials):
        # 매 trial: env.reset() 하지 않고 obs만 조작
        call_direction = "left" if np.random.random() > 0.5 else "right"

        # 시각 단서 마스킹 + call 주입
        test_obs = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        test_obs["food_rays_left"] = np.zeros(8)
        test_obs["food_rays_right"] = np.zeros(8)
        test_obs["good_food_rays_left"] = np.zeros(8)
        test_obs["good_food_rays_right"] = np.zeros(8)
        test_obs["bad_food_rays_left"] = np.zeros(8)
        test_obs["bad_food_rays_right"] = np.zeros(8)

        # C1 food sound cue 주입 (고음 = 좋은 음식 근처)
        # + NPC food call도 동시 주입
        call_strength = 0.8
        test_obs["food_sound_high"] = call_strength  # 좋은 음식 소리
        test_obs["food_sound_low"] = 0.0
        if call_direction == "left":
            test_obs["npc_call_food_left"] = call_strength
            test_obs["npc_call_food_right"] = call_strength * 0.3
            test_obs["sound_food_left"] = call_strength
            test_obs["sound_food_right"] = call_strength * 0.3
        else:
            test_obs["npc_call_food_left"] = call_strength * 0.3
            test_obs["npc_call_food_right"] = call_strength
            test_obs["sound_food_left"] = call_strength * 0.3
            test_obs["sound_food_right"] = call_strength

        # 에이전트 반응 측정 (1스텝)
        angle, info = brain.process(test_obs)

        # 판정: call 방향으로 회전했는가
        # angle_delta = (motor_R - motor_L) * 0.5
        # angle > 0 = 오른쪽 회전, angle < 0 = 왼쪽 회전
        if call_direction == "left" and angle < -0.01:
            correct += 1
        elif call_direction == "right" and angle > 0.01:
            correct += 1
        total += 1

        # env 상태 갱신 (brain 내부 카운터 정상 유지)
        obs, _, done, _ = env.step((angle,))
        if done:
            obs = env.reset()

    score = correct / max(total, 1) * 100
    return {
        "test": "call_semantics",
        "score": score,
        "correct": correct,
        "total": total,
        "pass": score > 60.0,
        "baseline": 50.0,  # random
        "threshold": 60.0,
    }


def test_food_selectivity_baseline(brain, n_episodes=20):
    """
    Baseline: 현재 음식 선택성 측정 (기존 환경에서)

    good/bad 구분이 학습되었는지의 기본 지표.
    """
    env_config = ForagerConfig()
    env = ForagerGym(env_config)

    total_good = 0
    total_bad = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            angle, info = brain.process(obs)
            obs, reward, done, step_info = env.step((angle,))

        total_good += step_info.get("good_food_eaten", 0)
        total_bad += step_info.get("bad_food_eaten", 0)

    total = total_good + total_bad
    selectivity = total_good / max(total, 1)
    return {
        "test": "food_selectivity",
        "selectivity": selectivity,
        "good_eaten": total_good,
        "bad_eaten": total_bad,
        "pass": selectivity > 0.65,
        "threshold": 0.65,
    }


def test_spatial_memory(brain, n_trials=50):
    """
    Baseline: 공간 기억 — Rich Zone 방향으로 더 많이 이동하는가

    Rich Zone에 음식이 집중되므로, 학습 후 Rich Zone 방향 선호가 있어야 함.
    """
    env_config = ForagerConfig()
    env = ForagerGym(env_config)

    time_in_rich = 0
    total_steps = 0

    for trial in range(n_trials):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < 500:  # 짧은 에피소드
            angle, info = brain.process(obs)
            obs, reward, done, step_info = env.step((angle,))
            # Rich zone 내 시간 측정
            if env._in_rich_zone(env.agent_x, env.agent_y) >= 0:
                time_in_rich += 1
            total_steps += 1
            step += 1

    ratio = time_in_rich / max(total_steps, 1)
    # Rich zone이 맵의 약 7% 면적 (2 * pi * 120^2 / 800^2 ≈ 14%)
    # random이면 ~14%, 학습되면 >20%
    return {
        "test": "spatial_memory",
        "time_in_rich_ratio": ratio,
        "expected_random": 0.14,
        "pass": ratio > 0.20,
        "threshold": 0.20,
    }


def run_all_tests(brain):
    """모든 테스트 실행"""
    results = []

    print("=" * 60)
    print("  CONCEPT FORMATION EVALUATION (C0)")
    print("=" * 60)

    # Test: Food Selectivity Baseline
    print("\n[1/3] Food Selectivity Baseline...")
    r = test_food_selectivity_baseline(brain, n_episodes=10)
    results.append(r)
    status = "✓ PASS" if r["pass"] else "✗ FAIL"
    print(f"  Selectivity: {r['selectivity']:.2f} (good={r['good_eaten']}, bad={r['bad_eaten']}) [{status}]")

    # Test: Spatial Memory
    print("\n[2/3] Spatial Memory (Rich Zone preference)...")
    r = test_spatial_memory(brain, n_trials=20)
    results.append(r)
    status = "✓ PASS" if r["pass"] else "✗ FAIL"
    print(f"  Time in Rich Zone: {r['time_in_rich_ratio']:.1%} (random≈{r['expected_random']:.0%}) [{status}]")

    # Test: Call Semantics
    print("\n[3/3] Call Semantics (sound-only response)...")
    r = test_call_semantics(brain, n_trials=30)
    results.append(r)
    status = "✓ PASS" if r["pass"] else "✗ FAIL"
    print(f"  Call Response: {r['score']:.1f}% ({r['correct']}/{r['total']}) (random=50%) [{status}]")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r["pass"])
    print(f"  RESULTS: {passed}/{len(results)} tests passed")
    for r in results:
        status = "✓" if r["pass"] else "✗"
        print(f"    {status} {r['test']}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept Formation Evaluation (C0)")
    parser.add_argument("--load-weights", type=str, required=True,
                       help="Trained weights file to evaluate")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "call_semantics", "selectivity", "spatial"],
                       help="Which test to run")
    args = parser.parse_args()

    # Brain 생성 + 가중치 로드
    config = ForagerBrainConfig()
    brain = ForagerBrain(config)
    brain.load_all_weights(args.load_weights)
    print(f"Loaded weights from {args.load_weights}")

    if args.test == "all":
        run_all_tests(brain)
    elif args.test == "call_semantics":
        r = test_call_semantics(brain)
        print(f"Call Semantics: {r['score']:.1f}% ({'PASS' if r['pass'] else 'FAIL'})")
    elif args.test == "selectivity":
        r = test_food_selectivity_baseline(brain)
        print(f"Selectivity: {r['selectivity']:.2f} ({'PASS' if r['pass'] else 'FAIL'})")
    elif args.test == "spatial":
        r = test_spatial_memory(brain)
        print(f"Spatial Memory: {r['time_in_rich_ratio']:.1%} ({'PASS' if r['pass'] else 'FAIL'})")
