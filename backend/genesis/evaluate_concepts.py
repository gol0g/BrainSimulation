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

        # C1: sound_food 채널로 방향 단서 주입 (KC_auditory가 학습한 채널)
        call_strength = 0.8
        test_obs["food_sound_high"] = call_strength
        test_obs["food_sound_low"] = 0.0
        if call_direction == "left":
            test_obs["sound_food_left"] = call_strength
            test_obs["sound_food_right"] = call_strength * 0.1
        else:
            test_obs["sound_food_left"] = call_strength * 0.1
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


def diagnose_auditory(brain, n_trials=20):
    """
    C1 진단: 소리가 행동에 영향 못 주는 원인 분리

    Stage 1: Auditory Decode — KC_auditory에서 sound vs no-sound 구분이 되는가
    Stage 2: Policy Bias — vision 없이 sound만으로 motor bias가 생기는가
    Stage 3: Eligibility — sound→food 순서에서 KC_aud→D1 가중치가 변하는가
    """
    env_config = ForagerConfig()
    env = ForagerGym(env_config)
    obs = env.reset()

    # Warmup
    for _ in range(50):
        angle, info = brain.process(obs)
        obs, _, done, _ = env.step((angle,))
        if done:
            obs = env.reset()

    print("\n" + "=" * 60)
    print("  AUDITORY DIAGNOSIS (C1)")
    print("=" * 60)

    # === Stage 1: Auditory Decode ===
    # sound_food가 있을 때 vs 없을 때 KC_auditory 스파이크 차이
    kc_aud_with_sound = []
    kc_aud_without_sound = []

    for trial in range(n_trials):
        # No sound
        test_obs = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        test_obs["food_sound_high"] = 0.0
        test_obs["food_sound_low"] = 0.0
        test_obs["sound_food_left"] = 0.0
        test_obs["sound_food_right"] = 0.0
        angle_no, info_no = brain.process(test_obs)
        obs, _, done, _ = env.step((angle_no,))
        if done: obs = env.reset()

        # KC_auditory rate (from info if available, else estimate from angle difference)
        kc_aud_without_sound.append(abs(angle_no))

        # With sound (left)
        test_obs2 = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        test_obs2["food_sound_high"] = 0.8
        test_obs2["food_sound_low"] = 0.0
        test_obs2["sound_food_left"] = 0.8
        test_obs2["sound_food_right"] = 0.1
        test_obs2["food_rays_left"] = np.zeros(8)
        test_obs2["food_rays_right"] = np.zeros(8)
        test_obs2["good_food_rays_left"] = np.zeros(8)
        test_obs2["good_food_rays_right"] = np.zeros(8)
        angle_snd, info_snd = brain.process(test_obs2)
        obs, _, done, _ = env.step((angle_snd,))
        if done: obs = env.reset()

        kc_aud_with_sound.append(angle_snd)

    # Analyze Stage 1
    avg_no_sound = np.mean([abs(a) for a in kc_aud_without_sound])
    avg_with_sound = np.mean([abs(a) for a in kc_aud_with_sound])
    sound_angles = kc_aud_with_sound
    left_turns = sum(1 for a in sound_angles if a < -0.01)
    right_turns = sum(1 for a in sound_angles if a > 0.01)
    no_turns = sum(1 for a in sound_angles if abs(a) <= 0.01)

    print(f"\n[Stage 1] Auditory Representation")
    print(f"  Motor magnitude: no_sound={avg_no_sound:.4f}, with_sound={avg_with_sound:.4f}")
    print(f"  Difference: {abs(avg_with_sound - avg_no_sound):.4f}")
    decode_pass = abs(avg_with_sound - avg_no_sound) > 0.005
    print(f"  Sound changes motor output: {'YES' if decode_pass else 'NO'}")

    # === Stage 2: Policy Bias ===
    # sound left일 때 왼쪽으로 도는 비율
    print(f"\n[Stage 2] Policy Bias (sound left → turn left?)")
    print(f"  Sound LEFT trials: left_turn={left_turns}, right_turn={right_turns}, no_turn={no_turns}")
    bias = left_turns / max(left_turns + right_turns, 1)
    bias_pass = bias > 0.55 or bias < 0.45  # 50%에서 벗어나면 bias 있음
    print(f"  Left turn ratio: {bias:.1%} (random=50%)")
    print(f"  Policy bias exists: {'YES' if bias_pass else 'NO (random)'}")

    # === Stage 3: Weight check ===
    print(f"\n[Stage 3] KC_auditory→D1 Weight Status")
    try:
        for name in ['kc_auditory_to_d1_l', 'kc_auditory_to_d2_l']:
            syn = getattr(brain, name, None)
            if syn:
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values
                print(f"  {name}: mean={np.mean(w):.4f}, std={np.std(w):.4f}, min={np.min(w):.4f}, max={np.max(w):.4f}")
    except Exception as e:
        print(f"  Weight read error: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"    Stage 1 (Representation):  {'✓' if decode_pass else '✗'} Sound changes motor")
    print(f"    Stage 2 (Policy bias):     {'✓' if bias_pass else '✗'} Directional response")
    print(f"    Bottleneck: ", end="")
    if not decode_pass:
        print("REPRESENTATION — sound doesn't affect network output at all")
    elif not bias_pass:
        print("POLICY COUPLING — sound affects network but not directionally")
    else:
        print("CREDIT ASSIGNMENT — bias exists but doesn't persist across episodes")
    print(f"{'='*60}")

    return {
        "decode_pass": decode_pass,
        "bias_pass": bias_pass,
        "avg_no_sound": avg_no_sound,
        "avg_with_sound": avg_with_sound,
        "left_bias": bias,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept Formation Evaluation (C0)")
    parser.add_argument("--load-weights", type=str, required=True,
                       help="Trained weights file to evaluate")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "call_semantics", "selectivity", "spatial", "diagnose_auditory"],
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
    elif args.test == "diagnose_auditory":
        diagnose_auditory(brain)
