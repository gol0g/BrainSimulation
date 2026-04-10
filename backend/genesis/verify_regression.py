#!/usr/bin/env python3
"""
Regression Verification Suite — 하네스 Evaluator 역할

모든 변경사항 후 반드시 실행. 단일 지표가 아닌 다면 평가.
커밋 전 이 스크립트가 PASS해야 함.

Usage:
    python verify_regression.py --load-weights brain_xxx.npz [--episodes 20]
    python verify_regression.py --fresh --episodes 20
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def run_survival_test(brain_config, env_config, episodes=20):
    """생존율 + 기본 지표 테스트"""
    env = ForagerGym(env_config, render_mode="none")
    brain = ForagerBrain(brain_config)
    return brain, env, brain_config, env_config


def verify_all(brain, env, brain_config, env_config, episodes=20,
               load_weights=None):
    """전체 회귀 테스트 실행"""
    if load_weights:
        brain.load_all_weights(load_weights)
        print(f"  Loaded: {load_weights}")

    results = {}
    failures = []

    print("=" * 60)
    print("  REGRESSION VERIFICATION SUITE")
    print("=" * 60)

    # === 1. Survival Test ===
    print(f"\n[1/5] Survival Test ({episodes} episodes)...")
    all_steps = []
    all_food = []
    death_causes = {}
    pain_deaths = 0

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            angle, info = brain.process(obs)
            obs, reward, done, step_info = env.step((angle,))
        steps = env.steps
        food = env.total_food_eaten
        cause = step_info.get("death_cause", "unknown")
        all_steps.append(steps)
        all_food.append(food)
        death_causes[cause] = death_causes.get(cause, 0) + 1
        if cause == "pain":
            pain_deaths += 1

        # SWR Replay between episodes
        if brain_config.swr_replay_enabled and brain_config.hippocampus_enabled:
            brain.replay_swr()

        if (ep + 1) % 5 == 0:
            surv_so_far = sum(1 for s in all_steps if s >= env_config.max_steps - 10) / (ep + 1)
            print(f"    ep {ep+1}: survival {surv_so_far:.0%}")

    survived = sum(1 for s in all_steps if s >= env_config.max_steps - 10)
    survival_rate = survived / episodes
    reward_freq = sum(all_food) / max(sum(all_steps), 1)
    pain_death_rate = pain_deaths / episodes

    results["survival"] = survival_rate
    results["reward_freq"] = reward_freq
    results["pain_death_rate"] = pain_death_rate
    results["avg_steps"] = np.mean(all_steps)
    results["avg_food"] = np.mean(all_food)

    # Criteria
    if survival_rate < 0.40:
        failures.append(f"Survival {survival_rate:.0%} < 40%")
    if pain_death_rate > 0.15:
        failures.append(f"Pain death {pain_death_rate:.0%} > 15%")

    status = "PASS" if survival_rate >= 0.40 else "FAIL"
    print(f"  Survival: {survival_rate:.0%} [{status}]")
    print(f"  Reward Freq: {reward_freq:.2%}")
    print(f"  Pain Death: {pain_death_rate:.0%}")
    print(f"  Deaths: {death_causes}")

    # === 2. Weight Health Check ===
    print(f"\n[2/5] Weight Health Check...")
    weight_issues = []

    # Check predictive plasticity weights
    if hasattr(brain, 'place_to_pred'):
        brain.place_to_pred.vars["g"].pull_from_device()
        w = brain.place_to_pred.vars["g"].view.copy()
        at_ceil = np.sum(w >= brain_config.place_to_pred_w_max * 0.95) / len(w)
        std = np.std(w)
        results["pred_at_ceil"] = at_ceil
        results["pred_std"] = std
        if at_ceil > 0.50:
            weight_issues.append(f"place→pred at_ceil={at_ceil:.0%} > 50%")
        print(f"  place→pred: avg={np.mean(w):.3f}, std={std:.3f}, at_ceil={at_ceil:.0%}")

    # Check Hebbian synapses for saturation
    hebbian_checks = [
        ("forward_model", "efference_to_predict_hebbian", "agency_forward_model_w_max"),
        ("body_narrative", "body_to_narrative_hebbian", "self_narrative_binding_w_max"),
        ("agency_narrative", "agency_to_narrative_hebbian", "agency_to_narrative_w_max"),
    ]
    for name, attr, wmax_attr in hebbian_checks:
        if hasattr(brain, attr):
            syn = getattr(brain, attr)
            syn.vars["g"].pull_from_device()
            w = syn.vars["g"].view.copy()
            w_max = getattr(brain_config, wmax_attr, 10.0)
            at_ceil = np.sum(w >= w_max * 0.95) / max(len(w), 1)
            results[f"{name}_at_ceil"] = at_ceil
            if at_ceil > 0.80:
                weight_issues.append(f"{name} at_ceil={at_ceil:.0%} > 80%")
            print(f"  {name}: avg={np.mean(w):.3f}, at_ceil={at_ceil:.0%}")

    if weight_issues:
        for issue in weight_issues:
            failures.append(f"Weight: {issue}")
        print(f"  Weight issues: {len(weight_issues)}")
    else:
        print(f"  All weights healthy")

    # === 3. Food Selectivity ===
    print(f"\n[3/5] Food Selectivity...")
    total_good = env.good_food_eaten
    total_bad = env.bad_food_eaten
    total = total_good + total_bad
    selectivity = total_good / max(total, 1)
    results["selectivity"] = selectivity
    if selectivity < 0.55:
        failures.append(f"Selectivity {selectivity:.2f} < 0.55")
    print(f"  Selectivity: {selectivity:.2f} (good={total_good}, bad={total_bad})")

    # === 4. Curiosity & Prediction Active ===
    print(f"\n[4/5] Circuit Activity Check...")
    pred_rate = brain.last_pred_food_rate
    curiosity_rate = brain.last_curiosity_rate
    surprise_rate = brain.last_surprise_rate
    results["pred_rate"] = pred_rate
    results["curiosity_rate"] = curiosity_rate
    results["surprise_rate"] = surprise_rate
    print(f"  Pred_FoodSoon: {pred_rate:.3f}")
    print(f"  Curiosity: {curiosity_rate:.3f}")
    print(f"  Surprise: {surprise_rate:.3f}")

    # === 5. Summary ===
    print(f"\n{'=' * 60}")
    if failures:
        print(f"  RESULT: FAIL ({len(failures)} issues)")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print(f"  RESULT: PASS (all checks passed)")

    print(f"\n  Survival:    {survival_rate:.0%}")
    print(f"  Selectivity: {selectivity:.2f}")
    print(f"  Reward Freq: {reward_freq:.2%}")
    print(f"  Pain Death:  {pain_death_rate:.0%}")
    print(f"{'=' * 60}")

    return len(failures) == 0, results, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression Verification Suite")
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--fresh", action="store_true",
                       help="Run with fresh (untrained) brain")
    args = parser.parse_args()

    brain_config = ForagerBrainConfig()
    env_config = ForagerConfig()
    env = ForagerGym(env_config, render_mode="none")
    brain = ForagerBrain(brain_config)

    passed, results, failures = verify_all(
        brain, env, brain_config, env_config,
        episodes=args.episodes,
        load_weights=args.load_weights
    )

    sys.exit(0 if passed else 1)
