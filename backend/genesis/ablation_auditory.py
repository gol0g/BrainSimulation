#!/usr/bin/env python3
"""
C1 Auditory Ablation: 20ep 체크포인트에서 3갈래 실험

A: 평가만 (300 trials) — 20ep 시점 실력 확인
B: 30ep 추가 학습 → 평가 — 퇴행 확인
C: 30ep 추가 학습 (auditory→D1 decay off) → 평가 — decay가 원인인지

Usage:
    python ablation_auditory.py --checkpoint brain_ablation_20ep.npz
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig
from evaluate_concepts import test_call_semantics, diagnose_auditory


def run_training(brain, n_episodes, freeze_auditory_decay=False):
    """Train brain for n_episodes, optionally freezing auditory→D1 decay"""
    env_config = ForagerConfig()
    brain_config = brain.config
    env = ForagerGym(env_config)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            angle, info = brain.process(obs)
            obs, reward, done, step_info = env.step((angle,))

            if step_info.get("food_eaten"):
                eaten_type = step_info.get("eaten_food_type", -1)
                food_pos = (obs["position_x"], obs["position_y"])
                if eaten_type == 0:
                    brain.learn_food_location(food_position=food_pos)
                    brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
                elif eaten_type == 1:
                    if brain_config.dopamine_dip_enabled:
                        brain.release_dopamine(reward_magnitude=-brain_config.dopamine_dip_magnitude)
                    if brain_config.taste_aversion_learning_enabled:
                        brain.trigger_taste_aversion()

            # Food sound incentive (보조)
            food_sound_high = obs.get("food_sound_high", 0.0)
            if food_sound_high > 0.3 and brain_config.basal_ganglia_enabled:
                brain.release_dopamine(reward_magnitude=0.05 * food_sound_high)

            brain.decay_dopamine()

        if brain_config.swr_replay_enabled:
            brain.replay_swr()

        # Freeze auditory weights after decay (restore to pre-decay values)
        if freeze_auditory_decay:
            for syn in [brain.kc_auditory_to_d1_l, brain.kc_auditory_to_d1_r]:
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values
                # Clamp: don't let decay reduce below current level
                w[:] = np.maximum(w, 0.5)  # At least init_w
                syn.vars["g"].values = w
                syn.vars["g"].push_to_device()


def get_auditory_weights(brain):
    """Get auditory→D1 weight stats"""
    stats = {}
    for name, syn in [("aud_d1_l", brain.kc_auditory_to_d1_l),
                       ("aud_d1_r", brain.kc_auditory_to_d1_r)]:
        syn.vars["g"].pull_from_device()
        w = syn.vars["g"].values
        stats[name] = {"mean": float(np.mean(w)), "std": float(np.std(w)),
                        "min": float(np.min(w)), "max": float(np.max(w))}
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("  C1 AUDITORY ABLATION STUDY")
    print("=" * 60)

    results = {}

    # === A: Eval only (300 trials) ===
    print(f"\n{'='*60}")
    print(f"  [A] Eval only at 20ep checkpoint (300 trials)")
    print(f"{'='*60}")
    config_a = ForagerBrainConfig()
    config_a.kc_auditory_to_d1_sparsity = 0.20
    brain_a = ForagerBrain(config_a)
    brain_a.load_all_weights(args.checkpoint)
    w_a = get_auditory_weights(brain_a)
    print(f"  Weights at 20ep: {w_a['aud_d1_l']['mean']:.4f}")
    call_a = test_call_semantics(brain_a, n_trials=100)
    print(f"  Call Semantics: {call_a['score']:.1f}% ({call_a['correct']}/{call_a['total']})")
    results["A"] = {"call": call_a["score"], "w_mean": w_a["aud_d1_l"]["mean"]}

    # === B: +30ep normal learning ===
    print(f"\n{'='*60}")
    print(f"  [B] +30ep normal learning → eval")
    print(f"{'='*60}")
    config_b = ForagerBrainConfig()
    config_b.kc_auditory_to_d1_sparsity = 0.20
    brain_b = ForagerBrain(config_b)
    brain_b.load_all_weights(args.checkpoint)
    run_training(brain_b, 30, freeze_auditory_decay=False)
    w_b = get_auditory_weights(brain_b)
    print(f"  Weights at 50ep: {w_b['aud_d1_l']['mean']:.4f}")
    call_b = test_call_semantics(brain_b, n_trials=100)
    print(f"  Call Semantics: {call_b['score']:.1f}% ({call_b['correct']}/{call_b['total']})")
    results["B"] = {"call": call_b["score"], "w_mean": w_b["aud_d1_l"]["mean"]}

    # === C: +30ep with auditory decay frozen ===
    print(f"\n{'='*60}")
    print(f"  [C] +30ep auditory→D1 decay frozen → eval")
    print(f"{'='*60}")
    config_c = ForagerBrainConfig()
    config_c.kc_auditory_to_d1_sparsity = 0.20
    brain_c = ForagerBrain(config_c)
    brain_c.load_all_weights(args.checkpoint)
    run_training(brain_c, 30, freeze_auditory_decay=True)
    w_c = get_auditory_weights(brain_c)
    print(f"  Weights at 50ep (frozen): {w_c['aud_d1_l']['mean']:.4f}")
    call_c = test_call_semantics(brain_c, n_trials=100)
    print(f"  Call Semantics: {call_c['score']:.1f}% ({call_c['correct']}/{call_c['total']})")
    results["C"] = {"call": call_c["score"], "w_mean": w_c["aud_d1_l"]["mean"]}

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':>20} | {'Call %':>8} | {'Weight':>8} | {'Interpretation'}")
    print(f"  {'-'*20} | {'-'*8} | {'-'*8} | {'-'*30}")
    for k, v in results.items():
        interp = ""
        if k == "A":
            interp = "20ep baseline"
        elif k == "B":
            if v["call"] < results["A"]["call"] - 5:
                interp = "REGRESSION — learning hurts"
            else:
                interp = "stable or improved"
        elif k == "C":
            if v["call"] > results["B"]["call"] + 5:
                interp = "DECAY IS THE CAUSE"
            elif v["call"] < results["A"]["call"] - 5:
                interp = "decay is not the only issue"
            else:
                interp = "similar to B"
        print(f"  {k:>20} | {v['call']:>7.1f}% | {v['w_mean']:>7.4f} | {interp}")

    print(f"\n  Decision:")
    if results["C"]["call"] > results["B"]["call"] + 10:
        print(f"  → Weight decay kills auditory learning. Fix: freeze or reduce decay for auditory.")
    elif results["A"]["call"] < 55:
        print(f"  → 20ep learning was weak/noisy. Need more training or stronger bridge.")
    else:
        print(f"  → Complex interaction. Consider KC_auditory expansion.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
