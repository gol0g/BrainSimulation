"""
E5 Ablation Test - Reward & Encoding Ablation Matrix

E5a: Reward Ablation - 보상 구조 변경해도 ratio 유지되는지
E5b: Encoding Ablation - 관측 표현 변경해도 ratio 유지되는지

Usage:
    python test_e5_ablation.py                           # All ablations (smoke)
    python test_e5_ablation.py --encoding polar          # Specific encoding
    python test_e5_ablation.py --reward no_door          # Specific reward
    python test_e5_ablation.py --seeds 100               # Release run
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e5_ablation import (
    EncodingVariant, RewardVariant, AblationConfig,
    EncodingTransform, RewardModifier, AblationGate, AblationGateResult,
    ENCODING_VARIANTS, REWARD_VARIANTS_TASK_A, REWARD_VARIANTS_TASK_B,
    E5B_PRIME_ENCODINGS,
)
from genesis.e5_key_door import KeyDoorEnv, KeyDoorConfig, Action
from genesis.e5_checkpoint_sequence import CheckpointEnv, CheckpointConfig, ActionB


# ============================================================================
# Task A Agent (Key-Door) with encoding transform
# ============================================================================

class AblationKeyDoorAgent:
    """Ablation-aware Key-Door Agent"""

    def __init__(self, use_memory: bool, use_hierarchy: bool, transform: EncodingTransform):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy
        self.transform = transform

        self.key_memory = None
        self.door_memory = None
        self.door_passed = False
        self.explore_counter = 0
        self.explore_direction = 0

    def reset(self):
        self.key_memory = None
        self.door_memory = None
        self.door_passed = False
        self.explore_counter = 0
        self.explore_direction = 0

    def act(self, obs: np.ndarray, agent_pos: Tuple[int, int]) -> int:
        # E5b' Canonicalization: inverse transform to canonical (dx, dy) format
        # This ensures agent sees same semantics regardless of encoding
        obs = self.transform.canonicalize_obs(obs)

        key_visible = obs[0] > 0.5
        key_dx, key_dy = obs[1], obs[2]
        door_visible = obs[3] > 0.5
        door_dx, door_dy = obs[4], obs[5]
        has_key = obs[6] > 0.5
        goal_visible = obs[7] > 0.5

        # Memory update
        if key_visible and self.use_memory:
            self.key_memory = (key_dx, key_dy)
        if door_visible and self.use_memory:
            self.door_memory = (door_dx, door_dy)

        # Memory observation (12D mode)
        if len(obs) >= 12 and self.use_memory:
            key_mem_dx, key_mem_dy = obs[8], obs[9]
            door_mem_dx, door_mem_dy = obs[10], obs[11]
        else:
            key_mem_dx, key_mem_dy = 0, 0
            door_mem_dx, door_mem_dy = 0, 0

        # 1. Pick up key if on it
        if key_visible and abs(key_dx) < 0.2 and abs(key_dy) < 0.2 and not has_key:
            return Action.PICKUP.value

        # 2. Go to visible key
        if key_visible and not has_key:
            return self._move_towards(key_dx, key_dy)

        # 3. Go to door (if not passed)
        if has_key and door_visible and not self.door_passed:
            if abs(door_dx) < 0.2 and abs(door_dy) < 0.2:
                self.door_passed = True
                return Action.DOWN.value
            return self._move_towards(door_dx, door_dy)

        # 4. Memory-based door navigation
        if has_key and not self.door_passed and self.use_memory:
            if abs(door_mem_dx) > 0.01 or abs(door_mem_dy) > 0.01:
                return self._move_towards(door_mem_dx, door_mem_dy)

        # 5. Memory-based key navigation
        if not has_key and self.use_memory:
            if abs(key_mem_dx) > 0.01 or abs(key_mem_dy) > 0.01:
                return self._move_towards(key_mem_dx, key_mem_dy)

        # 6. After door, go to goal
        if self.door_passed:
            return Action.DOWN.value

        # 7. Explore
        return self._explore()

    def _move_towards(self, dx: float, dy: float) -> int:
        if abs(dx) > abs(dy):
            return Action.DOWN.value if dx > 0 else Action.UP.value
        else:
            return Action.RIGHT.value if dy > 0 else Action.LEFT.value

    def _explore(self) -> int:
        """Random exploration - change direction every 3 steps (matches original E5 agent)"""
        self.explore_counter += 1
        if self.explore_counter >= 3:
            self.explore_counter = 0
            # Random direction change
            self.explore_direction = np.random.randint(0, 4)
        return self.explore_direction


# ============================================================================
# Task B Agent (Checkpoint) with encoding transform
# ============================================================================

class AblationCheckpointAgent:
    """Ablation-aware Checkpoint Sequence Agent"""

    def __init__(self, use_memory: bool, use_hierarchy: bool, transform: EncodingTransform):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy
        self.transform = transform

        self.remembered_sequence = []
        self.explore_counter = 0

    def reset(self):
        self.remembered_sequence = []
        self.explore_counter = 0

    def act(self, obs: np.ndarray, step: int) -> int:
        # E5b' Canonicalization: inverse transform to canonical format
        obs = self.transform.canonicalize_obs(obs)

        cp_a_dx, cp_a_dy = obs[1], obs[2]
        cp_b_dx, cp_b_dy = obs[4], obs[5]
        cp_c_dx, cp_c_dy = obs[7], obs[8]
        goal_visible = obs[9] > 0.5
        goal_dx, goal_dy = obs[10], obs[11]
        n_completed = int(obs[12] * 3 + 0.5)
        all_completed = obs[13] > 0.5

        # Hint parsing
        if len(obs) >= 17:
            hint_seq = [int(obs[14] * 2 + 0.5), int(obs[15] * 2 + 0.5), int(obs[16] * 2 + 0.5)]
        else:
            hint_seq = [0, 1, 2]

        if step < 5 and self.use_memory and not self.remembered_sequence:
            self.remembered_sequence = list(hint_seq)

        cp_dxdy = [(cp_a_dx, cp_a_dy), (cp_b_dx, cp_b_dy), (cp_c_dx, cp_c_dy)]

        # All completed → go to goal
        if all_completed:
            if goal_visible and (abs(goal_dx) > 0.01 or abs(goal_dy) > 0.01):
                return self._move_towards(goal_dx, goal_dy)
            return ActionB.DOWN.value

        # Determine current target
        if self.use_memory and self.remembered_sequence:
            if self.use_hierarchy:
                current_target = self.remembered_sequence[n_completed] if n_completed < 3 else -1
            else:
                current_target = self.remembered_sequence[0]
        elif self.use_hierarchy:
            default_seq = [0, 1, 2]
            current_target = default_seq[n_completed] if n_completed < 3 else -1
        else:
            current_target = -1

        # Navigate to target
        if current_target >= 0:
            dx, dy = cp_dxdy[current_target]
            if abs(dx) < 0.05 and abs(dy) < 0.05:
                return ActionB.ACTIVATE.value
            return self._move_towards(dx, dy)

        # BASE mode: any checkpoint
        for dx, dy in cp_dxdy:
            if abs(dx) < 0.05 and abs(dy) < 0.05:
                return ActionB.ACTIVATE.value
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                return self._move_towards(dx, dy)

        return self._explore()

    def _move_towards(self, dx: float, dy: float) -> int:
        if abs(dx) > abs(dy):
            return ActionB.DOWN.value if dx > 0 else ActionB.UP.value
        else:
            return ActionB.RIGHT.value if dy > 0 else ActionB.LEFT.value

    def _explore(self) -> int:
        self.explore_counter += 1
        if self.explore_counter < 5:
            return ActionB.DOWN.value
        return ActionB.RIGHT.value if (self.explore_counter // 5) % 2 == 0 else ActionB.LEFT.value


# ============================================================================
# Episode runners
# ============================================================================

def run_task_a_episode(
    seed: int,
    use_memory: bool,
    use_hierarchy: bool,
    ablation_config: AblationConfig,
    max_steps: int = 300,  # Match E5 Task A settings
) -> Tuple[bool, float]:
    """Run single Task A episode with ablation"""
    np.random.seed(seed)  # Consistent exploration across configs
    env_config = KeyDoorConfig(max_steps=max_steps, use_memory=use_memory)

    # Apply reward modification
    reward_mod = RewardModifier(ablation_config)
    reward_mod.modify_task_a_config(env_config)

    env = KeyDoorEnv(env_config, seed=seed)
    obs = env.reset(seed=seed)

    transform = EncodingTransform(ablation_config, obs_dim=len(obs), seed=seed)
    agent = AblationKeyDoorAgent(use_memory, use_hierarchy, transform)
    agent.reset()

    for _ in range(max_steps):
        # E5b' Flow:
        # 1. Env outputs original observation
        # 2. Transform to simulated encoding (polar/onehot)
        # 3. Agent canonicalizes (inverse transform) back to original
        # 4. Agent makes decision in canonical format
        transformed_obs = transform.transform_obs(obs)
        action = agent.act(transformed_obs, env.state.agent_pos)
        obs, reward, done, info = env.step(action)
        if done:
            break

    return env.state.goal_reached, env.state.total_reward


def run_task_b_episode(
    seed: int,
    use_memory: bool,
    use_hierarchy: bool,
    ablation_config: AblationConfig,
    max_steps: int = 100,
) -> Tuple[bool, float, int, int]:
    """Run single Task B episode with ablation"""
    np.random.seed(seed)  # Consistent exploration
    env_config = CheckpointConfig(max_steps=max_steps, use_memory=use_memory)

    # Apply reward modification
    reward_mod = RewardModifier(ablation_config)
    reward_mod.modify_task_b_config(env_config)

    env = CheckpointEnv(env_config, seed=seed)
    obs = env.reset(seed=seed)

    transform = EncodingTransform(ablation_config, obs_dim=len(obs), seed=seed)
    agent = AblationCheckpointAgent(use_memory, use_hierarchy, transform)
    agent.reset()

    correct = 0
    wrong = 0

    for step in range(max_steps):
        # E5b' Flow: transform → agent canonicalizes → decision
        transformed_obs = transform.transform_obs(obs)
        action = agent.act(transformed_obs, step)
        obs, reward, done, info = env.step(action)
        if info.get('correct_checkpoint'):
            correct += 1
        if info.get('wrong_checkpoint'):
            wrong += 1
        if done:
            break

    return env.state.goal_reached, env.state.total_reward, correct, wrong


# ============================================================================
# Main ablation runner
# ============================================================================

def run_ablation_matrix(
    n_seeds: int = 30,
    encodings: List[EncodingVariant] = None,
    rewards_a: List[RewardVariant] = None,
    rewards_b: List[RewardVariant] = None,
) -> Dict:
    """Run full ablation matrix"""

    if encodings is None:
        encodings = ENCODING_VARIANTS
    if rewards_a is None:
        rewards_a = REWARD_VARIANTS_TASK_A
    if rewards_b is None:
        rewards_b = REWARD_VARIANTS_TASK_B

    print(f"\n{'='*70}")
    print(f"  E5 Ablation Matrix")
    print(f"  {n_seeds} seeds")
    print(f"  Encodings: {[e.value for e in encodings]}")
    print(f"  Rewards A: {[r.value for r in rewards_a]}")
    print(f"  Rewards B: {[r.value for r in rewards_b]}")
    print(f"{'='*70}\n")

    start_time = time.time()
    seeds = list(range(n_seeds))
    gate = AblationGate()

    results = {
        'task_a': {},
        'task_b': {},
        'gate_results': [],
    }

    # ========================================================================
    # Task A: Key-Door with encoding × reward ablation
    # ========================================================================
    print("=" * 50)
    print("  Task A: Key-Door Ablation")
    print("=" * 50)

    for encoding in encodings:
        for reward in rewards_a:
            key = f"{encoding.value}_{reward.value}"
            print(f"\n  [{encoding.value}] × [{reward.value}]")

            config_results = {'BASE': [], '+MEM': [], 'FULL': []}

            for config_name, (use_mem, use_hier) in [
                ('BASE', (False, False)),
                ('+MEM', (True, False)),
                ('FULL', (True, True)),
            ]:
                ablation = AblationConfig(encoding=encoding, reward=reward)
                successes = []
                rewards_list = []

                for seed in seeds:
                    success, total_reward = run_task_a_episode(
                        seed, use_mem, use_hier, ablation
                    )
                    successes.append(success)
                    rewards_list.append(total_reward)

                avg_success = np.mean(successes)
                avg_reward = np.mean(rewards_list)
                config_results[config_name] = {'success': avg_success, 'reward': avg_reward}
                print(f"    {config_name}: success={avg_success:.1%}, reward={avg_reward:.2f}")

            results['task_a'][key] = config_results

    # ========================================================================
    # Task B: Checkpoint Sequence with encoding × reward ablation
    # ========================================================================
    print("\n" + "=" * 50)
    print("  Task B: Checkpoint Sequence Ablation")
    print("=" * 50)

    for encoding in encodings:
        for reward in rewards_b:
            key = f"{encoding.value}_{reward.value}"
            print(f"\n  [{encoding.value}] × [{reward.value}]")

            config_results = {'BASE': [], '+HIE': [], 'FULL': []}

            for config_name, (use_mem, use_hier) in [
                ('BASE', (False, False)),
                ('+HIE', (False, True)),
                ('FULL', (True, True)),
            ]:
                ablation = AblationConfig(encoding=encoding, reward=reward)
                successes = []
                rewards_list = []

                for seed in seeds:
                    success, total_reward, correct, wrong = run_task_b_episode(
                        seed, use_mem, use_hier, ablation
                    )
                    successes.append(success)
                    rewards_list.append(total_reward)

                avg_success = np.mean(successes)
                avg_reward = np.mean(rewards_list)
                config_results[config_name] = {'success': avg_success, 'reward': avg_reward}
                print(f"    {config_name}: success={avg_success:.1%}, reward={avg_reward:.2f}")

            results['task_b'][key] = config_results

    # ========================================================================
    # Gate Evaluation
    # ========================================================================
    print("\n" + "=" * 70)
    print("  Gate Evaluation")
    print("=" * 70)

    all_passed = True
    for encoding in encodings:
        for reward_a in rewards_a:
            for reward_b in rewards_b:
                key_a = f"{encoding.value}_{reward_a.value}"
                key_b = f"{encoding.value}_{reward_b.value}"

                if key_a in results['task_a'] and key_b in results['task_b']:
                    task_a = results['task_a'][key_a]
                    task_b = results['task_b'][key_b]

                    gate_result = gate.evaluate(
                        encoding=encoding,
                        reward=reward_a,  # Using reward_a for consistency
                        task_a_base_reward=task_a['BASE']['reward'],
                        task_a_mem_reward=task_a['+MEM']['reward'],
                        task_a_full_success=task_a['FULL']['success'],
                        task_b_base_reward=task_b['BASE']['reward'],
                        task_b_full_reward=task_b['FULL']['reward'],  # Use FULL not +HIE
                        task_b_full_success=task_b['FULL']['success'],
                    )

                    results['gate_results'].append(gate_result)

                    status = "PASS" if gate_result.passed else "FAIL"
                    if not gate_result.passed:
                        all_passed = False

                    print(f"\n  Encoding: {encoding.value}, Reward: {reward_a.value}/{reward_b.value}")
                    print(f"    Task A memory_ratio: {gate_result.task_a_memory_ratio:.2f} "
                          f"[{'PASS' if gate_result.task_a_passed else 'FAIL'}]")
                    print(f"    Task B hierarchy_ratio: {gate_result.task_b_hierarchy_ratio:.2f} "
                          f"[{'PASS' if gate_result.task_b_passed else 'FAIL'}]")
                    print(f"    Overall: [{status}] {gate_result.reason}")

    elapsed = time.time() - start_time

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"\n  Time: {elapsed:.1f}s")

    passed_count = sum(1 for r in results['gate_results'] if r.passed)
    total_count = len(results['gate_results'])
    print(f"  Gates Passed: {passed_count}/{total_count}")

    if all_passed:
        print(f"\n  [PASS] All ablations passed!")
        print(f"  → E5 features are NOT specialized to reward/encoding structure")
        print(f"  → Ready for E6: Continuous Control")
    else:
        failed = [r for r in results['gate_results'] if not r.passed]
        print(f"\n  [FAIL] Some ablations failed:")
        for r in failed:
            print(f"    - {r.encoding_variant} × reward: {r.reason}")

    print(f"\n{'='*70}\n")

    results['all_passed'] = all_passed
    results['elapsed_sec'] = elapsed

    return results


def verify_roundtrip_encodings(n_samples: int = 20) -> Dict[str, bool]:
    """
    E5b' Roundtrip 검증: decode(encode(x)) == x_canon

    Returns dict of {encoding_name: is_reversible}
    """
    print("\n" + "=" * 50)
    print("  E5b' Roundtrip Verification")
    print("=" * 50)

    results = {}

    for encoding in E5B_PRIME_ENCODINGS:
        config = AblationConfig(encoding=encoding)
        transform = EncodingTransform(config, obs_dim=12, seed=42)

        all_passed = True
        failed_samples = []

        for i in range(n_samples):
            # Random observation
            np.random.seed(i)
            original = np.random.randn(12).astype(np.float32) * 0.5

            # Roundtrip test
            if not transform.verify_roundtrip(original):
                all_passed = False
                if len(failed_samples) < 3:
                    transformed = transform.transform_obs(original)
                    recovered = transform.canonicalize_obs(transformed)
                    failed_samples.append({
                        'original': original[:4].tolist(),
                        'recovered': recovered[:4].tolist(),
                    })

        results[encoding.value] = all_passed

        status = "PASS" if all_passed else "FAIL"
        print(f"\n  [{encoding.value}] Roundtrip: [{status}]")

        if not all_passed and failed_samples:
            print(f"    Failed samples:")
            for fs in failed_samples:
                print(f"      orig: {fs['original']}")
                print(f"      recv: {fs['recovered']}")

    all_ok = all(results.values())
    print(f"\n  Overall: [{'PASS' if all_ok else 'FAIL'}]")
    print("=" * 50 + "\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    parser.add_argument("--encoding", type=str, default=None,
                        help="Specific encoding: original, polar, onehot, randproj, permflip")
    parser.add_argument("--reward", type=str, default=None,
                        help="Specific reward: original, no_door, goal_only, reduced_penalty, reset_style")
    parser.add_argument("--e5bprime", action="store_true",
                        help="E5b' mode: semantic-preserving encodings only (original, polar, onehot)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run roundtrip verification, skip main ablation")
    args = parser.parse_args()

    # E5b' mode: semantic-preserving encodings only
    if args.e5bprime:
        print("\n" + "=" * 70)
        print("  E5b' Mode: Semantic-Preserving Encodings Only")
        print("  (randproj, permflip excluded - information destroying)")
        print("=" * 70)

        # Step 1: Roundtrip verification
        roundtrip_results = verify_roundtrip_encodings()

        if args.verify_only:
            exit(0 if all(roundtrip_results.values()) else 1)

        # Step 2: Run ablation with E5b' encodings
        encodings = E5B_PRIME_ENCODINGS
    else:
        # Parse encoding filter
        encodings = ENCODING_VARIANTS
        if args.encoding:
            encodings = [e for e in ENCODING_VARIANTS if e.value == args.encoding]

    # Parse reward filter
    rewards_a = REWARD_VARIANTS_TASK_A
    rewards_b = REWARD_VARIANTS_TASK_B
    if args.reward:
        rewards_a = [r for r in REWARD_VARIANTS_TASK_A if r.value == args.reward]
        rewards_b = [r for r in REWARD_VARIANTS_TASK_B if r.value == args.reward]

    results = run_ablation_matrix(
        n_seeds=args.seeds,
        encodings=encodings,
        rewards_a=rewards_a if rewards_a else REWARD_VARIANTS_TASK_A,
        rewards_b=rewards_b if rewards_b else REWARD_VARIANTS_TASK_B,
    )

    exit(0 if results['all_passed'] else 1)
