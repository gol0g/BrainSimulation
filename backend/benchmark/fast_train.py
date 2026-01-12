"""
Fast MiniGrid Training - No visualization, maximum speed
"""

import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=10000):
        super().__init__()
        self.total = total_timesteps
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_count = 0
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.locals.get('dones') is not None and self.locals['dones'][0]:
            self.episode_count += 1
            if 'episode' in self.locals.get('infos', [{}])[0]:
                self.episode_rewards.append(self.locals['infos'][0]['episode']['r'])

        if self.num_timesteps % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed
            pct = self.num_timesteps / self.total * 100

            recent_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"  {self.num_timesteps:,}/{self.total:,} ({pct:.0f}%) | "
                  f"Episodes: {self.episode_count} | "
                  f"Reward: {recent_reward:.2f} | "
                  f"FPS: {fps:.0f}")
        return True


def make_env(env_id):
    def _init():
        env = gym.make(env_id)
        env = FlatObsWrapper(env)
        return env
    return _init


def train_fast(env_id: str, total_timesteps: int = 500000, n_envs: int = 8):
    """Train as fast as possible."""
    print(f"\n{'='*60}")
    print(f"FAST TRAINING: {env_id}")
    print(f"Steps: {total_timesteps:,} | Parallel envs: {n_envs}")
    print(f"{'='*60}\n")

    # Parallel environments for speed
    env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])

    # PPO with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0
    )

    callback = ProgressCallback(total_timesteps)

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - start

    print(f"\nTraining complete in {elapsed:.1f}s ({total_timesteps/elapsed:.0f} FPS)")

    # Evaluate
    print("\nEvaluating...")
    eval_env = DummyVecEnv([make_env(env_id)])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

    print(f"\n{'='*60}")
    print(f"RESULTS: {env_id}")
    print(f"{'='*60}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate: {(mean_reward > 0).mean() if hasattr(mean_reward, 'mean') else ('YES' if mean_reward > 0 else 'NO')}")

    env.close()
    eval_env.close()

    return model, mean_reward


def run_curriculum():
    """Run full curriculum training."""
    print("\n" + "="*60)
    print("CURRICULUM TRAINING")
    print("="*60)

    curriculum = [
        ("MiniGrid-Empty-5x5-v0", 100000),
        ("MiniGrid-Empty-8x8-v0", 200000),
        ("MiniGrid-Empty-Random-6x6-v0", 300000),
        ("MiniGrid-DoorKey-5x5-v0", 500000),
    ]

    results = []
    for env_id, steps in curriculum:
        model, reward = train_fast(env_id, steps, n_envs=8)
        results.append((env_id, reward))

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for env_id, reward in results:
        status = "PASS" if reward > 0 else "FAIL"
        print(f"  [{status}] {env_id}: {reward:.2f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        env_id = sys.argv[1]
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
        train_fast(env_id, steps)
    else:
        # Default: DoorKey with 500k steps
        train_fast("MiniGrid-DoorKey-5x5-v0", 500000, n_envs=8)
