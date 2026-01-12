"""
Real MiniGrid Growth - PPO learning on actual MiniGrid environments
===================================================================
Watch REAL learning on challenging MiniGrid tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import FlattenObservation
import time

plt.ion()


class LiveVisualizationCallback(BaseCallback):
    """Callback for live visualization during training."""

    def __init__(self, env, fig, axes, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.fig = fig
        self.ax_grid, self.ax_reward, self.ax_length, self.ax_success = axes

        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

        # Plot elements
        self.reward_line, = self.ax_reward.plot([], [], 'g-', lw=2)
        self.length_line, = self.ax_length.plot([], [], 'b-', lw=2)
        self.success_line, = self.ax_success.plot([], [], 'purple', lw=2)

        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.set_title('Episode Reward')
        self.ax_reward.grid(alpha=0.3)

        self.ax_length.set_xlabel('Episode')
        self.ax_length.set_ylabel('Steps')
        self.ax_length.set_title('Episode Length')
        self.ax_length.grid(alpha=0.3)

        self.ax_success.set_xlabel('Episode')
        self.ax_success.set_ylabel('Success Rate')
        self.ax_success.set_title('Success Rate (rolling 20)')
        self.ax_success.set_ylim(0, 1)
        self.ax_success.grid(alpha=0.3)

        self.info_text = self.ax_grid.text(0.02, 0.98, '', transform=self.ax_grid.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        self.last_render_time = 0

    def _on_step(self) -> bool:
        # Track rewards
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # Check for episode end
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Success = positive reward (reached goal)
            success = 1 if self.current_episode_reward > 0 else 0
            self.successes.append(success)

            self.current_episode_reward = 0
            self.current_episode_length = 0

        # Update visualization every 0.1 seconds
        current_time = time.time()
        if current_time - self.last_render_time > 0.1:
            self._update_viz()
            self.last_render_time = current_time

        return plt.fignum_exists(self.fig.number)

    def _update_viz(self):
        # Render environment
        try:
            self.ax_grid.clear()
            self.ax_grid.set_title('MiniGrid DoorKey')

            # Get RGB render
            rgb = self.env.envs[0].unwrapped.render()
            if rgb is not None:
                self.ax_grid.imshow(rgb)
            self.ax_grid.axis('off')

            # Info text
            n_eps = len(self.episode_rewards)
            if n_eps > 0:
                recent_reward = np.mean(self.episode_rewards[-20:]) if n_eps >= 20 else np.mean(self.episode_rewards)
                recent_success = np.mean(self.successes[-20:]) if n_eps >= 20 else np.mean(self.successes)
                self.info_text = self.ax_grid.text(0.02, 0.98,
                    f"Episodes: {n_eps}\n"
                    f"Avg Reward: {recent_reward:.2f}\n"
                    f"Success: {recent_success:.0%}",
                    transform=self.ax_grid.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        except Exception as e:
            pass

        # Update plots
        if len(self.episode_rewards) > 0:
            eps = range(len(self.episode_rewards))

            # Smooth curves
            window = min(20, len(self.episode_rewards))
            if len(self.episode_rewards) >= window:
                rewards_smooth = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                lengths_smooth = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
                success_smooth = np.convolve(self.successes, np.ones(window)/window, mode='valid')

                self.reward_line.set_data(range(window-1, len(self.episode_rewards)), rewards_smooth)
                self.length_line.set_data(range(window-1, len(self.episode_lengths)), lengths_smooth)
                self.success_line.set_data(range(window-1, len(self.successes)), success_smooth)
            else:
                self.reward_line.set_data(eps, self.episode_rewards)
                self.length_line.set_data(eps, self.episode_lengths)
                self.success_line.set_data(eps, self.successes)

            # Update axis limits
            self.ax_reward.set_xlim(0, max(10, len(self.episode_rewards)))
            self.ax_reward.set_ylim(min(self.episode_rewards) - 0.1, max(self.episode_rewards) + 0.1)

            self.ax_length.set_xlim(0, max(10, len(self.episode_lengths)))
            self.ax_length.set_ylim(0, max(self.episode_lengths) + 10)

            self.ax_success.set_xlim(0, max(10, len(self.successes)))

        plt.pause(0.001)


def make_env(env_id: str, render_mode: str = "rgb_array"):
    """Create MiniGrid environment."""
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = FlatObsWrapper(env)  # Flatten observations for MLP
        return env
    return _init


def run_minigrid_growth():
    """Train PPO on MiniGrid and watch it learn."""

    print("=" * 60)
    print("REAL MINIGRID GROWTH - PPO Learning")
    print("Watch the agent learn through curriculum!")
    print("Close the window to stop.")
    print("=" * 60)

    # Curriculum: Easy -> Hard
    curriculum = [
        ("MiniGrid-Empty-5x5-v0", "Empty room - just reach goal", 0.8),
        ("MiniGrid-Empty-8x8-v0", "Larger empty room", 0.7),
        ("MiniGrid-DoorKey-5x5-v0", "Find key, open door, reach goal", 0.5),
    ]

    current_level = 0
    env_id, task_desc, pass_threshold = curriculum[current_level]
    print(f"\nLevel {current_level + 1}: {env_id}")
    print(f"Task: {task_desc}")
    print()

    env = DummyVecEnv([make_env(env_id)])

    # Setup figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax_grid = fig.add_subplot(gs[0, 0])
    ax_reward = fig.add_subplot(gs[0, 1])
    ax_length = fig.add_subplot(gs[1, 0])
    ax_success = fig.add_subplot(gs[1, 1])

    axes = (ax_grid, ax_reward, ax_length, ax_success)

    fig.suptitle(f'MiniGrid Learning: {env_id}', fontsize=14, fontweight='bold')

    # Create callback
    callback = LiveVisualizationCallback(env, fig, axes)

    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=0
    )

    # Train
    print("Training... (close window to stop)")
    print()

    try:
        total_timesteps = 100000  # 100k steps
        model.learn(total_timesteps=total_timesteps, callback=callback)

    except Exception as e:
        if "figure" not in str(e).lower():
            print(f"Training stopped: {e}")

    # Final stats
    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    if len(callback.episode_rewards) > 0:
        n_eps = len(callback.episode_rewards)
        final_reward = np.mean(callback.episode_rewards[-50:]) if n_eps >= 50 else np.mean(callback.episode_rewards)
        final_success = np.mean(callback.successes[-50:]) if n_eps >= 50 else np.mean(callback.successes)

        print(f"Total Episodes: {n_eps}")
        print(f"Final Avg Reward: {final_reward:.2f}")
        print(f"Final Success Rate: {final_success:.0%}")

        # Growth analysis
        if n_eps >= 100:
            early_success = np.mean(callback.successes[:50])
            late_success = np.mean(callback.successes[-50:])
            print(f"\nGROWTH: {early_success:.0%} -> {late_success:.0%}")

    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_minigrid_growth()
