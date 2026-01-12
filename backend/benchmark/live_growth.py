"""
LIVE Growth Visualization - Watch learning in REAL-TIME
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass, field
from typing import Dict, Tuple

plt.ion()  # Interactive mode


@dataclass
class ForagingEnv:
    grid_size: int = 10
    max_steps: int = 100
    agent_pos: np.ndarray = field(default_factory=lambda: np.array([5, 5]))
    food_pos: np.ndarray = field(default_factory=lambda: np.array([8, 8]))
    step_count: int = 0
    food_collected: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = np.array([5, 5])
        self.food_pos = self.rng.integers(0, self.grid_size, 2)
        while np.array_equal(self.food_pos, self.agent_pos):
            self.food_pos = self.rng.integers(0, self.grid_size, 2)
        self.step_count = 0
        self.food_collected = 0
        return self._obs()

    def _obs(self):
        diff = self.food_pos - self.agent_pos
        return {'agent_pos': self.agent_pos.copy(), 'food_pos': self.food_pos.copy(),
                'food_dx': diff[0], 'food_dy': diff[1], 'food_dist': np.linalg.norm(diff)}

    def step(self, action):
        self.step_count += 1
        dx, dy = [(-1,0),(1,0),(0,1),(0,-1),(0,0)][action]
        self.agent_pos = np.clip(self.agent_pos + [dx, dy], 0, self.grid_size-1)

        reward = -0.01
        info = {'food': False}
        if np.linalg.norm(self.agent_pos - self.food_pos) < 1.5:
            reward = 1.0
            self.food_collected += 1
            info['food'] = True
            self.food_pos = self.rng.integers(0, self.grid_size, 2)
            while np.linalg.norm(self.agent_pos - self.food_pos) < 2:
                self.food_pos = self.rng.integers(0, self.grid_size, 2)

        return self._obs(), reward, self.step_count >= self.max_steps, info


class LearningAgent:
    def __init__(self):
        self.Q = np.zeros((5, 5, 5))  # [dx_bucket, dy_bucket, action]
        self.epsilon = 1.0
        self.lr = 0.3

    def _state(self, obs):
        dx = int(np.clip(obs['food_dx'], -2, 2) + 2)
        dy = int(np.clip(obs['food_dy'], -2, 2) + 2)
        return dx, dy

    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(5)
        sx, sy = self._state(obs)
        return np.argmax(self.Q[sx, sy])

    def learn(self, obs, action, reward, next_obs, done):
        sx, sy = self._state(obs)
        nsx, nsy = self._state(next_obs)
        target = reward + 0.95 * (0 if done else np.max(self.Q[nsx, nsy]))
        self.Q[sx, sy, action] += self.lr * (target - self.Q[sx, sy, action])

    def decay_epsilon(self):
        self.epsilon = max(0.05, self.epsilon * 0.97)


def run_live():
    print("=" * 50)
    print("LIVE LEARNING - Watch the agent grow!")
    print("Close the window to stop.")
    print("=" * 50)

    env = ForagingEnv()
    agent = LearningAgent()

    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax_grid, ax_curve, ax_eps = axes

    # Grid
    ax_grid.set_xlim(-0.5, 9.5)
    ax_grid.set_ylim(-0.5, 9.5)
    ax_grid.set_aspect('equal')
    ax_grid.grid(alpha=0.3)
    ax_grid.set_title('Agent (blue) finding Food (green)')

    agent_circle = Circle((5, 5), 0.4, color='blue')
    food_circle = Circle((8, 8), 0.35, color='green')
    ax_grid.add_patch(agent_circle)
    ax_grid.add_patch(food_circle)
    trail_line, = ax_grid.plot([], [], 'b-', alpha=0.3, lw=1)

    # Learning curve
    ax_curve.set_xlabel('Episode')
    ax_curve.set_ylabel('Food Collected')
    ax_curve.set_title('Learning Progress')
    ax_curve.grid(alpha=0.3)
    learn_line, = ax_curve.plot([], [], 'g-', lw=2)

    # Epsilon
    ax_eps.set_xlabel('Episode')
    ax_eps.set_ylabel('Exploration Rate')
    ax_eps.set_title('Exploration -> Exploitation')
    ax_eps.set_ylim(0, 1)
    ax_eps.grid(alpha=0.3)
    eps_line, = ax_eps.plot([], [], 'purple', lw=2)

    info_text = ax_grid.text(0.02, 0.98, '', transform=ax_grid.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    episode_foods = []
    episode_epsilons = []
    trail = []

    episode = 0
    max_episodes = 150

    try:
        while episode < max_episodes and plt.fignum_exists(fig.number):
            obs = env.reset(seed=episode)
            trail = [obs['agent_pos'].copy()]
            done = False

            while not done and plt.fignum_exists(fig.number):
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                agent.learn(obs, action, reward, next_obs, done)
                obs = next_obs
                trail.append(obs['agent_pos'].copy())

                # Update display every 5 steps
                if env.step_count % 5 == 0:
                    agent_circle.center = obs['agent_pos']
                    food_circle.center = obs['food_pos']

                    trail_arr = np.array(trail[-30:])
                    if len(trail_arr) > 1:
                        trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])

                    info_text.set_text(f"Episode {episode+1}\nStep {env.step_count}\n"
                                       f"Food: {env.food_collected}\n"
                                       f"Epsilon: {agent.epsilon:.2f}")

                    plt.pause(0.01)

            # End of episode
            agent.decay_epsilon()
            episode_foods.append(env.food_collected)
            episode_epsilons.append(agent.epsilon)

            # Update curves
            learn_line.set_data(range(len(episode_foods)), episode_foods)
            ax_curve.set_xlim(0, max(10, len(episode_foods)))
            ax_curve.set_ylim(0, max(episode_foods) + 2)

            eps_line.set_data(range(len(episode_epsilons)), episode_epsilons)
            ax_eps.set_xlim(0, max(10, len(episode_epsilons)))

            episode += 1

            if episode % 10 == 0:
                avg = np.mean(episode_foods[-10:])
                print(f"Episode {episode}: avg food = {avg:.1f}, epsilon = {agent.epsilon:.2f}")

        print()
        print("=" * 50)
        print("DONE!")
        early = np.mean(episode_foods[:10]) if len(episode_foods) >= 10 else 0
        late = np.mean(episode_foods[-10:]) if len(episode_foods) >= 10 else 0
        print(f"Growth: {early:.1f} -> {late:.1f} food per episode")
        print("=" * 50)

        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_live()
