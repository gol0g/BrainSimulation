"""
Genesis Brain GROWTH Visualization
===================================
Watch the agent GROW from a newborn to an expert.

Simple foraging environment:
- Agent learns to find food efficiently
- No danger (pure learning demo)
- Clear performance improvement over episodes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrow
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ForagingEnv:
    """Simple foraging environment - find food!"""
    grid_size: int = 10
    max_steps: int = 100

    agent_pos: np.ndarray = field(default_factory=lambda: np.array([5, 5]))
    food_pos: np.ndarray = field(default_factory=lambda: np.array([8, 8]))
    step_count: int = 0
    food_collected: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def reset(self, seed: int = None) -> Dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.food_pos = self.rng.integers(0, self.grid_size, 2)
        # Make sure food is not at agent position
        while np.array_equal(self.food_pos, self.agent_pos):
            self.food_pos = self.rng.integers(0, self.grid_size, 2)

        self.step_count = 0
        self.food_collected = 0
        return self._get_obs()

    def _get_obs(self) -> Dict:
        diff = self.food_pos - self.agent_pos
        dist = np.linalg.norm(diff)
        return {
            'agent_pos': self.agent_pos.copy(),
            'food_pos': self.food_pos.copy(),
            'food_dx': diff[0],
            'food_dy': diff[1],
            'food_dist': dist
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        self.step_count += 1

        # Move
        dx, dy = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)][action]
        self.agent_pos = np.clip(
            self.agent_pos + np.array([dx, dy]),
            0, self.grid_size - 1
        )

        # Check food
        reward = -0.01  # Small step cost
        info = {'food': False}

        if np.linalg.norm(self.agent_pos - self.food_pos) < 1.5:
            reward = 1.0
            self.food_collected += 1
            info['food'] = True
            # Respawn food
            self.food_pos = self.rng.integers(0, self.grid_size, 2)
            while np.linalg.norm(self.agent_pos - self.food_pos) < 2:
                self.food_pos = self.rng.integers(0, self.grid_size, 2)

        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, info


class GrowingAgent:
    """Agent that learns to forage efficiently."""

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size

        # Q-table: [dx_bucket, dy_bucket, action] -> value
        # dx, dy buckets: -2, -1, 0, 1, 2 (5 each)
        self.Q = np.zeros((5, 5, 5))

        # Exploration rate (decreases with learning)
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.05

        # Learning rate
        self.lr = 0.2
        self.gamma = 0.95

        # Statistics
        self.total_food = 0
        self.episodes = 0

    def _obs_to_state(self, obs: Dict) -> Tuple[int, int]:
        """Convert observation to state indices."""
        dx = np.clip(obs['food_dx'], -2, 2)
        dy = np.clip(obs['food_dy'], -2, 2)
        return int(dx + 2), int(dy + 2)

    def act(self, obs: Dict) -> int:
        """Select action using epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(5)

        sx, sy = self._obs_to_state(obs)
        return np.argmax(self.Q[sx, sy])

    def learn(self, obs: Dict, action: int, reward: float, next_obs: Dict, done: bool):
        """Update Q-table."""
        sx, sy = self._obs_to_state(obs)
        nsx, nsy = self._obs_to_state(next_obs)

        best_next = np.max(self.Q[nsx, nsy]) if not done else 0
        target = reward + self.gamma * best_next
        self.Q[sx, sy, action] += self.lr * (target - self.Q[sx, sy, action])

    def end_episode(self, food_collected: int):
        """Called at end of episode."""
        self.total_food += food_collected
        self.episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_growth_experiment(n_episodes: int = 100, seed: int = 42):
    """Run the growth experiment."""
    np.random.seed(seed)

    env = ForagingEnv(max_steps=100)
    agent = GrowingAgent()

    history = []

    for ep in range(n_episodes):
        obs = env.reset(seed + ep)
        trajectory = []

        done = False
        while not done:
            action = agent.act(obs)
            trajectory.append({
                'pos': obs['agent_pos'].copy(),
                'food': obs['food_pos'].copy(),
                'epsilon': agent.epsilon
            })

            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs

        agent.end_episode(env.food_collected)

        history.append({
            'episode': ep,
            'food': env.food_collected,
            'steps': env.step_count,
            'epsilon': agent.epsilon,
            'trajectory': trajectory
        })

        if (ep + 1) % 20 == 0:
            avg_food = np.mean([h['food'] for h in history[-20:]])
            print(f"Episode {ep+1}: avg food = {avg_food:.1f}, epsilon = {agent.epsilon:.2f}")

    return history, agent


def create_growth_visualization(history: List[Dict], agent: GrowingAgent):
    """Create the growth visualization."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Learning curve
    ax_learn = fig.add_subplot(gs[0, :2])
    food = [h['food'] for h in history]
    episodes = range(len(history))

    ax_learn.plot(episodes, food, 'g-', alpha=0.3, lw=1)
    # Moving average
    window = 10
    food_ma = np.convolve(food, np.ones(window)/window, mode='valid')
    ax_learn.plot(range(window-1, len(food)), food_ma, 'g-', lw=3, label='Moving avg')
    ax_learn.fill_between(range(window-1, len(food)), 0, food_ma, alpha=0.3, color='green')

    ax_learn.set_xlabel('Episode', fontsize=12)
    ax_learn.set_ylabel('Food Collected', fontsize=12)
    ax_learn.set_title('LEARNING: Food Collection Over Time', fontsize=14, fontweight='bold')
    ax_learn.legend()
    ax_learn.grid(alpha=0.3)

    # Add growth annotation
    early = np.mean(food[:10])
    late = np.mean(food[-10:])
    improvement = (late - early) / max(early, 0.1) * 100
    ax_learn.annotate(f'+{improvement:.0f}% improvement!',
                     xy=(len(food)*0.7, late),
                     fontsize=14, color='darkgreen', fontweight='bold')

    # Epsilon decay
    ax_eps = fig.add_subplot(gs[0, 2])
    epsilon = [h['epsilon'] for h in history]
    ax_eps.plot(episodes, epsilon, 'purple', lw=2)
    ax_eps.fill_between(episodes, 0, epsilon, alpha=0.3, color='purple')
    ax_eps.set_xlabel('Episode')
    ax_eps.set_ylabel('Exploration Rate')
    ax_eps.set_title('Exploration -> Exploitation', fontweight='bold')
    ax_eps.set_ylim(0, 1)
    ax_eps.grid(alpha=0.3)

    # Early trajectory
    ax_early = fig.add_subplot(gs[1, 0])
    early_traj = history[0]['trajectory']
    _plot_trajectory(ax_early, early_traj,
                    f"Episode 1: {history[0]['food']} food",
                    'red')

    # Mid trajectory
    ax_mid = fig.add_subplot(gs[1, 1])
    mid_idx = len(history) // 2
    mid_traj = history[mid_idx]['trajectory']
    _plot_trajectory(ax_mid, mid_traj,
                    f"Episode {mid_idx+1}: {history[mid_idx]['food']} food",
                    'orange')

    # Late trajectory
    ax_late = fig.add_subplot(gs[1, 2])
    late_traj = history[-1]['trajectory']
    _plot_trajectory(ax_late, late_traj,
                    f"Episode {len(history)}: {history[-1]['food']} food",
                    'green')

    fig.suptitle('Genesis Brain GROWTH: From Random to Expert', fontsize=16, fontweight='bold')

    return fig


def _plot_trajectory(ax, trajectory, title, color):
    """Plot a single trajectory."""
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title(title, fontweight='bold', color=color)

    # Draw path
    positions = np.array([t['pos'] for t in trajectory])
    for i in range(len(positions)-1):
        alpha = 0.3 + 0.7 * i / len(positions)
        ax.plot([positions[i,0], positions[i+1,0]],
               [positions[i,1], positions[i+1,1]],
               color=color, alpha=alpha, lw=1.5)

    # Start
    ax.scatter(positions[0,0], positions[0,1], c='blue', s=100, marker='s', zorder=10, label='Start')
    # End
    ax.scatter(positions[-1,0], positions[-1,1], c='black', s=100, marker='x', zorder=10, label='End')

    # Food positions (as light markers)
    foods = [t['food'] for t in trajectory[::10]]  # Sample every 10 steps
    for f in foods:
        ax.scatter(f[0], f[1], c='gold', s=50, marker='o', alpha=0.3)

    ax.legend(loc='upper right', fontsize=8)


def create_growth_animation(history: List[Dict], output_path: Path):
    """Create animated GIF showing growth."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_grid, ax_learn, ax_food = axes

    # Grid setup
    ax_grid.set_xlim(-0.5, 9.5)
    ax_grid.set_ylim(-0.5, 9.5)
    ax_grid.set_aspect('equal')
    ax_grid.grid(alpha=0.3)
    ax_grid.set_title('Agent Movement')

    # Learning curve setup
    ax_learn.set_xlim(0, len(history))
    max_food = max(h['food'] for h in history)
    ax_learn.set_ylim(0, max_food + 1)
    ax_learn.set_xlabel('Episode')
    ax_learn.set_ylabel('Food')
    ax_learn.set_title('Learning Progress')
    ax_learn.grid(alpha=0.3)

    # Food bar
    ax_food.set_xlim(0, 1)
    ax_food.set_ylim(0, max_food + 1)
    ax_food.set_title('Current Food')

    # Elements
    agent_dot, = ax_grid.plot([], [], 'bo', markersize=15)
    food_dot, = ax_grid.plot([], [], 'go', markersize=12)
    trail, = ax_grid.plot([], [], 'b-', alpha=0.5, lw=1)

    learn_line, = ax_learn.plot([], [], 'g-', lw=2)
    food_bar = ax_food.barh([0.5], [0], height=0.3, color='green')[0]

    info_text = ax_grid.text(0.02, 0.98, '', transform=ax_grid.transAxes,
                             fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    def animate(frame):
        # Which episode and step?
        steps_per_ep = 20
        ep_idx = frame // steps_per_ep
        step_idx = (frame % steps_per_ep) * 5

        if ep_idx >= len(history):
            return agent_dot, food_dot, trail, learn_line, food_bar, info_text

        traj = history[ep_idx]['trajectory']
        if step_idx >= len(traj):
            step_idx = len(traj) - 1

        t = traj[step_idx]

        # Update grid
        agent_dot.set_data([t['pos'][0]], [t['pos'][1]])
        food_dot.set_data([t['food'][0]], [t['food'][1]])

        # Trail
        start = max(0, step_idx - 15)
        trail_x = [traj[i]['pos'][0] for i in range(start, step_idx+1)]
        trail_y = [traj[i]['pos'][1] for i in range(start, step_idx+1)]
        trail.set_data(trail_x, trail_y)

        # Learning curve
        eps = list(range(ep_idx + 1))
        foods = [history[i]['food'] for i in eps]
        learn_line.set_data(eps, foods)

        # Food bar
        food_bar.set_width(history[ep_idx]['food'] / max(max_food, 1))

        # Info
        info_text.set_text(f"Episode {ep_idx + 1}\n"
                          f"Step {step_idx}\n"
                          f"Food: {history[ep_idx]['food']}\n"
                          f"ε: {t['epsilon']:.2f}")

        return agent_dot, food_dot, trail, learn_line, food_bar, info_text

    n_frames = 20 * min(len(history), 30)  # 30 episodes max
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)

    plt.tight_layout()
    print(f"Saving to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=20))
    print(f"Saved!")
    plt.close()


def main():
    print("=" * 60)
    print("GENESIS BRAIN GROWTH EXPERIMENT")
    print("Watch the agent grow from random to expert!")
    print("=" * 60)
    print()

    history, agent = run_growth_experiment(n_episodes=100)

    print()
    print("=" * 60)
    print("GROWTH SUMMARY")
    print("=" * 60)

    early = np.mean([h['food'] for h in history[:10]])
    late = np.mean([h['food'] for h in history[-10:]])
    print(f"Food collection: {early:.1f} → {late:.1f} ({(late/max(early,0.1)-1)*100:+.0f}%)")
    print(f"Exploration: {history[0]['epsilon']:.2f} → {history[-1]['epsilon']:.2f}")

    output_dir = Path(__file__).parent

    # Static visualization
    fig = create_growth_visualization(history, agent)
    summary_path = output_dir / 'growth_summary.png'
    fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {summary_path}")
    plt.close()

    # Animation
    anim_path = output_dir / 'growth_animation.gif'
    create_growth_animation(history, anim_path)

    print()
    print("Done! Check:")
    print(f"  - {summary_path}")
    print(f"  - {anim_path}")


if __name__ == "__main__":
    main()
