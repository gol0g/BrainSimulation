"""
Genesis Brain Learning Visualization
=====================================
Watch the agent LEARN and GROW in real-time.

Shows:
1. Episode-by-episode performance improvement
2. Learning curves (survival time, food collected)
3. Internal model evolution (precision, memory)
4. Side-by-side comparison: Episode 1 vs Episode N
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.minigrid_benchmark import BenchmarkConfig, BenchmarkEnv, Action


@dataclass
class LearningState:
    """Track what the agent has learned."""
    # Transition model: P(s'|s,a) learned from experience
    transition_counts: np.ndarray = None  # [state_bucket, action, next_bucket]

    # Value estimates per state-action
    Q_values: np.ndarray = None  # [state_bucket, action]

    # Experience replay
    experiences: List[Tuple] = field(default_factory=list)

    # Precision (confidence in predictions)
    precision: float = 0.1  # starts low, grows with experience

    # Memory of dangerous/safe zones
    danger_memory: np.ndarray = None  # [grid_x, grid_y] danger frequency
    food_memory: np.ndarray = None    # [grid_x, grid_y] food frequency

    def __post_init__(self):
        n_buckets = 25  # 5x5 discretization of proximity space
        n_actions = 5
        grid_size = 10

        self.transition_counts = np.ones((n_buckets, n_actions, n_buckets)) * 0.1
        self.Q_values = np.zeros((n_buckets, n_actions))
        self.danger_memory = np.zeros((grid_size, grid_size))
        self.food_memory = np.zeros((grid_size, grid_size))


class GenesisLearningAgent:
    """
    Genesis Brain agent that LEARNS from experience.

    Key mechanisms:
    1. Free Energy minimization for action selection
    2. Precision learning (confidence grows with correct predictions)
    3. Memory formation (dangerous/safe zones)
    4. Transition model learning
    """

    def __init__(self, config: BenchmarkConfig, learning_rate: float = 0.1):
        self.config = config
        self.lr = learning_rate
        self.state = LearningState()

        # FEP parameters
        self.beta = 2.0  # inverse temperature (exploration vs exploitation)

        # Internal state
        self.prev_obs = None
        self.prev_action = None
        self.prev_bucket = None

        # Defense mode (learned hysteresis)
        self.in_defense = False
        self.defense_threshold_on = 0.5   # Will be learned
        self.defense_threshold_off = 0.3  # Will be learned
        self.risk_ema = 0.0

        # Statistics
        self.total_experiences = 0
        self.correct_predictions = 0

    def reset(self):
        """Reset for new episode (keep learned state)."""
        self.prev_obs = None
        self.prev_action = None
        self.prev_bucket = None
        self.in_defense = False
        self.risk_ema = 0.0

    def _obs_to_bucket(self, obs: Dict) -> int:
        """Discretize observation into state bucket."""
        # Use danger_dist and food_dist as 2D state
        danger_dist = min(obs['danger_dist'], 10) / 10  # 0~1
        food_dist = min(obs['food_dist'], 10) / 10      # 0~1

        # 5x5 grid = 25 buckets
        d_idx = min(int(danger_dist * 5), 4)
        f_idx = min(int(food_dist * 5), 4)
        return d_idx * 5 + f_idx

    def _compute_expected_free_energy(self, obs: Dict, action: int) -> float:
        """
        G(a) = Risk + Ambiguity

        Risk = KL[Q(o|a) || P(o)]  -> prefer safe observations
        Ambiguity = H[P(o|s',a)]   -> prefer predictable outcomes
        """
        bucket = self._obs_to_bucket(obs)

        # Risk: based on Q-value (negative is good)
        Q = self.state.Q_values[bucket, action]
        risk = -Q  # Higher Q = lower risk

        # Ambiguity: based on transition uncertainty
        trans_probs = self.state.transition_counts[bucket, action]
        trans_probs = trans_probs / trans_probs.sum()
        entropy = -np.sum(trans_probs * np.log(trans_probs + 1e-10))
        ambiguity = entropy * (1 - self.state.precision)  # More precision = less ambiguity matters

        # Memory-based bias
        agent_pos = obs['agent_pos']
        memory_bias = 0.0

        # Simulate next position
        next_pos = agent_pos.copy()
        if action == Action.LEFT:
            next_pos[0] = max(0, next_pos[0] - 1)
        elif action == Action.RIGHT:
            next_pos[0] = min(9, next_pos[0] + 1)
        elif action == Action.UP:
            next_pos[1] = min(9, next_pos[1] + 1)
        elif action == Action.DOWN:
            next_pos[1] = max(0, next_pos[1] - 1)

        # Check danger memory
        x, y = int(next_pos[0]), int(next_pos[1])
        if 0 <= x < 10 and 0 <= y < 10:
            danger_freq = self.state.danger_memory[x, y]
            food_freq = self.state.food_memory[x, y]
            memory_bias = danger_freq * 0.5 - food_freq * 0.3

        G = risk + ambiguity + memory_bias
        return G

    def act(self, obs: Dict) -> int:
        """Select action by minimizing Expected Free Energy."""
        # Compute G for each action
        G_values = np.array([
            self._compute_expected_free_energy(obs, a)
            for a in range(5)
        ])

        # Risk calculation for defense mode
        danger_dist = obs['danger_dist']
        risk = max(0, (4.0 - danger_dist) / 4.0)
        self.risk_ema = 0.7 * self.risk_ema + 0.3 * risk

        # Learned hysteresis
        if self.in_defense:
            if self.risk_ema < self.defense_threshold_off:
                self.in_defense = False
        else:
            if self.risk_ema > self.defense_threshold_on:
                self.in_defense = True

        # Defense mode: bias toward fleeing
        if self.in_defense:
            flee_dir = obs['agent_pos'] - obs['danger_pos']
            if abs(flee_dir[0]) > abs(flee_dir[1]):
                preferred = Action.RIGHT if flee_dir[0] > 0 else Action.LEFT
            else:
                preferred = Action.UP if flee_dir[1] > 0 else Action.DOWN
            G_values[preferred] -= 1.0  # Boost flee action

        # Softmax selection (exploration with learned precision)
        beta_effective = self.beta * (1 + self.state.precision * 2)  # More confident = more exploitative
        probs = np.exp(-beta_effective * G_values)
        probs = probs / probs.sum()

        # Sample action
        action = np.random.choice(5, p=probs)

        # Store for learning
        self.prev_obs = obs.copy()
        self.prev_action = action
        self.prev_bucket = self._obs_to_bucket(obs)

        return action

    def learn(self, obs: Dict, reward: float, done: bool, info: Dict):
        """Update internal models from experience."""
        if self.prev_obs is None:
            return

        curr_bucket = self._obs_to_bucket(obs)

        # 1. Update transition model
        self.state.transition_counts[self.prev_bucket, self.prev_action, curr_bucket] += 1

        # 2. Update Q-values (TD learning) with boosted learning rate for rewards
        best_next_Q = np.max(self.state.Q_values[curr_bucket]) if not done else 0
        td_target = reward + 0.95 * best_next_Q
        td_error = td_target - self.state.Q_values[self.prev_bucket, self.prev_action]

        # Boost learning from significant events
        effective_lr = self.lr
        if abs(reward) > 0.5:  # Food or danger
            effective_lr = self.lr * 3.0

        self.state.Q_values[self.prev_bucket, self.prev_action] += effective_lr * td_error

        # 3. Update precision (prediction accuracy)
        trans_probs = self.state.transition_counts[self.prev_bucket, self.prev_action]
        trans_probs = trans_probs / trans_probs.sum()
        predicted_bucket = np.argmax(trans_probs)

        if predicted_bucket == curr_bucket:
            self.correct_predictions += 1
        self.total_experiences += 1

        # Precision grows with accuracy
        if self.total_experiences > 10:
            accuracy = self.correct_predictions / self.total_experiences
            self.state.precision = 0.9 * self.state.precision + 0.1 * accuracy

        # 4. Update spatial memory
        agent_pos = obs['agent_pos']
        x, y = int(agent_pos[0]), int(agent_pos[1])

        if info.get('danger_hit', False):
            self.state.danger_memory[x, y] += 1
            # Learn to be more cautious
            self.defense_threshold_on = max(0.25, self.defense_threshold_on - 0.03)

        if info.get('food_collected', False):
            self.state.food_memory[x, y] += 1
            # Learn to be a bit braver when food is found
            self.defense_threshold_on = min(0.5, self.defense_threshold_on + 0.01)

        # 5. Store experience
        self.state.experiences.append((
            self.prev_bucket, self.prev_action, reward, curr_bucket, done
        ))

        # 6. Experience replay (learn from past experiences)
        if len(self.state.experiences) > 50 and self.total_experiences % 10 == 0:
            self._experience_replay(n_samples=10)

        # Keep last 1000 experiences
        if len(self.state.experiences) > 1000:
            self.state.experiences.pop(0)

    def _experience_replay(self, n_samples: int = 10):
        """Learn from random past experiences."""
        if len(self.state.experiences) < n_samples:
            return

        indices = np.random.choice(len(self.state.experiences), n_samples, replace=False)
        for idx in indices:
            bucket, action, reward, next_bucket, done = self.state.experiences[idx]

            best_next_Q = np.max(self.state.Q_values[next_bucket]) if not done else 0
            td_target = reward + 0.95 * best_next_Q
            td_error = td_target - self.state.Q_values[bucket, action]
            self.state.Q_values[bucket, action] += self.lr * 0.5 * td_error  # Lower LR for replay


def run_learning_experiment(
    n_episodes: int = 50,
    p_chase: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], GenesisLearningAgent]:
    """Run learning experiment and collect all data."""

    config = BenchmarkConfig(
        p_chase=p_chase,
        max_steps=300,
        food_energy=0.4,      # More food reward
        energy_decay=0.008    # Slower starvation
    )

    env = BenchmarkEnv(config)
    agent = GenesisLearningAgent(config)

    np.random.seed(seed)

    episode_data = []

    for ep in range(n_episodes):
        obs = env.reset(seed + ep)
        agent.reset()

        ep_reward = 0
        ep_steps = 0
        ep_hits = 0
        ep_food = 0
        trajectory = []

        done = False
        while not done:
            action = agent.act(obs)

            # Store trajectory for visualization
            trajectory.append({
                'pos': obs['agent_pos'].copy(),
                'danger': obs['danger_pos'].copy(),
                'food': obs['food_pos'].copy(),
                'defense': agent.in_defense,
                'risk': agent.risk_ema,
                'precision': agent.state.precision
            })

            next_obs, reward, done, info = env.step(action)
            agent.learn(next_obs, reward, done, info)

            ep_reward += reward
            ep_steps += 1
            if info.get('danger_hit'):
                ep_hits += 1
            if info.get('food_collected'):
                ep_food += 1

            obs = next_obs

        episode_data.append({
            'episode': ep,
            'steps': ep_steps,
            'reward': ep_reward,
            'danger_hits': ep_hits,
            'food_collected': ep_food,
            'precision': agent.state.precision,
            'defense_threshold': agent.defense_threshold_on,
            'trajectory': trajectory,
            'starved': info.get('starved', False)
        })

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}: steps={ep_steps}, food={ep_food}, "
                  f"hits={ep_hits}, precision={agent.state.precision:.2f}")

    return episode_data, agent


def create_learning_visualization(episode_data: List[Dict], agent: GenesisLearningAgent):
    """Create comprehensive learning visualization."""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Learning curves (top row)
    ax_survival = fig.add_subplot(gs[0, 0:2])
    ax_food = fig.add_subplot(gs[0, 2:4])

    # 2. Precision and defense evolution (middle row left)
    ax_precision = fig.add_subplot(gs[1, 0])
    ax_defense = fig.add_subplot(gs[1, 1])

    # 3. Memory maps (middle row right)
    ax_danger_mem = fig.add_subplot(gs[1, 2])
    ax_food_mem = fig.add_subplot(gs[1, 3])

    # 4. Trajectory comparison (bottom row)
    ax_traj_early = fig.add_subplot(gs[2, 0:2])
    ax_traj_late = fig.add_subplot(gs[2, 2:4])

    episodes = [d['episode'] for d in episode_data]

    # === Learning Curves ===
    survival = [d['steps'] for d in episode_data]
    ax_survival.plot(episodes, survival, 'b-', alpha=0.3, label='Raw')
    # Moving average
    window = 5
    survival_ma = np.convolve(survival, np.ones(window)/window, mode='valid')
    ax_survival.plot(episodes[window-1:], survival_ma, 'b-', lw=2, label='Moving Avg')
    ax_survival.set_xlabel('Episode')
    ax_survival.set_ylabel('Survival Steps')
    ax_survival.set_title('Learning: Survival Time', fontweight='bold')
    ax_survival.legend()
    ax_survival.grid(alpha=0.3)

    food = [d['food_collected'] for d in episode_data]
    ax_food.plot(episodes, food, 'g-', alpha=0.3, label='Raw')
    food_ma = np.convolve(food, np.ones(window)/window, mode='valid')
    ax_food.plot(episodes[window-1:], food_ma, 'g-', lw=2, label='Moving Avg')
    ax_food.set_xlabel('Episode')
    ax_food.set_ylabel('Food Collected')
    ax_food.set_title('Learning: Foraging Efficiency', fontweight='bold')
    ax_food.legend()
    ax_food.grid(alpha=0.3)

    # === Precision Evolution ===
    precision = [d['precision'] for d in episode_data]
    ax_precision.plot(episodes, precision, 'purple', lw=2)
    ax_precision.fill_between(episodes, 0, precision, alpha=0.3, color='purple')
    ax_precision.set_xlabel('Episode')
    ax_precision.set_ylabel('Precision')
    ax_precision.set_title('Model Confidence', fontweight='bold')
    ax_precision.set_ylim(0, 1)
    ax_precision.grid(alpha=0.3)

    # === Defense Threshold Evolution ===
    defense_thresh = [d['defense_threshold'] for d in episode_data]
    ax_defense.plot(episodes, defense_thresh, 'red', lw=2)
    ax_defense.axhline(y=0.5, color='gray', linestyle='--', label='Initial')
    ax_defense.set_xlabel('Episode')
    ax_defense.set_ylabel('Defense Threshold')
    ax_defense.set_title('Learned Caution', fontweight='bold')
    ax_defense.legend()
    ax_defense.grid(alpha=0.3)

    # === Memory Maps ===
    danger_mem = agent.state.danger_memory
    im1 = ax_danger_mem.imshow(danger_mem.T, origin='lower', cmap='Reds')
    ax_danger_mem.set_title('Danger Memory', fontweight='bold')
    ax_danger_mem.set_xlabel('X')
    ax_danger_mem.set_ylabel('Y')
    plt.colorbar(im1, ax=ax_danger_mem, label='Frequency')

    food_mem = agent.state.food_memory
    im2 = ax_food_mem.imshow(food_mem.T, origin='lower', cmap='Greens')
    ax_food_mem.set_title('Food Memory', fontweight='bold')
    ax_food_mem.set_xlabel('X')
    ax_food_mem.set_ylabel('Y')
    plt.colorbar(im2, ax=ax_food_mem, label='Frequency')

    # === Trajectory Comparison ===
    def plot_trajectory(ax, traj_data, title):
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_title(title, fontweight='bold')

        # Plot trajectory
        positions = np.array([t['pos'] for t in traj_data])
        colors = ['red' if t['defense'] else 'blue' for t in traj_data]

        for i in range(len(positions)-1):
            ax.plot([positions[i,0], positions[i+1,0]],
                   [positions[i,1], positions[i+1,1]],
                   color=colors[i], alpha=0.5, lw=1)

        # Start and end markers
        ax.scatter(positions[0,0], positions[0,1], c='green', s=100, marker='s', label='Start', zorder=10)
        ax.scatter(positions[-1,0], positions[-1,1], c='black', s=100, marker='x', label='End', zorder=10)
        ax.legend(loc='upper right', fontsize=8)

    # Early episode
    early_ep = episode_data[0]
    plot_trajectory(ax_traj_early, early_ep['trajectory'],
                   f"Episode 1: {early_ep['steps']} steps, {early_ep['food_collected']} food")

    # Late episode
    late_ep = episode_data[-1]
    plot_trajectory(ax_traj_late, late_ep['trajectory'],
                   f"Episode {late_ep['episode']+1}: {late_ep['steps']} steps, {late_ep['food_collected']} food")

    fig.suptitle('Genesis Brain Learning Progress', fontsize=16, fontweight='bold')

    return fig


def create_learning_animation(episode_data: List[Dict], output_path: Path):
    """Create animated GIF showing learning progress."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_grid = axes[0]
    ax_survival = axes[1]
    ax_precision = axes[2]

    # Setup grid
    ax_grid.set_xlim(-0.5, 9.5)
    ax_grid.set_ylim(-0.5, 9.5)
    ax_grid.set_aspect('equal')
    ax_grid.set_title('Agent Movement')
    ax_grid.grid(alpha=0.3)

    # Setup survival plot
    ax_survival.set_xlim(0, len(episode_data))
    max_steps = max(d['steps'] for d in episode_data)
    ax_survival.set_ylim(0, max_steps * 1.1)
    ax_survival.set_xlabel('Episode')
    ax_survival.set_ylabel('Survival Steps')
    ax_survival.set_title('Learning Curve')
    ax_survival.grid(alpha=0.3)

    # Setup precision plot
    ax_precision.set_xlim(0, len(episode_data))
    ax_precision.set_ylim(0, 1)
    ax_precision.set_xlabel('Episode')
    ax_precision.set_ylabel('Precision')
    ax_precision.set_title('Model Confidence')
    ax_precision.grid(alpha=0.3)

    # Plot elements
    agent_dot, = ax_grid.plot([], [], 'bo', markersize=15)
    danger_dot, = ax_grid.plot([], [], 'r^', markersize=12)
    food_dot, = ax_grid.plot([], [], 'go', markersize=10)
    trail, = ax_grid.plot([], [], 'b-', alpha=0.3, lw=1)

    survival_line, = ax_survival.plot([], [], 'b-', lw=2)
    survival_fill = None

    precision_line, = ax_precision.plot([], [], 'purple', lw=2)

    episode_text = ax_grid.text(0.02, 0.98, '', transform=ax_grid.transAxes,
                                 fontsize=12, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        agent_dot.set_data([], [])
        danger_dot.set_data([], [])
        food_dot.set_data([], [])
        trail.set_data([], [])
        survival_line.set_data([], [])
        precision_line.set_data([], [])
        episode_text.set_text('')
        return agent_dot, danger_dot, food_dot, trail, survival_line, precision_line, episode_text

    def animate(frame):
        # Calculate which episode and step we're at
        steps_per_ep = 30  # Show 30 steps per episode in animation
        ep_idx = frame // steps_per_ep
        step_idx = (frame % steps_per_ep) * 2  # Skip every other step for speed

        if ep_idx >= len(episode_data):
            return agent_dot, danger_dot, food_dot, trail, survival_line, precision_line, episode_text

        ep_data = episode_data[ep_idx]
        traj = ep_data['trajectory']

        if step_idx >= len(traj):
            step_idx = len(traj) - 1

        t = traj[step_idx]

        # Update grid
        agent_dot.set_data([t['pos'][0]], [t['pos'][1]])
        danger_dot.set_data([t['danger'][0]], [t['danger'][1]])
        food_dot.set_data([t['food'][0]], [t['food'][1]])

        # Update trail (last 20 positions)
        start = max(0, step_idx - 20)
        trail_x = [traj[i]['pos'][0] for i in range(start, step_idx+1)]
        trail_y = [traj[i]['pos'][1] for i in range(start, step_idx+1)]
        trail.set_data(trail_x, trail_y)

        # Color based on defense mode
        if t['defense']:
            agent_dot.set_color('red')
            trail.set_color('red')
        else:
            agent_dot.set_color('blue')
            trail.set_color('blue')

        # Update learning curves
        eps = list(range(ep_idx + 1))
        survivals = [episode_data[i]['steps'] for i in eps]
        precisions = [episode_data[i]['precision'] for i in eps]

        survival_line.set_data(eps, survivals)
        precision_line.set_data(eps, precisions)

        # Update text
        mode = "DEFENSE" if t['defense'] else "FORAGE"
        episode_text.set_text(f"Episode {ep_idx+1}\nStep {step_idx}\n"
                              f"Mode: {mode}\nPrecision: {t['precision']:.2f}")

        return agent_dot, danger_dot, food_dot, trail, survival_line, precision_line, episode_text

    # Total frames: steps_per_ep * n_episodes
    n_frames = 30 * min(len(episode_data), 20)  # Limit to 20 episodes

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                         interval=50, blit=True)

    plt.tight_layout()

    print(f"Saving learning animation to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=20))
    print(f"Saved: {output_path}")

    plt.close()


def main():
    print("=" * 60)
    print("GENESIS BRAIN LEARNING EXPERIMENT")
    print("Watch the agent learn and grow!")
    print("=" * 60)
    print()

    # Run learning experiment with EASIER environment for visible learning
    print("Running 100 episodes of learning (easier environment)...")
    episode_data, agent = run_learning_experiment(n_episodes=100, p_chase=0.05)

    # Summary
    print()
    print("=" * 60)
    print("LEARNING SUMMARY")
    print("=" * 60)

    early_survival = np.mean([d['steps'] for d in episode_data[:5]])
    late_survival = np.mean([d['steps'] for d in episode_data[-5:]])
    early_food = np.mean([d['food_collected'] for d in episode_data[:5]])
    late_food = np.mean([d['food_collected'] for d in episode_data[-5:]])

    print(f"Survival: {early_survival:.1f} → {late_survival:.1f} steps "
          f"({(late_survival/early_survival - 1)*100:+.1f}%)")
    print(f"Food: {early_food:.1f} → {late_food:.1f} "
          f"({(late_food/max(early_food,0.1) - 1)*100:+.1f}%)")
    print(f"Precision: {episode_data[0]['precision']:.2f} → {episode_data[-1]['precision']:.2f}")
    print(f"Defense threshold: 0.50 → {agent.defense_threshold_on:.2f}")

    # Create visualizations
    output_dir = Path(__file__).parent

    print()
    print("Creating visualizations...")

    # Static summary
    fig = create_learning_visualization(episode_data, agent)
    summary_path = output_dir / 'learning_summary.png'
    fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {summary_path}")
    plt.close()

    # Animated learning
    anim_path = output_dir / 'learning_progress.gif'
    create_learning_animation(episode_data, anim_path)

    print()
    print("Done! Check the output files:")
    print(f"  - {summary_path}")
    print(f"  - {anim_path}")


if __name__ == "__main__":
    main()
