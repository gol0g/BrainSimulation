"""
Curriculum-based Growth System
==============================
Watch an agent GROW through increasingly difficult challenges.

Key components:
1. Curriculum: Easy -> Hard (grid size, hazards, memory requirements)
2. Cumulative Learning: TTC params, habit maps, Q-values persist across episodes
3. Live visualization: See growth in real-time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

plt.ion()


# =============================================================================
# CURRICULUM ENVIRONMENTS
# =============================================================================

@dataclass
class CurriculumLevel:
    """A single difficulty level."""
    name: str
    grid_size: int
    n_hazards: int
    hazard_speed: float      # 0 = static, 1 = fast tracking
    memory_required: bool    # Does the task require memory?
    max_steps: int
    pass_threshold: float    # Success rate to advance


CURRICULUM = [
    CurriculumLevel("1. Baby Steps", 6, 0, 0.0, False, 50, 0.7),
    CurriculumLevel("2. First Hazard", 7, 1, 0.0, False, 60, 0.6),
    CurriculumLevel("3. Moving Threat", 8, 1, 0.3, False, 80, 0.5),
    CurriculumLevel("4. Memory Test", 8, 1, 0.3, True, 100, 0.5),
    CurriculumLevel("5. Dual Hazards", 9, 2, 0.4, True, 120, 0.4),
    CurriculumLevel("6. Expert Mode", 10, 2, 0.5, True, 150, 0.3),
]


@dataclass
class CurriculumEnv:
    """Environment that adapts to curriculum level."""
    level: CurriculumLevel = None
    grid_size: int = 6

    agent_pos: np.ndarray = field(default_factory=lambda: np.array([3, 3]))
    food_pos: np.ndarray = field(default_factory=lambda: np.array([5, 5]))
    hazard_positions: List[np.ndarray] = field(default_factory=list)
    memory_target: Optional[np.ndarray] = None  # For memory tasks

    step_count: int = 0
    food_collected: int = 0
    hazard_hits: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def set_level(self, level: CurriculumLevel):
        self.level = level
        self.grid_size = level.grid_size

    def reset(self, seed: int = None) -> Dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.food_pos = self._random_pos(avoid=[self.agent_pos])

        # Setup hazards
        self.hazard_positions = []
        avoid = [self.agent_pos, self.food_pos]
        for _ in range(self.level.n_hazards):
            pos = self._random_pos(avoid=avoid, min_dist=3)
            self.hazard_positions.append(pos)
            avoid.append(pos)

        # Memory target (shown briefly at start, then hidden)
        if self.level.memory_required:
            self.memory_target = self._random_pos(avoid=avoid)
        else:
            self.memory_target = None

        self.step_count = 0
        self.food_collected = 0
        self.hazard_hits = 0

        return self._get_obs()

    def _random_pos(self, avoid=None, min_dist=2):
        for _ in range(100):
            pos = self.rng.integers(1, self.grid_size - 1, 2)
            if avoid:
                ok = all(np.linalg.norm(pos - a) >= min_dist for a in avoid)
                if ok:
                    return pos
            else:
                return pos
        return self.rng.integers(1, self.grid_size - 1, 2)

    def _get_obs(self) -> Dict:
        # Compute distances
        food_diff = self.food_pos - self.agent_pos

        hazard_dists = []
        hazard_diffs = []
        for h in self.hazard_positions:
            hazard_dists.append(np.linalg.norm(self.agent_pos - h))
            hazard_diffs.append(h - self.agent_pos)

        min_hazard_dist = min(hazard_dists) if hazard_dists else 999
        closest_hazard_diff = hazard_diffs[np.argmin(hazard_dists)] if hazard_diffs else np.array([0, 0])

        return {
            'agent_pos': self.agent_pos.copy(),
            'food_pos': self.food_pos.copy(),
            'food_dx': food_diff[0],
            'food_dy': food_diff[1],
            'food_dist': np.linalg.norm(food_diff),
            'hazard_dist': min_hazard_dist,
            'hazard_dx': closest_hazard_diff[0],
            'hazard_dy': closest_hazard_diff[1],
            'hazard_positions': [h.copy() for h in self.hazard_positions],
            'memory_target': self.memory_target.copy() if self.memory_target is not None else None,
            'step': self.step_count,
            'show_memory': self.step_count < 10,  # Show memory target for first 10 steps
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        self.step_count += 1

        # Move agent
        dx, dy = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)][action]
        new_pos = self.agent_pos + np.array([dx, dy])
        self.agent_pos = np.clip(new_pos, 0, self.grid_size - 1)

        # Move hazards (lazy tracking)
        for i, h in enumerate(self.hazard_positions):
            if self.rng.random() < self.level.hazard_speed:
                direction = np.sign(self.agent_pos - h)
                self.hazard_positions[i] = np.clip(h + direction, 0, self.grid_size - 1)

        # Check outcomes
        reward = -0.01  # Step cost
        info = {'food': False, 'hazard': False, 'memory_bonus': False, 'success': False}

        # Food collection
        if np.linalg.norm(self.agent_pos - self.food_pos) < 1.5:
            reward += 1.0
            self.food_collected += 1
            info['food'] = True
            self.food_pos = self._random_pos(avoid=[self.agent_pos] + self.hazard_positions)

        # Hazard collision
        for h in self.hazard_positions:
            if np.linalg.norm(self.agent_pos - h) < 1.5:
                reward -= 0.5
                self.hazard_hits += 1
                info['hazard'] = True

        # Memory bonus
        if self.memory_target is not None and np.linalg.norm(self.agent_pos - self.memory_target) < 1.5:
            reward += 2.0
            info['memory_bonus'] = True
            self.memory_target = None  # Can only collect once

        # Check termination
        done = self.step_count >= self.level.max_steps

        # Success criteria: collected enough food
        if self.food_collected >= 3:
            info['success'] = True

        return self._get_obs(), reward, done, info


# =============================================================================
# CUMULATIVE LEARNING AGENT
# =============================================================================

@dataclass
class CumulativeMemory:
    """Persistent memory that accumulates across episodes."""
    # Q-learning table (survives across episodes)
    Q: np.ndarray = None

    # Habit map: where have we been hit by hazards?
    danger_map: np.ndarray = None
    safe_map: np.ndarray = None

    # TTC parameters (learned from experience)
    ttc_threshold: float = 3.0
    defense_threshold: float = 0.5

    # Statistics
    total_episodes: int = 0
    total_food: int = 0
    total_hazard_hits: int = 0

    def initialize(self, max_grid: int = 12):
        self.Q = np.zeros((max_grid, max_grid, 5, 5, 5))  # [food_dx, food_dy, hazard_dx, hazard_dy, action]
        self.danger_map = np.zeros((max_grid, max_grid))
        self.safe_map = np.zeros((max_grid, max_grid))


class CurriculumAgent:
    """Agent with cumulative learning across episodes."""

    def __init__(self):
        self.memory = CumulativeMemory()
        self.memory.initialize()

        # Learning parameters
        self.epsilon = 0.5
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.lr = 0.2
        self.gamma = 0.95

        # Episode state
        self.in_defense = False
        self.prev_hazard_dist = None
        self.approach_streak = 0

    def _state_indices(self, obs: Dict) -> Tuple:
        """Convert observation to state indices for Q-table."""
        fdx = int(np.clip(obs['food_dx'], -2, 2) + 2)
        fdy = int(np.clip(obs['food_dy'], -2, 2) + 2)
        hdx = int(np.clip(obs['hazard_dx'], -2, 2) + 2)
        hdy = int(np.clip(obs['hazard_dy'], -2, 2) + 2)
        return fdx, fdy, hdx, hdy

    def act(self, obs: Dict) -> int:
        # TTC-based defense mode
        hazard_dist = obs['hazard_dist']

        # Track approach
        if self.prev_hazard_dist is not None:
            closing = self.prev_hazard_dist - hazard_dist
            if closing > 0.1:
                self.approach_streak += 1
            else:
                self.approach_streak = 0
        self.prev_hazard_dist = hazard_dist

        # Defense trigger (using learned threshold)
        risk = max(0, (self.memory.ttc_threshold - hazard_dist) / self.memory.ttc_threshold)

        if self.in_defense:
            if risk < self.memory.defense_threshold * 0.6:
                self.in_defense = False
        else:
            if risk > self.memory.defense_threshold or self.approach_streak >= 2:
                self.in_defense = True

        # Action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(5)
        else:
            fdx, fdy, hdx, hdy = self._state_indices(obs)
            q_values = self.memory.Q[fdx, fdy, hdx, hdy].copy()

            # Apply habit map bias
            pos = obs['agent_pos'].astype(int)
            for a in range(4):  # Movement actions
                dx, dy = [(-1, 0), (1, 0), (0, 1), (0, -1)][a]
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < 12 and 0 <= ny < 12:
                    q_values[a] -= self.memory.danger_map[nx, ny] * 0.3
                    q_values[a] += self.memory.safe_map[nx, ny] * 0.1

            # Defense mode: bias toward fleeing
            if self.in_defense and hazard_dist < 5:
                flee_action = self._get_flee_action(obs)
                q_values[flee_action] += 0.5

            action = np.argmax(q_values)

        return action

    def _get_flee_action(self, obs: Dict) -> int:
        """Get action that moves away from hazard."""
        hdx, hdy = obs['hazard_dx'], obs['hazard_dy']
        if abs(hdx) > abs(hdy):
            return 0 if hdx > 0 else 1  # LEFT or RIGHT
        else:
            return 3 if hdy > 0 else 2  # DOWN or UP

    def learn(self, obs: Dict, action: int, reward: float, next_obs: Dict, done: bool, info: Dict):
        """Update Q-table and habit maps."""
        # Q-learning update
        s = self._state_indices(obs)
        ns = self._state_indices(next_obs)

        best_next = 0 if done else np.max(self.memory.Q[ns])
        target = reward + self.gamma * best_next
        self.memory.Q[s][action] += self.lr * (target - self.memory.Q[s][action])

        # Update habit maps
        pos = obs['agent_pos'].astype(int)
        x, y = min(pos[0], 11), min(pos[1], 11)

        if info.get('hazard'):
            self.memory.danger_map[x, y] += 1.0
            # Adapt defense threshold
            self.memory.defense_threshold = max(0.3, self.memory.defense_threshold - 0.02)

        if info.get('food'):
            self.memory.safe_map[x, y] += 0.5

        # Decay maps slowly (forgetting)
        self.memory.danger_map *= 0.999
        self.memory.safe_map *= 0.999

    def end_episode(self, food: int, hazard_hits: int):
        """Called at end of each episode."""
        self.memory.total_episodes += 1
        self.memory.total_food += food
        self.memory.total_hazard_hits += hazard_hits

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Reset episode state
        self.in_defense = False
        self.prev_hazard_dist = None
        self.approach_streak = 0


# =============================================================================
# LIVE VISUALIZATION
# =============================================================================

def run_curriculum_live():
    """Run curriculum with live visualization."""

    print("=" * 60)
    print("CURRICULUM GROWTH - Watch the agent master each level!")
    print("Close the window to stop.")
    print("=" * 60)

    env = CurriculumEnv()
    agent = CurriculumAgent()

    # Setup figure
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax_grid = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_level = fig.add_subplot(gs[0, 2])
    ax_danger = fig.add_subplot(gs[1, 0])
    ax_params = fig.add_subplot(gs[1, 1])
    ax_stats = fig.add_subplot(gs[1, 2])

    # Grid setup
    ax_grid.set_aspect('equal')
    ax_grid.grid(alpha=0.3)

    agent_circle = Circle((3, 3), 0.4, color='blue', zorder=10)
    food_circle = Circle((5, 5), 0.35, color='green', zorder=5)
    ax_grid.add_patch(agent_circle)
    ax_grid.add_patch(food_circle)

    hazard_circles = [Circle((0, 0), 0.35, color='red', zorder=5) for _ in range(3)]
    for h in hazard_circles:
        h.set_visible(False)
        ax_grid.add_patch(h)

    memory_circle = Circle((0, 0), 0.3, color='gold', zorder=5, alpha=0.7)
    memory_circle.set_visible(False)
    ax_grid.add_patch(memory_circle)

    defense_ring = Circle((3, 3), 0.55, fill=False, color='cyan', linewidth=3, zorder=9)
    defense_ring.set_visible(False)
    ax_grid.add_patch(defense_ring)

    # Learning curve
    ax_curve.set_xlabel('Episode')
    ax_curve.set_ylabel('Food Collected')
    ax_curve.set_title('Learning Progress')
    ax_curve.grid(alpha=0.3)
    learn_line, = ax_curve.plot([], [], 'g-', lw=2)

    # Level progress
    ax_level.set_xlim(0, len(CURRICULUM))
    ax_level.set_ylim(0, 1)
    ax_level.set_xlabel('Level')
    ax_level.set_ylabel('Success Rate')
    ax_level.set_title('Curriculum Progress')
    ax_level.grid(alpha=0.3)
    level_bars = ax_level.bar(range(len(CURRICULUM)), [0]*len(CURRICULUM),
                              color=['gray']*len(CURRICULUM), alpha=0.7)

    # Danger map
    ax_danger.set_title('Danger Memory')
    danger_img = ax_danger.imshow(np.zeros((12, 12)), cmap='Reds', vmin=0, vmax=5)

    # Parameters
    ax_params.axis('off')
    params_text = ax_params.text(0.1, 0.9, '', transform=ax_params.transAxes,
                                  fontsize=11, verticalalignment='top', family='monospace')

    # Stats
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.1, 0.9, '', transform=ax_stats.transAxes,
                                fontsize=11, verticalalignment='top', family='monospace')

    info_text = ax_grid.text(0.02, 0.98, '', transform=ax_grid.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Tracking
    current_level = 0
    level_successes = [0] * len(CURRICULUM)
    level_attempts = [0] * len(CURRICULUM)
    all_foods = []

    episode = 0
    max_episodes = 500

    try:
        while episode < max_episodes and plt.fignum_exists(fig.number):
            level = CURRICULUM[current_level]
            env.set_level(level)
            obs = env.reset(seed=episode)

            # Update grid limits
            ax_grid.set_xlim(-0.5, level.grid_size - 0.5)
            ax_grid.set_ylim(-0.5, level.grid_size - 0.5)
            ax_grid.set_title(f'{level.name}')

            done = False
            while not done and plt.fignum_exists(fig.number):
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                agent.learn(obs, action, reward, next_obs, done, info)
                obs = next_obs

                # Update display every 3 steps
                if env.step_count % 3 == 0:
                    # Update positions
                    agent_circle.center = obs['agent_pos']
                    food_circle.center = obs['food_pos']

                    # Update hazards
                    for i, hc in enumerate(hazard_circles):
                        if i < len(obs['hazard_positions']):
                            hc.center = obs['hazard_positions'][i]
                            hc.set_visible(True)
                        else:
                            hc.set_visible(False)

                    # Memory target
                    if obs['memory_target'] is not None and obs['show_memory']:
                        memory_circle.center = obs['memory_target']
                        memory_circle.set_visible(True)
                    else:
                        memory_circle.set_visible(False)

                    # Defense ring
                    defense_ring.center = obs['agent_pos']
                    defense_ring.set_visible(agent.in_defense)

                    # Info text
                    mode = "DEFENSE" if agent.in_defense else "FORAGE"
                    info_text.set_text(f"Episode {episode+1}\n"
                                      f"Step {env.step_count}/{level.max_steps}\n"
                                      f"Food: {env.food_collected}\n"
                                      f"Mode: {mode}")

                    plt.pause(0.02)

            # End of episode
            agent.end_episode(env.food_collected, env.hazard_hits)
            all_foods.append(env.food_collected)

            # Track level success
            level_attempts[current_level] += 1
            if env.food_collected >= 3:  # Success threshold
                level_successes[current_level] += 1

            # Update level bars
            for i, bar in enumerate(level_bars):
                if level_attempts[i] > 0:
                    rate = level_successes[i] / level_attempts[i]
                    bar.set_height(rate)
                    bar.set_color('green' if i < current_level else ('blue' if i == current_level else 'gray'))

            # Check level advancement
            if level_attempts[current_level] >= 10:
                success_rate = level_successes[current_level] / level_attempts[current_level]
                if success_rate >= level.pass_threshold and current_level < len(CURRICULUM) - 1:
                    print(f"\n*** LEVEL UP! {CURRICULUM[current_level].name} -> {CURRICULUM[current_level+1].name} ***\n")
                    current_level += 1

            # Update learning curve
            learn_line.set_data(range(len(all_foods)), all_foods)
            ax_curve.set_xlim(0, max(10, len(all_foods)))
            ax_curve.set_ylim(0, max(all_foods) + 1)

            # Update danger map
            danger_img.set_data(agent.memory.danger_map[:level.grid_size, :level.grid_size].T)

            # Update params text
            params_text.set_text(
                f"LEARNED PARAMETERS:\n"
                f"------------------\n"
                f"TTC threshold: {agent.memory.ttc_threshold:.2f}\n"
                f"Defense thresh: {agent.memory.defense_threshold:.2f}\n"
                f"Exploration: {agent.epsilon:.3f}\n"
                f"\n"
                f"CURRENT LEVEL:\n"
                f"--------------\n"
                f"Level: {current_level + 1}/{len(CURRICULUM)}\n"
                f"Grid: {level.grid_size}x{level.grid_size}\n"
                f"Hazards: {level.n_hazards}\n"
                f"Speed: {level.hazard_speed:.1f}"
            )

            # Update stats text
            stats_text.set_text(
                f"CUMULATIVE STATS:\n"
                f"-----------------\n"
                f"Total episodes: {agent.memory.total_episodes}\n"
                f"Total food: {agent.memory.total_food}\n"
                f"Total hits: {agent.memory.total_hazard_hits}\n"
                f"\n"
                f"RECENT PERFORMANCE:\n"
                f"-------------------\n"
                f"Last 10 avg: {np.mean(all_foods[-10:]):.1f}\n"
                f"Level success: {level_successes[current_level]}/{level_attempts[current_level]}"
            )

            episode += 1

            if episode % 20 == 0:
                avg = np.mean(all_foods[-20:]) if len(all_foods) >= 20 else np.mean(all_foods)
                print(f"Episode {episode}: Level {current_level+1}, avg food = {avg:.1f}, "
                      f"epsilon = {agent.epsilon:.2f}")

        print()
        print("=" * 60)
        print("TRAINING COMPLETE!")
        print(f"Final Level: {current_level + 1}/{len(CURRICULUM)} - {CURRICULUM[current_level].name}")
        print(f"Total Food: {agent.memory.total_food}")
        print("=" * 60)

        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    run_curriculum_live()
