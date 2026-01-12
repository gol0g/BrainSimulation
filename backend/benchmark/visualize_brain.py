"""
Brain Activity Visualization - See the agent "thinking"

Shows:
1. Goal/Risk/Action vectors as arrows
2. Defense mode ring + TTC approach indicator
3. Hysteresis footprints (defense trail)
4. Internal state panel (neuron-like activations)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.minigrid_benchmark import (
    BenchmarkConfig, BenchmarkEnv, Action
)


@dataclass
class BrainState:
    """Snapshot of agent's internal state."""
    # Position
    agent_pos: np.ndarray
    danger_pos: np.ndarray
    food_pos: np.ndarray

    # Vectors
    goal_vec: np.ndarray      # Direction to food
    risk_vec: np.ndarray      # Direction away from danger
    action_vec: np.ndarray    # Actual movement

    # Mode
    defensive: bool
    risk_raw: float
    risk_filtered: float      # After hysteresis

    # TTC
    ttc: float
    approach_streak: int
    ttc_triggered: bool

    # Resources
    energy: float
    step: int


class BrainAgent:
    """Agent that exposes internal state for visualization."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.in_defense = False
        self.prev_danger_dist = None
        self.approach_streak = 0
        self.risk_ema = 0.0
        self.ttc_triggered = False

    def act(self, obs: Dict) -> tuple:
        """Returns (action, brain_state)."""
        agent_pos = obs['agent_pos']
        danger_pos = obs['danger_pos']
        food_pos = obs['food_pos']
        danger_dist = obs['danger_dist']
        energy = obs['energy']

        # Goal vector (toward food)
        goal_vec = food_pos - agent_pos
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm > 0.1:
            goal_vec = goal_vec / goal_norm

        # Risk vector (away from danger)
        risk_vec = agent_pos - danger_pos
        risk_norm = np.linalg.norm(risk_vec)
        if risk_norm > 0.1:
            risk_vec = risk_vec / risk_norm

        # Risk calculation
        risk_raw = max(0, (4.0 - danger_dist) / 4.0)
        self.risk_ema = 0.7 * self.risk_ema + 0.3 * risk_raw

        # TTC calculation
        ttc = float('inf')
        closing = 0
        if self.prev_danger_dist is not None:
            closing = self.prev_danger_dist - danger_dist
            if closing > 0.1:
                self.approach_streak += 1
                ttc = danger_dist / closing
            else:
                self.approach_streak = 0
        self.prev_danger_dist = danger_dist

        # TTC trigger
        self.ttc_triggered = (
            danger_dist < self.config.ttc_threshold and
            closing > 0 and
            self.approach_streak >= self.config.approach_streak_min
        )

        # Hysteresis defense mode
        if self.in_defense:
            if self.risk_ema < self.config.risk_threshold_off and not self.ttc_triggered:
                self.in_defense = False
        else:
            if self.risk_ema > self.config.risk_threshold_on or self.ttc_triggered:
                self.in_defense = True

        # Action selection
        if self.in_defense:
            action_vec = risk_vec  # Flee
        else:
            action_vec = goal_vec  # Pursue food

        # Convert to discrete action
        action = self._vec_to_action(action_vec)

        # Create brain state
        state = BrainState(
            agent_pos=agent_pos.copy(),
            danger_pos=danger_pos.copy(),
            food_pos=food_pos.copy(),
            goal_vec=goal_vec,
            risk_vec=risk_vec,
            action_vec=action_vec,
            defensive=self.in_defense,
            risk_raw=risk_raw,
            risk_filtered=self.risk_ema,
            ttc=min(ttc, 20),
            approach_streak=self.approach_streak,
            ttc_triggered=self.ttc_triggered,
            energy=energy,
            step=obs.get('step', 0)
        )

        return action, state

    def _vec_to_action(self, vec: np.ndarray) -> int:
        if abs(vec[0]) > abs(vec[1]):
            return Action.RIGHT if vec[0] > 0 else Action.LEFT
        elif abs(vec[1]) > 0:
            return Action.UP if vec[1] > 0 else Action.DOWN
        return Action.STAY


def create_brain_visualization(p_chase: float = 0.2, max_frames: int = 120, seed: int = 42):
    """Create visualization showing agent's internal brain state."""

    config = BenchmarkConfig(
        p_chase=p_chase,
        max_steps=max_frames,
        food_energy=0.3,
        energy_decay=0.012
    )

    env = BenchmarkEnv(config)
    agent = BrainAgent(config)

    obs = env.reset(seed)
    agent.reset()

    # Collect all states
    states: List[BrainState] = []
    done = False
    step = 0

    while not done and step < max_frames:
        obs['step'] = step
        action, brain_state = agent.act(obs)
        states.append(brain_state)
        obs, reward, done, info = env.step(action)
        step += 1

    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[1, 1])

    ax_main = fig.add_subplot(gs[:, 0])      # Main grid view
    ax_vectors = fig.add_subplot(gs[0, 1])   # Vector compass
    ax_trail = fig.add_subplot(gs[1, 1])     # Defense trail
    ax_neurons = fig.add_subplot(gs[:, 2])   # Neuron panel

    fig.suptitle(f'Brain Activity Visualization | Tracking: p={p_chase}', fontsize=14, fontweight='bold')

    # Setup main grid
    ax_main.set_xlim(-0.5, config.grid_size - 0.5)
    ax_main.set_ylim(-0.5, config.grid_size - 0.5)
    ax_main.set_aspect('equal')
    ax_main.set_title('Environment', fontsize=11)
    ax_main.grid(True, alpha=0.3)

    # Setup vector compass
    ax_vectors.set_xlim(-1.5, 1.5)
    ax_vectors.set_ylim(-1.5, 1.5)
    ax_vectors.set_aspect('equal')
    ax_vectors.set_title('Decision Vectors', fontsize=11)
    ax_vectors.axhline(0, color='gray', linewidth=0.5)
    ax_vectors.axvline(0, color='gray', linewidth=0.5)
    ax_vectors.set_xticks([])
    ax_vectors.set_yticks([])

    # Setup trail view
    ax_trail.set_xlim(-0.5, config.grid_size - 0.5)
    ax_trail.set_ylim(-0.5, config.grid_size - 0.5)
    ax_trail.set_aspect('equal')
    ax_trail.set_title('Defense Trail (Hysteresis)', fontsize=11)
    ax_trail.grid(True, alpha=0.3)

    # Setup neuron panel
    ax_neurons.set_xlim(0, 1)
    ax_neurons.set_ylim(0, 1)
    ax_neurons.set_title('Internal States', fontsize=11)
    ax_neurons.set_xticks([])
    ax_neurons.set_yticks([])

    # Create plot elements
    # Main view
    agent_circle = plt.Circle((0, 0), 0.4, color='#2ecc71', zorder=10)
    danger_circle = plt.Circle((0, 0), 0.4, color='#e74c3c', zorder=5)
    food_circle = plt.Circle((0, 0), 0.35, color='#f39c12', zorder=5)
    defense_ring = plt.Circle((0, 0), 0.55, fill=False, color='blue', linewidth=4, zorder=9)
    ttc_ring = plt.Circle((0, 0), 0.65, fill=False, color='red', linewidth=2, linestyle='--', zorder=8)

    ax_main.add_patch(agent_circle)
    ax_main.add_patch(danger_circle)
    ax_main.add_patch(food_circle)
    ax_main.add_patch(defense_ring)
    ax_main.add_patch(ttc_ring)

    # Arrows on main view
    arrow_action = ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle='->', color='black', lw=2))
    arrow_goal = ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5, alpha=0.5))
    arrow_risk = ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, alpha=0.5))

    # Vector compass arrows
    goal_arrow = ax_vectors.annotate('', xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle='->', color='#f39c12', lw=3))
    risk_arrow = ax_vectors.annotate('', xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    action_arrow = ax_vectors.annotate('', xy=(0, 0), xytext=(0, 0),
                                        arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=4))

    # Legend for compass
    ax_vectors.plot([], [], color='#f39c12', lw=3, label='Goal')
    ax_vectors.plot([], [], color='#e74c3c', lw=3, label='Flee')
    ax_vectors.plot([], [], color='#2ecc71', lw=4, label='Action')
    ax_vectors.legend(loc='upper right', fontsize=8)

    # Trail scatter
    trail_scatter = ax_trail.scatter([], [], c=[], cmap='RdYlGn_r', s=50, alpha=0.7)

    # Status text
    status_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Neuron state labels and bars
    neuron_labels = ['Risk Raw', 'Risk Filtered', 'Defense Mode', 'TTC Signal',
                     'Approach Streak', 'Energy', 'TTC Triggered', 'Goal Pull']
    neuron_positions = np.linspace(0.9, 0.1, len(neuron_labels))
    neuron_bars = []

    for i, (label, y) in enumerate(zip(neuron_labels, neuron_positions)):
        ax_neurons.text(0.05, y, label, fontsize=9, va='center')
        bar = patches.Rectangle((0.45, y - 0.03), 0, 0.05, color='#3498db')
        ax_neurons.add_patch(bar)
        neuron_bars.append(bar)

    # Trail history
    trail_positions = []
    trail_defense = []

    def update(frame):
        if frame >= len(states):
            return []

        s = states[frame]

        # Update positions
        agent_circle.center = s.agent_pos
        danger_circle.center = s.danger_pos
        food_circle.center = s.food_pos
        defense_ring.center = s.agent_pos
        ttc_ring.center = s.agent_pos

        # Defense ring visibility
        if s.defensive:
            defense_ring.set_visible(True)
            defense_ring.set_color('blue')
            agent_circle.set_color('#3498db')
        else:
            defense_ring.set_visible(False)
            agent_circle.set_color('#2ecc71')

        # TTC ring (approach warning)
        if s.approach_streak >= 1 and s.ttc < 15:
            ttc_ring.set_visible(True)
            alpha = min(1.0, s.approach_streak / 3.0)
            ttc_ring.set_alpha(alpha)
        else:
            ttc_ring.set_visible(False)

        # Arrows on main view
        scale = 1.5
        arrow_action.xy = s.agent_pos + s.action_vec * scale
        arrow_action.xyann = s.agent_pos

        arrow_goal.xy = s.agent_pos + s.goal_vec * scale * 0.8
        arrow_goal.xyann = s.agent_pos

        arrow_risk.xy = s.agent_pos + s.risk_vec * scale * 0.6
        arrow_risk.xyann = s.agent_pos

        # Vector compass
        goal_arrow.xy = s.goal_vec
        goal_arrow.xyann = (0, 0)
        risk_arrow.xy = s.risk_vec * 0.8
        risk_arrow.xyann = (0, 0)
        action_arrow.xy = s.action_vec * 1.2
        action_arrow.xyann = (0, 0)

        # Trail
        trail_positions.append(s.agent_pos.copy())
        trail_defense.append(1 if s.defensive else 0)

        # Keep last 30 steps
        if len(trail_positions) > 30:
            trail_positions.pop(0)
            trail_defense.pop(0)

        if trail_positions:
            positions = np.array(trail_positions)
            trail_scatter.set_offsets(positions)
            trail_scatter.set_array(np.array(trail_defense))

        # Status text
        mode = "DEFENSE" if s.defensive else "FORAGE"
        ttc_str = f"{s.ttc:.1f}" if s.ttc < 20 else "safe"
        status_text.set_text(
            f"Step: {s.step}\n"
            f"Mode: {mode}\n"
            f"Risk: {s.risk_filtered:.2f}\n"
            f"TTC: {ttc_str}\n"
            f"Approach: {s.approach_streak}"
        )

        # Neuron bars
        neuron_values = [
            s.risk_raw,
            s.risk_filtered,
            1.0 if s.defensive else 0.0,
            min(1.0, 10.0 / max(s.ttc, 1)),  # TTC signal (higher when closer)
            min(1.0, s.approach_streak / 4.0),
            s.energy,
            1.0 if s.ttc_triggered else 0.0,
            1.0 - s.risk_filtered  # Goal pull (inverse of risk)
        ]

        for bar, val in zip(neuron_bars, neuron_values):
            bar.set_width(val * 0.5)
            if val > 0.7:
                bar.set_color('#e74c3c')
            elif val > 0.4:
                bar.set_color('#f39c12')
            else:
                bar.set_color('#2ecc71')

        return [agent_circle, danger_circle, food_circle, defense_ring, ttc_ring,
                trail_scatter, status_text] + neuron_bars

    anim = FuncAnimation(fig, update, frames=len(states), interval=150, blit=False)

    plt.tight_layout()
    return fig, anim, states


def main():
    print("Creating brain activity visualization...")

    # Create for strong tracking (where TTC matters most)
    fig, anim, states = create_brain_visualization(p_chase=0.2, max_frames=100, seed=42)

    output_path = Path(__file__).parent / 'brain_activity.gif'
    print(f"Saving to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=7))
    print(f"Saved: {output_path}")

    # Also create for DIP zone
    fig2, anim2, states2 = create_brain_visualization(p_chase=0.15, max_frames=100, seed=42)
    output_path2 = Path(__file__).parent / 'brain_activity_dip.gif'
    print(f"Saving DIP zone version to {output_path2}...")
    anim2.save(output_path2, writer=PillowWriter(fps=7))
    print(f"Saved: {output_path2}")

    plt.close('all')
    print("\nDone! Check the GIF files.")


if __name__ == "__main__":
    main()
