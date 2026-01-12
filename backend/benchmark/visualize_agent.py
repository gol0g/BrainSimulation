"""
Visualize Agent Movement - Watch the defense stack in action
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.minigrid_benchmark import (
    BenchmarkConfig, BenchmarkEnv,
    BaseAgent, RFHysteresisAgent, RFTTCAgent, Action
)


def create_animation(agent_type='RF+TTC', p_chase=0.2, max_frames=150, seed=42):
    """Create animation of agent navigating the environment."""

    # Setup
    config = BenchmarkConfig(
        p_chase=p_chase,
        max_steps=max_frames,
        food_energy=0.3,
        energy_decay=0.012
    )

    env = BenchmarkEnv(config)

    if agent_type == 'BASE':
        agent = BaseAgent(config)
        color = '#e74c3c'
    elif agent_type == 'RF+Hyst':
        agent = RFHysteresisAgent(config)
        color = '#3498db'
    else:  # RF+TTC
        agent = RFTTCAgent(config)
        color = '#2ecc71'

    # Initialize
    obs = env.reset(seed)
    if hasattr(agent, 'reset'):
        agent.reset()

    # Storage for animation
    frames = []

    # Run episode and store states
    done = False
    step = 0
    while not done and step < max_frames:
        action = agent.act(obs)

        # Store frame data
        frame_data = {
            'agent_pos': obs['agent_pos'].copy(),
            'danger_pos': obs['danger_pos'].copy(),
            'food_pos': obs['food_pos'].copy(),
            'energy': obs['energy'],
            'in_defense': getattr(agent, 'in_defense', False),
            'step': step
        }
        frames.append(frame_data)

        obs, reward, done, info = env.step(action)
        step += 1

    # Create figure
    fig, (ax_main, ax_energy) = plt.subplots(1, 2, figsize=(12, 6),
                                              gridspec_kw={'width_ratios': [3, 1]})

    fig.suptitle(f'Agent: {agent_type} | Tracking: p={p_chase}', fontsize=14, fontweight='bold')

    # Main grid
    ax_main.set_xlim(-0.5, config.grid_size - 0.5)
    ax_main.set_ylim(-0.5, config.grid_size - 0.5)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y')

    # Grid lines
    for i in range(config.grid_size + 1):
        ax_main.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax_main.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Energy bar setup
    ax_energy.set_xlim(0, 1)
    ax_energy.set_ylim(0, 1)
    ax_energy.set_title('Energy')
    ax_energy.set_xticks([])

    # Create plot elements
    agent_circle = plt.Circle((0, 0), 0.35, color=color, zorder=10)
    danger_circle = plt.Circle((0, 0), 0.35, color='#e74c3c', zorder=5)
    food_circle = plt.Circle((0, 0), 0.3, color='#f39c12', zorder=5)
    defense_ring = plt.Circle((0, 0), 0.45, fill=False, color='blue', linewidth=3, zorder=9)

    ax_main.add_patch(agent_circle)
    ax_main.add_patch(danger_circle)
    ax_main.add_patch(food_circle)
    ax_main.add_patch(defense_ring)

    # Energy bar
    energy_bar = ax_energy.barh([0.5], [1.0], height=0.3, color='#2ecc71')[0]

    # Text elements
    status_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                                fontsize=11, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Legend
    legend_elements = [
        plt.Circle((0, 0), 0.1, color=color, label=f'Agent ({agent_type})'),
        plt.Circle((0, 0), 0.1, color='#e74c3c', label='Danger'),
        plt.Circle((0, 0), 0.1, color='#f39c12', label='Food'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def init():
        return agent_circle, danger_circle, food_circle, defense_ring, energy_bar, status_text

    def animate(i):
        if i >= len(frames):
            return agent_circle, danger_circle, food_circle, defense_ring, energy_bar, status_text

        frame = frames[i]

        # Update positions
        agent_circle.center = frame['agent_pos']
        danger_circle.center = frame['danger_pos']
        food_circle.center = frame['food_pos']
        defense_ring.center = frame['agent_pos']

        # Defense mode visualization
        if frame['in_defense']:
            defense_ring.set_visible(True)
            agent_circle.set_edgecolor('blue')
            agent_circle.set_linewidth(2)
        else:
            defense_ring.set_visible(False)
            agent_circle.set_edgecolor('black')
            agent_circle.set_linewidth(1)

        # Energy bar
        energy_bar.set_width(frame['energy'])
        if frame['energy'] > 0.5:
            energy_bar.set_color('#2ecc71')
        elif frame['energy'] > 0.25:
            energy_bar.set_color('#f39c12')
        else:
            energy_bar.set_color('#e74c3c')

        # Status text
        danger_dist = np.linalg.norm(frame['agent_pos'] - frame['danger_pos'])
        mode = "DEFENSE" if frame['in_defense'] else "FORAGE"
        status_text.set_text(f"Step: {frame['step']}\n"
                            f"Mode: {mode}\n"
                            f"Danger dist: {danger_dist:.1f}\n"
                            f"Energy: {frame['energy']:.0%}")

        return agent_circle, danger_circle, food_circle, defense_ring, energy_bar, status_text

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(frames), interval=100, blit=True)

    return fig, anim, frames


def save_comparison_gif():
    """Save GIF comparing different agents."""
    output_dir = Path(__file__).parent

    # Create side-by-side comparison for Strong tracking
    print("Creating agent comparison animation...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Defense Stack Comparison (Strong Tracking p=0.2)', fontsize=14, fontweight='bold')

    configs = [
        ('BASE', '#e74c3c', BaseAgent),
        ('RF+Hyst', '#3498db', RFHysteresisAgent),
        ('RF+TTC', '#2ecc71', RFTTCAgent),
    ]

    seed = 42
    p_chase = 0.2
    max_frames = 100

    config = BenchmarkConfig(
        p_chase=p_chase,
        max_steps=max_frames,
        food_energy=0.3,
        energy_decay=0.012
    )

    # Run all agents and store frames
    all_frames = []
    for name, color, agent_class in configs:
        env = BenchmarkEnv(config)
        agent = agent_class(config)
        obs = env.reset(seed)
        if hasattr(agent, 'reset'):
            agent.reset()

        frames = []
        done = False
        step = 0
        while not done and step < max_frames:
            action = agent.act(obs)
            frames.append({
                'agent_pos': obs['agent_pos'].copy(),
                'danger_pos': obs['danger_pos'].copy(),
                'food_pos': obs['food_pos'].copy(),
                'energy': obs['energy'],
                'in_defense': getattr(agent, 'in_defense', False),
                'step': step
            })
            obs, reward, done, info = env.step(action)
            step += 1

        all_frames.append((name, color, frames))

    # Setup axes
    plot_elements = []
    for idx, (ax, (name, color, frames)) in enumerate(zip(axes, all_frames)):
        ax.set_xlim(-0.5, config.grid_size - 0.5)
        ax.set_ylim(-0.5, config.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(name, fontsize=12, fontweight='bold', color=color)

        agent_c = plt.Circle((0, 0), 0.4, color=color, zorder=10)
        danger_c = plt.Circle((0, 0), 0.4, color='#e74c3c', zorder=5)
        food_c = plt.Circle((0, 0), 0.35, color='#f39c12', zorder=5)
        defense_r = plt.Circle((0, 0), 0.5, fill=False, color='blue', linewidth=3, zorder=9)

        ax.add_patch(agent_c)
        ax.add_patch(danger_c)
        ax.add_patch(food_c)
        ax.add_patch(defense_r)

        status = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_elements.append({
            'agent': agent_c, 'danger': danger_c, 'food': food_c,
            'defense': defense_r, 'status': status, 'frames': frames, 'color': color
        })

    def animate(i):
        artists = []
        for elem in plot_elements:
            frames = elem['frames']
            if i < len(frames):
                f = frames[i]
                elem['agent'].center = f['agent_pos']
                elem['danger'].center = f['danger_pos']
                elem['food'].center = f['food_pos']
                elem['defense'].center = f['agent_pos']
                elem['defense'].set_visible(f['in_defense'])

                mode = "DEF" if f['in_defense'] else "FOR"
                elem['status'].set_text(f"Step:{f['step']} E:{f['energy']:.0%}\n{mode}")

            artists.extend([elem['agent'], elem['danger'], elem['food'],
                          elem['defense'], elem['status']])
        return artists

    max_len = max(len(e['frames']) for e in plot_elements)
    anim = FuncAnimation(fig, animate, frames=max_len, interval=100, blit=True)

    # Save as GIF
    output_path = output_dir / 'agent_comparison.gif'
    print(f"Saving to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"Saved: {output_path}")

    plt.close()
    return output_path


def run_single_demo(agent_type='RF+TTC', p_chase=0.2):
    """Run and display single agent demo."""
    print(f"Running {agent_type} with p_chase={p_chase}...")
    fig, anim, frames = create_animation(agent_type, p_chase, max_frames=100)

    output_path = Path(__file__).parent / f'demo_{agent_type.replace("+", "_")}_{p_chase}.gif'
    print(f"Saving to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


if __name__ == "__main__":
    # Create comparison GIF
    gif_path = save_comparison_gif()
    print(f"\nAnimation saved: {gif_path}")
