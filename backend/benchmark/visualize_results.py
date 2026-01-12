"""
Visualize MiniGrid Benchmark Results - Phase Diagram Pattern
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from benchmark run
regimes = ['Random\n(p=0.0)', 'Weak\n(p=0.1)', 'Medium\n(p=0.15)', 'Strong\n(p=0.2)']
p_chase = [0.0, 0.1, 0.15, 0.2]

# Starvation rates (death rates)
base_death = [82, 98, 96, 98]
rf_death = [64, 84, 84, 84]
ttc_death = [58, 84, 90, 80]

# Survival times
base_survival = [100.5, 72.2, 76.1, 64.7]
rf_survival = [141.6, 103.1, 88.7, 88.8]
ttc_survival = [144.8, 116.2, 96.8, 110.1]

# Danger hits
base_hits = [8.5, 8.2, 9.2, 9.2]
rf_hits = [4.3, 4.6, 5.4, 5.3]
ttc_hits = [3.6, 4.3, 5.1, 4.9]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('MiniGrid Benchmark: Defense Stack Validation\n(E8 Phase Diagram Pattern)', fontsize=14, fontweight='bold')

# Colors
colors = {'BASE': '#e74c3c', 'RF+Hyst': '#3498db', 'RF+TTC': '#2ecc71'}

# Plot 1: Death Rate (Starvation)
ax1 = axes[0]
x = np.arange(len(regimes))
width = 0.25

bars1 = ax1.bar(x - width, base_death, width, label='BASE', color=colors['BASE'], alpha=0.8)
bars2 = ax1.bar(x, rf_death, width, label='RF+Hyst', color=colors['RF+Hyst'], alpha=0.8)
bars3 = ax1.bar(x + width, ttc_death, width, label='RF+TTC', color=colors['RF+TTC'], alpha=0.8)

# Highlight the DIP zone
ax1.axvspan(1.5, 2.5, alpha=0.2, color='yellow', label='DIP Zone')
ax1.annotate('TTC\nOverreacts', xy=(2, 92), fontsize=9, ha='center', color='#d35400', fontweight='bold')

ax1.set_ylabel('Death Rate (%)', fontsize=11)
ax1.set_title('Starvation Rate by Regime', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(regimes)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# Add winner markers
winners = ['TTC', 'TTC', 'RF', 'TTC']
for i, winner in enumerate(winners):
    if winner == 'TTC':
        ax1.plot(i + width, ttc_death[i] - 5, 'v', color='gold', markersize=10)
    else:
        ax1.plot(i, rf_death[i] - 5, 'v', color='gold', markersize=10)

# Plot 2: Survival Time
ax2 = axes[1]
ax2.plot(p_chase, base_survival, 'o-', color=colors['BASE'], label='BASE', linewidth=2, markersize=8)
ax2.plot(p_chase, rf_survival, 's-', color=colors['RF+Hyst'], label='RF+Hyst', linewidth=2, markersize=8)
ax2.plot(p_chase, ttc_survival, '^-', color=colors['RF+TTC'], label='RF+TTC', linewidth=2, markersize=8)

ax2.axvspan(0.12, 0.18, alpha=0.2, color='yellow')
ax2.set_xlabel('Tracking Probability (p_chase)', fontsize=11)
ax2.set_ylabel('Survival Steps', fontsize=11)
ax2.set_title('Survival Time by Tracking Level', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Danger Hits (Safety)
ax3 = axes[2]
ax3.plot(p_chase, base_hits, 'o-', color=colors['BASE'], label='BASE', linewidth=2, markersize=8)
ax3.plot(p_chase, rf_hits, 's-', color=colors['RF+Hyst'], label='RF+Hyst', linewidth=2, markersize=8)
ax3.plot(p_chase, ttc_hits, '^-', color=colors['RF+TTC'], label='RF+TTC', linewidth=2, markersize=8)

ax3.axvspan(0.12, 0.18, alpha=0.2, color='yellow')
ax3.set_xlabel('Tracking Probability (p_chase)', fontsize=11)
ax3.set_ylabel('Danger Hits (lower = safer)', fontsize=11)
ax3.set_title('Safety: Danger Encounters', fontsize=12)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(alpha=0.3)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / 'benchmark_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

# Also save a summary comparison chart
fig2, ax = plt.subplots(figsize=(10, 6))

# Delta: RF - TTC (positive = RF better, negative = TTC better)
delta = [rf_death[i] - ttc_death[i] for i in range(len(regimes))]

colors_delta = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta]
bars = ax.bar(regimes, delta, color=colors_delta, edgecolor='black', linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvspan(1.5, 2.5, alpha=0.2, color='yellow')

# Annotations
for i, (d, regime) in enumerate(zip(delta, regimes)):
    label = 'TTC wins' if d > 0 else 'RF wins'
    y_offset = 2 if d > 0 else -4
    ax.annotate(f'{abs(d)}%p\n{label}', xy=(i, d + y_offset), ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Death Rate Difference (RF - TTC)', fontsize=12)
ax.set_title('TTC vs RF: Who Wins by Regime?\n(Positive = TTC better, Negative = RF better)', fontsize=13, fontweight='bold')
ax.set_ylim(-15, 15)
ax.grid(axis='y', alpha=0.3)

# Add legend annotation
ax.annotate('DIP ZONE\n(TTC overreacts)', xy=(2, -12), fontsize=10, ha='center',
            color='#d35400', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

output_path2 = Path(__file__).parent / 'ttc_vs_rf_comparison.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path2}")

plt.show()
