#!/usr/bin/env python3
"""Plot loss vs iteration from RustScan training log"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RUN_DIR = Path(__file__).resolve().parent

# Parse log file
iterations = []
losses = []

with (RUN_DIR / 'log.md').open() as f:
    for line in f:
        match = re.search(r'Metal iter\s+(\d+)/30000.*loss\s+([\d.]+)', line)
        if match:
            iterations.append(int(match.group(1)))
            losses.append(float(match.group(2)))

print(f"Parsed {len(iterations)} data points from log")

# Create figure with larger size
fig, ax = plt.subplots(figsize=(14, 8))

# Plot raw loss with transparency
ax.plot(iterations, losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw Loss')

# Calculate and plot moving average (window=20)
window = 20
if len(losses) >= window:
    moving_avg = []
    moving_avg_x = []
    for i in range(window - 1, len(losses)):
        avg = sum(losses[i-window+1:i+1]) / window
        moving_avg.append(avg)
        moving_avg_x.append(iterations[i])
    ax.plot(moving_avg_x, moving_avg, 'r-', linewidth=2, label=f'Moving Average (n={window})')

# Add trend line (linear regression)
if len(iterations) > 10:
    z = np.polyfit(iterations, losses, 1)
    p = np.poly1d(z)
    ax.plot(iterations, p(iterations), 'g--', linewidth=2, label=f'Trend Line (slope={z[0]:.6f})')

# Labels and title
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('RustScan 3DGS Training: Loss vs Iteration', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add legend
ax.legend(loc='upper right', fontsize=10)

# Add annotation for final loss
ax.annotate(f'Final Loss: {losses[-1]:.6f}', 
            xy=(iterations[-1], losses[-1]), 
            xytext=(iterations[-1]-5000, losses[-1]+0.05),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'))

# Add annotation for min loss
min_idx = losses.index(min(losses))
ax.annotate(f'Min Loss: {min(losses):.6f}\n@ iter {iterations[min_idx]}', 
            xy=(iterations[min_idx], losses[min_idx]), 
            xytext=(iterations[min_idx]-5000, losses[min_idx]-0.08),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plot_path = RUN_DIR / 'loss_plot.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
