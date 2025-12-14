"""
Visualize the campus layout for 2 offices, 4 parking lots scenario
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
import numpy as np

# Set up figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define campus layout (x, y coordinates)
# Parking lots with depot distances (asymmetric layout)
parking_lots = {
    'Lot A': {
        'pos': (1.8, 7.2),
        'capacity': 80,
        'depot_dist': 5.0,
        'color': '#90EE90'
    },
    'Lot B': {
        'pos': (6.5, 6.8),
        'capacity': 70,
        'depot_dist': 3.0,
        'color': '#98FB98'
    },
    'Lot C': {
        'pos': (2.3, 2.2),
        'capacity': 60,
        'depot_dist': 7.0,
        'color': '#8FBC8F'
    },
    'Lot D': {
        'pos': (7.2, 2.5),
        'capacity': 50,
        'depot_dist': 4.0,
        'color': '#3CB371'
    },
}

# Office buildings (asymmetric layout)
offices = {
    'Office 1': {
        'pos': (3.3, 4.8),
        'desks': 150,
        'color': '#4169E1',
        'height': 1.2,
        'width': 1.4
    },
    'Office 2': {
        'pos': (5.8, 5.2),
        'desks': 110,
        'color': '#1E90FF',
        'height': 1.0,
        'width': 1.2
    },
}

# Depot location
depot_pos = (9.5, 5)

# Draw depot (star shape)
depot_size = 0.5
depot = plt.Polygon([
    (depot_pos[0], depot_pos[1] + depot_size),
    (depot_pos[0] + 0.15, depot_pos[1] + 0.15),
    (depot_pos[0] + depot_size, depot_pos[1] + 0.15),
    (depot_pos[0] + 0.2, depot_pos[1] - 0.1),
    (depot_pos[0] + 0.3, depot_pos[1] - depot_size),
    (depot_pos[0], depot_pos[1] - 0.2),
    (depot_pos[0] - 0.3, depot_pos[1] - depot_size),
    (depot_pos[0] - 0.2, depot_pos[1] - 0.1),
    (depot_pos[0] - depot_size, depot_pos[1] + 0.15),
    (depot_pos[0] - 0.15, depot_pos[1] + 0.15),
], closed=True, facecolor='#FFD700', edgecolor='darkorange', linewidth=2.5, zorder=5)
ax.add_patch(depot)
ax.text(depot_pos[0], depot_pos[1] - 0.9, 'Depot', ha='center', va='top',
        fontweight='bold', fontsize=10, color='darkorange')

# Draw depot distance lines (dashed)
for name, info in parking_lots.items():
    x, y = info['pos']
    dist = info['depot_dist']
    ax.plot([x, depot_pos[0]], [y, depot_pos[1]], 'k--', alpha=0.25, linewidth=1, zorder=1)
    # Label distance at midpoint
    mid_x, mid_y = (x + depot_pos[0]) / 2, (y + depot_pos[1]) / 2
    ax.text(mid_x, mid_y, f'{dist}mi', fontsize=7, ha='center', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, linewidth=0))

# Draw parking lots (circles)
for name, info in parking_lots.items():
    x, y = info['pos']
    capacity = info['capacity']

    # Size based on capacity
    radius = 0.35 + (capacity / 200) * 0.3

    circle = Circle((x, y), radius, color=info['color'], alpha=0.7,
                   edgecolor='darkgreen', linewidth=3, zorder=2)
    ax.add_patch(circle)

    # Label
    ax.text(x, y, f"{name}\n{capacity}", ha='center', va='center',
            fontweight='bold', fontsize=10, zorder=3)

# Draw office buildings (rectangles)
for name, info in offices.items():
    x, y = info['pos']
    desks = info['desks']
    height = info['height']
    width = info['width']

    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.08",
                          facecolor=info['color'], alpha=0.85,
                          edgecolor='darkblue', linewidth=3, zorder=2)
    ax.add_patch(rect)

    # Label
    ax.text(x, y, f"{name}\n{desks}", ha='center', va='center',
            fontweight='bold', fontsize=11, color='white', zorder=3)

# Add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#90EE90', edgecolor='darkgreen', label='Parking Lot'),
    Patch(facecolor='#4169E1', edgecolor='darkblue', label='Office'),
    Patch(facecolor='#FFD700', edgecolor='darkorange', label='Depot'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# Add title
ax.set_title('Campus Network Layout (2×4 Configuration)',
            fontsize=16, fontweight='bold', pad=15)

# Set limits and aspect
ax.set_xlim(-0.5, 11)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()

# Create plots directory if it doesn't exist
import os
os.makedirs('plots', exist_ok=True)

plt.savefig('plots/campus_layout_2x4.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Campus layout saved to: plots/campus_layout_2x4.png")

plt.show()
