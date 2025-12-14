"""
Visualize routing matrices: Stochastic Capacity, Multicommodity, Multi-Temporal Morning
Clean version with minimal text
Shows the 3 current model variants (Example 1, 2, 3)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define nodes
nodes = ['Lot A', 'Lot B', 'Lot C', 'Lot D', 'Office 1', 'Office 2']

# Create figure with 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(8, 15))

# ===========================
# STOCHASTIC CAPACITY MODEL (Example 1, p=20%)
# ===========================
routes_stochastic = np.zeros((6, 6))
routes_stochastic[0, 4] = 80  # Lot A → Office 1
routes_stochastic[1, 4] = 70  # Lot B → Office 1
routes_stochastic[2, 5] = 60  # Lot C → Office 2
routes_stochastic[3, 5] = 50  # Lot D → Office 2

mask1 = routes_stochastic == 0
sns.heatmap(routes_stochastic, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=False, yticklabels=False,
            cbar=False,
            ax=axes[0], mask=mask1, linewidths=1, linecolor='gray',
            vmin=0, vmax=80, square=True)
axes[0].set_title('Stochastic Capacity (p=20%)', fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('')
axes[0].set_ylabel('')

# ===========================
# MULTICOMMODITY MODEL
# ===========================
routes_multi = np.zeros((6, 6))
routes_multi[0, 2] = 10  # Lot A → Lot C
routes_multi[0, 4] = 70  # Lot A → Office 1
routes_multi[1, 5] = 70  # Lot B → Office 2
routes_multi[2, 4] = 70  # Lot C → Office 1
routes_multi[3, 5] = 50  # Lot D → Office 2
routes_multi[4, 5] = 55  # Office 1 → Office 2
routes_multi[5, 4] = 65  # Office 2 → Office 1

mask2 = routes_multi == 0
sns.heatmap(routes_multi, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=False, yticklabels=False,
            cbar=False,
            ax=axes[1], mask=mask2, linewidths=1, linecolor='gray',
            vmin=0, vmax=80, square=True)
axes[1].set_title('Multicommodity', fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('')
axes[1].set_ylabel('')

# ===========================
# MULTI-TEMPORAL MODEL: MORNING
# ===========================
routes_morning = np.zeros((6, 6))
routes_morning[0, 3] = 5   # Lot A → Lot D
routes_morning[0, 4] = 65  # Lot A → Office 1
routes_morning[0, 5] = 10  # Lot A → Office 2
routes_morning[1, 4] = 70  # Lot B → Office 1
routes_morning[2, 4] = 60  # Lot C → Office 1
routes_morning[3, 4] = 25  # Lot D → Office 1
routes_morning[3, 5] = 30  # Lot D → Office 2
routes_morning[4, 5] = 70  # Office 1 → Office 2

mask3 = routes_morning == 0
sns.heatmap(routes_morning, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=False, yticklabels=False,
            cbar=False,
            ax=axes[2], mask=mask3, linewidths=1, linecolor='gray',
            vmin=0, vmax=80, square=True)
axes[2].set_title('Multi-Temporal (Morning)', fontsize=14, fontweight='bold', pad=10)
axes[2].set_xlabel('')
axes[2].set_ylabel('')

plt.tight_layout()

# Create plots directory if it doesn't exist
import os
os.makedirs('plots', exist_ok=True)

plt.savefig('plots/routing_matrices_clean.png', dpi=300, bbox_inches='tight')
print("✓ Clean routing matrices visualization saved to: plots/routing_matrices_clean.png")

plt.show()
