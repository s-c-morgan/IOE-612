"""
Generate heatmaps for route costs and depot costs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define nodes
lots = ['Lot A', 'Lot B', 'Lot C', 'Lot D']
offices = ['Office 1', 'Office 2']
all_nodes = lots + offices

# Create route cost matrix
# Base costs: lot-to-office=10, lot-to-lot=5, office-to-office=5
route_costs = np.zeros((len(all_nodes), len(all_nodes)))

# Lot to Office costs (rows 0-3, cols 4-5)
for i in range(4):  # Lots
    for j in range(4, 6):  # Offices
        route_costs[i, j] = 10.0

# Office to Lot costs (rows 4-5, cols 0-3)
for i in range(4, 6):  # Offices
    for j in range(4):  # Lots
        route_costs[i, j] = 10.0

# Lot to Lot costs (rows 0-3, cols 0-3)
for i in range(4):
    for j in range(4):
        if i != j:
            route_costs[i, j] = 5.0

# Office to Office costs (rows 4-5, cols 4-5)
for i in range(4, 6):
    for j in range(4, 6):
        if i != j:
            route_costs[i, j] = 5.0

# Set diagonal to NaN (no self-loops)
np.fill_diagonal(route_costs, np.nan)

# Create depot cost matrix (only for lots and offices as starting points)
# Depot distances: A=5, B=3, C=7, D=4 miles
# Depot costs for buses: fixed_cost + distance * rate * cost_mult
# We'll show costs for a typical Long Bus (cost_mult = 4.0)

depot_distances = {
    'Lot A': 5.0,
    'Lot B': 3.0,
    'Lot C': 7.0,
    'Lot D': 4.0,
    'Office 1': 0.0,  # Assume offices have no inherent depot distance
    'Office 2': 0.0
}

fixed_cost = 50.0
cost_per_mile = 1.0

# Calculate depot costs for different bus types
bus_types = {
    'Bike': 0.6,
    'Stub Bus': 1.0,
    'Medium Bus': 2.5,
    'Long Bus': 4.0
}

depot_cost_matrix = np.zeros((len(all_nodes), len(bus_types)))

for i, node in enumerate(all_nodes):
    distance = depot_distances[node]
    for j, (bus_name, cost_mult) in enumerate(bus_types.items()):
        if bus_name == 'Bike':
            depot_cost_matrix[i, j] = 0.0  # Bikes exempt from depot costs
        else:
            depot_cost_matrix[i, j] = fixed_cost + distance * cost_per_mile * cost_mult

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Route Cost Matrix
mask1 = np.isnan(route_costs)
sns.heatmap(route_costs,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=all_nodes,
            yticklabels=all_nodes,
            cbar_kws={'label': 'Route Cost ($)'},
            ax=ax1,
            mask=mask1,
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=10)

ax1.set_title('Route Cost Matrix\n(Base costs before bus type multiplier)',
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Destination', fontsize=12, fontweight='bold')
ax1.set_ylabel('Origin', fontsize=12, fontweight='bold')

# Add text annotations for cost types
ax1.text(0.5, -0.15,
         'Lot→Office: $10 | Lot→Lot: $5 | Office→Office: $5 | Office→Lot: $10',
         transform=ax1.transAxes,
         ha='center',
         fontsize=10,
         style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 2: Depot Cost Matrix
sns.heatmap(depot_cost_matrix,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=list(bus_types.keys()),
            yticklabels=all_nodes,
            cbar_kws={'label': 'Depot Cost ($)'},
            ax=ax2,
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=100)

ax2.set_title('Depot Start Cost Matrix\n(Fixed $50 + distance × $1/mile × cost multiplier)',
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Bus Type', fontsize=12, fontweight='bold')
ax2.set_ylabel('Starting Location', fontsize=12, fontweight='bold')

# Add depot distances annotation
depot_text = 'Depot Distances: Lot A=5mi, Lot B=3mi, Lot C=7mi, Lot D=4mi\nBikes exempt from depot costs'
ax2.text(0.5, -0.15,
         depot_text,
         transform=ax2.transAxes,
         ha='center',
         fontsize=10,
         style='italic',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()

# Create plots directory if it doesn't exist
import os
os.makedirs('plots', exist_ok=True)

plt.savefig('plots/cost_matrices_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as 'plots/cost_matrices_heatmap.png'")

# Also create separate larger heatmaps for better visibility

# Route Cost Matrix (standalone)
fig2, ax = plt.subplots(figsize=(10, 8))
mask = np.isnan(route_costs)
sns.heatmap(route_costs,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=all_nodes,
            yticklabels=all_nodes,
            cbar_kws={'label': 'Route Cost ($)'},
            ax=ax,
            mask=mask,
            linewidths=1,
            linecolor='gray',
            vmin=0,
            vmax=10,
            square=True)

ax.set_title('Route Cost Matrix\n(Base costs before bus type multiplier)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Destination', fontsize=14, fontweight='bold')
ax.set_ylabel('Origin', fontsize=14, fontweight='bold')

# Add legend box
legend_text = ('Route Types:\n'
               '• Lot → Office: $10\n'
               '• Office → Lot: $10\n'
               '• Lot → Lot: $5\n'
               '• Office → Office: $5\n\n'
               'Actual route cost = base cost × bus cost multiplier')
ax.text(1.15, 0.5,
        legend_text,
        transform=ax.transAxes,
        va='center',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/route_cost_matrix.png', dpi=300, bbox_inches='tight')
print("Route cost matrix saved as 'plots/route_cost_matrix.png'")

# Depot Cost Matrix (standalone)
fig3, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(depot_cost_matrix,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=list(bus_types.keys()),
            yticklabels=all_nodes,
            cbar_kws={'label': 'Depot Start Cost ($)'},
            ax=ax,
            linewidths=1,
            linecolor='gray',
            vmin=0,
            vmax=100,
            square=False)

ax.set_title('Depot Start Cost Matrix by Location and Bus Type\n(Fixed $50 + distance × $1/mile × cost multiplier)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Bus Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Starting Location', fontsize=14, fontweight='bold')

# Add legend box
depot_legend = ('Formula: $50 + d × $1/mi × m\n\n'
                'Depot Distances:\n'
                '• Lot A: 5 miles\n'
                '• Lot B: 3 miles\n'
                '• Lot C: 7 miles\n'
                '• Lot D: 4 miles\n\n'
                'Cost Multipliers:\n'
                '• Bike: 0.6× (exempt)\n'
                '• Stub Bus: 1.0×\n'
                '• Medium Bus: 2.5×\n'
                '• Long Bus: 4.0×\n\n'
                'Note: Bikes do not incur\n'
                'depot costs (presumed\n'
                'already distributed)')
ax.text(1.15, 0.5,
        depot_legend,
        transform=ax.transAxes,
        va='center',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/depot_cost_matrix.png', dpi=300, bbox_inches='tight')
print("Depot cost matrix saved as 'plots/depot_cost_matrix.png'")

print("\nAll heatmaps generated successfully!")
print("Files created in plots/ directory:")
print("  1. cost_matrices_heatmap.png (combined view)")
print("  2. route_cost_matrix.png (detailed route costs)")
print("  3. depot_cost_matrix.png (detailed depot costs)")
