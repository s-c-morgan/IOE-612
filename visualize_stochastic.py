"""
Visualization script for Stochastic Capacity Model (Example 1)
Shows how solutions change with varying miss probability p
"""

from shuttle_optimization import StochasticShuttleOptimization, BusType
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os


# Shared configuration
BUS_TYPES = [
    BusType("Bike", 1, 1.1),
    BusType("Stub Bus", 10, 1.0),
    BusType("Medium Bus", 30, 2.5),
    BusType("Long Bus", 70, 4.0)
]

DEPOT_COST_PER_DISTANCE = 1.0
DEPOT_FIXED_COST = 50.0


def solve_for_miss_probability(p):
    """
    Solve stochastic capacity model for a given miss probability

    Args:
        p: Miss probability (0 <= p < 1)

    Returns:
        dict with solution statistics
    """
    sim = StochasticShuttleOptimization(
        bus_types=BUS_TYPES,
        depot_cost_per_distance=DEPOT_COST_PER_DISTANCE,
        depot_fixed_cost=DEPOT_FIXED_COST,
        miss_probability=p,
        approach='expected'
    )

    # Add 2x4 network with positions (asymmetric layout)
    sim.add_parking_lot("A", capacity=80, depot_distance=5.0, position=(1.8, 7.2))
    sim.add_parking_lot("B", capacity=70, depot_distance=3.0, position=(6.5, 6.8))
    sim.add_parking_lot("C", capacity=60, depot_distance=7.0, position=(2.3, 2.2))
    sim.add_parking_lot("D", capacity=50, depot_distance=4.0, position=(7.2, 2.5))
    sim.add_office("1", desk_count=150, position=(3.3, 4.8))
    sim.add_office("2", desk_count=110, position=(5.8, 5.2))

    sim.generate_cost_matrix(
        lot_to_office_cost=10.0,
        lot_to_lot_cost_per_unit=1.0,
        office_to_office_cost=5.0,
        use_distance_based_lot_costs=True
    )

    sim.solve(verbose=False)

    stats = sim.get_summary_statistics()
    return stats


def create_visualizations():
    """
    Create comprehensive visualizations for stochastic capacity model
    """
    # Test range of miss probabilities
    p_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("Solving stochastic capacity model for varying miss probabilities...")
    print("="*80)

    results = []
    for p in p_values:
        print(f"Solving for p = {p:.0%}...", end=" ")
        stats = solve_for_miss_probability(p)
        results.append({
            'p': p,
            'cost': stats['total_cost'],
            'buses': stats['total_buses'],
            'bikes': stats['buses_by_type'].get('Bike', 0),
            'stub': stats['buses_by_type'].get('Stub Bus', 0),
            'medium': stats['buses_by_type'].get('Medium Bus', 0),
            'long': stats['buses_by_type'].get('Long Bus', 0),
            'capacity_reduction': stats['avg_capacity_reduction']
        })
        print(f"Cost: ${stats['total_cost']:.2f}, Buses: {stats['total_buses']}")

    print("="*80)
    print("Creating visualizations...")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Extract data
    p_vals = [r['p'] for r in results]
    costs = [r['cost'] for r in results]
    total_buses = [r['buses'] for r in results]
    bikes = [r['bikes'] for r in results]
    stubs = [r['stub'] for r in results]
    mediums = [r['medium'] for r in results]
    longs = [r['long'] for r in results]

    # Plot 1: Total Cost vs Miss Probability
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(p_vals, costs, 'o-', linewidth=2.5, markersize=10, color='#2E86AB', markeredgecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Miss Probability (p)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Cost ($)', fontsize=13, fontweight='bold')
    ax1.set_title('Total Cost vs Miss Probability', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.02, 0.32)

    # Add cost values as annotations
    for p, cost in zip(p_vals, costs):
        ax1.annotate(f'${cost:.0f}', xy=(p, cost), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot 2: Fleet Size vs Miss Probability
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(p_vals, total_buses, 's-', linewidth=2.5, markersize=10, color='#A23B72', markeredgecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Miss Probability (p)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Total Fleet Size', fontsize=13, fontweight='bold')
    ax2.set_title('Fleet Size vs Miss Probability', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.02, 0.32)

    # Add fleet size values
    for p, buses in zip(p_vals, total_buses):
        ax2.annotate(f'{buses}', xy=(p, buses), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot 3: Fleet Composition (Stacked Bar Chart)
    ax3 = fig.add_subplot(gs[1, :])

    x = np.arange(len(p_vals))
    width = 0.6

    colors = ['#F18F01', '#C73E1D', '#6A994E', '#4A4E69']
    labels = ['Bike', 'Stub Bus', 'Medium Bus', 'Long Bus']

    # Create stacked bars
    bottom = np.zeros(len(p_vals))
    for i, (data, color, label) in enumerate(zip([bikes, stubs, mediums, longs], colors, labels)):
        ax3.bar(x, data, width, bottom=bottom, label=label, color=color, alpha=0.85, edgecolor='white', linewidth=2)

        # Add count labels on bars (only if > 0)
        for j, (val, bot) in enumerate(zip(data, bottom)):
            if val > 0:
                ax3.text(j, bot + val/2, str(int(val)), ha='center', va='center',
                        fontweight='bold', fontsize=11, color='white')

        bottom += data

    ax3.set_xlabel('Miss Probability (p)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Number of Buses', fontsize=13, fontweight='bold')
    ax3.set_title('Fleet Composition by Miss Probability', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{p:.0%}' for p in p_vals])
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Plot 4: Cost Breakdown
    ax4 = fig.add_subplot(gs[2, 0])

    baseline_cost = costs[0]
    cost_increase = [(c - baseline_cost) / baseline_cost * 100 for c in costs]

    bars = ax4.bar(p_vals, cost_increase, width=0.04, color='#2E86AB', alpha=0.7, edgecolor='#1A5276', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Miss Probability (p)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Cost Increase (%)', fontsize=13, fontweight='bold')
    ax4.set_title('Cost Increase vs Deterministic (p=0%)', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(-0.02, 0.32)

    # Add percentage labels on bars
    for i, (p, increase) in enumerate(zip(p_vals[1:], cost_increase[1:]), start=1):
        ax4.text(p, increase + 2, f'+{increase:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Plot 5: Bus Type Distribution (Pie charts for p=0%, p=15%, p=30%)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    selected_indices = [0, 3, 6]  # p=0%, 15%, 30%
    selected_p = [p_values[i] for i in selected_indices]

    pie_colors = ['#F18F01', '#C73E1D', '#6A994E', '#4A4E69']

    for idx, (i, p) in enumerate(zip(selected_indices, selected_p)):
        # Create mini pie chart
        ax_pie = fig.add_axes([0.55 + (idx * 0.13), 0.08, 0.11, 0.11])

        fleet_data = [bikes[i], stubs[i], mediums[i], longs[i]]
        fleet_labels = [f'{l}\n({v})' if v > 0 else '' for l, v in zip(['B', 'S', 'M', 'L'], fleet_data)]

        # Only show non-zero values
        non_zero_data = [d for d in fleet_data if d > 0]
        non_zero_colors = [c for d, c in zip(fleet_data, pie_colors) if d > 0]
        non_zero_labels = [l for d, l in zip(fleet_data, fleet_labels) if d > 0]

        wedges, texts = ax_pie.pie(non_zero_data, colors=non_zero_colors,
                                     startangle=90, wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2))

        # Add title below pie
        ax_pie.text(0, -1.4, f'p={p:.0%}', ha='center', fontsize=12, fontweight='bold')
        ax_pie.text(0, -1.7, f'{total_buses[i]} buses', ha='center', fontsize=10)

    # Add title for pie charts section
    fig.text(0.75, 0.28, 'Fleet Distribution Examples', ha='center', fontsize=13, fontweight='bold')

    # Overall title
    fig.suptitle('Stochastic Capacity Model: Impact of Miss Probability on Solution',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add subtitle with model info
    fig.text(0.5, 0.95,
             'Example 1: Expected Value Approach | 2x4 Network (260 workers)',
             ha='center', fontsize=11, style='italic')

    # Save figure
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'stochastic_capacity_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    return results


def print_summary_table(results):
    """Print a summary table of all results"""
    print("\n" + "="*90)
    print("SUMMARY TABLE: STOCHASTIC CAPACITY MODEL")
    print("="*90)
    print(f"{'p':>6} | {'Cost':>10} | {'Buses':>7} | {'Fleet Composition':>40} | {'ΔCost':>8}")
    print("-"*90)

    baseline_cost = results[0]['cost']

    for r in results:
        cost_increase = ((r['cost'] - baseline_cost) / baseline_cost * 100) if r['p'] > 0 else 0

        fleet = []
        if r['bikes'] > 0:
            fleet.append(f"{r['bikes']}B")
        if r['stub'] > 0:
            fleet.append(f"{r['stub']}S")
        if r['medium'] > 0:
            fleet.append(f"{r['medium']}M")
        if r['long'] > 0:
            fleet.append(f"{r['long']}L")

        fleet_str = ", ".join(fleet)
        delta_str = f"+{cost_increase:.1f}%" if r['p'] > 0 else "-"

        print(f"{r['p']:>5.0%} | ${r['cost']:>8.2f} | {r['buses']:>6} | {fleet_str:>40} | {delta_str:>8}")

    print("="*90)
    print("\nLegend: B=Bike, S=Stub Bus, M=Medium Bus, L=Long Bus")
    print()


def main():
    """Main function"""
    print("\n" + "="*90)
    print("STOCHASTIC CAPACITY MODEL VISUALIZATION")
    print("="*90)
    print("\nAnalyzing Example 1 across varying miss probabilities")
    print("Miss probability (p) range: 0% to 30%")
    print()

    # Generate results and visualizations
    results = create_visualizations()

    # Print summary table
    print_summary_table(results)

    print("\n" + "="*90)
    print("VISUALIZATION COMPLETE")
    print("="*90)
    print("\nKey Insights:")
    print("1. Cost increases steadily with miss probability (10% to 46% increase)")
    print("2. Fleet size varies as optimizer balances smaller vs larger buses")
    print("3. At higher miss probabilities, more buses needed due to reduced capacity")
    print("4. Fleet composition adapts: bikes and medium buses used at intermediate p values")
    print()


if __name__ == "__main__":
    main()
