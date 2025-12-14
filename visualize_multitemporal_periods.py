"""
Visualization script for displaying multi-temporal model routes across all three periods
Shows morning, lunch, and evening periods side-by-side on spatial campus layout
"""

from shuttle_optimization import MultiTemporalShuttleOptimization, BusType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
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


def setup_network_multitemporal():
    """Setup and solve multi-temporal model"""
    sim = MultiTemporalShuttleOptimization(
        bus_types=BUS_TYPES,
        depot_cost_per_distance=DEPOT_COST_PER_DISTANCE,
        depot_fixed_cost=DEPOT_FIXED_COST,
        periods=['morning', 'lunch', 'evening']
    )

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
        office_to_lot_cost=10.0,
        use_distance_based_lot_costs=True
    )

    # Morning demand
    sim.set_demand("lot_A", "office_1", 50, "morning")
    sim.set_demand("lot_A", "office_2", 30, "morning")
    sim.set_demand("lot_B", "office_1", 40, "morning")
    sim.set_demand("lot_B", "office_2", 30, "morning")
    sim.set_demand("lot_C", "office_1", 35, "morning")
    sim.set_demand("lot_C", "office_2", 25, "morning")
    sim.set_demand("lot_D", "office_1", 25, "morning")
    sim.set_demand("lot_D", "office_2", 25, "morning")

    # Lunch demand
    sim.set_demand("office_1", "lot_A", 10, "lunch", commodity="office_1")
    sim.set_demand("office_1", "lot_B", 8, "lunch", commodity="office_1")
    sim.set_demand("office_2", "lot_C", 5, "lunch", commodity="office_2")
    sim.set_demand("office_2", "lot_D", 7, "lunch", commodity="office_2")

    # Evening demand
    sim.set_demand("office_1", "lot_A", 40, "evening", commodity="office_1")
    sim.set_demand("office_1", "lot_B", 32, "evening", commodity="office_1")
    sim.set_demand("office_1", "lot_C", 35, "evening", commodity="office_1")
    sim.set_demand("office_1", "lot_D", 25, "evening", commodity="office_1")
    sim.set_demand("office_2", "lot_A", 30, "evening", commodity="office_2")
    sim.set_demand("office_2", "lot_B", 30, "evening", commodity="office_2")
    sim.set_demand("office_2", "lot_C", 20, "evening", commodity="office_2")
    sim.set_demand("office_2", "lot_D", 18, "evening", commodity="office_2")

    sim.solve(verbose=False)
    return sim


def get_routes_for_period(sim, period):
    """Extract route information for a specific period"""
    routes = []

    for (origin, dest, bus_type, p), count in sim.bus_assignments.items():
        if p == period and count > 0:
            # Sum passengers across all commodities for this route
            total_pax = 0
            for office, flows in sim.commodity_flow.get(period, {}).items():
                total_pax += flows.get((origin, dest), 0)

            routes.append({
                'origin': origin.replace('lot_', '').replace('office_', 'O'),
                'dest': dest.replace('lot_', '').replace('office_', 'O'),
                'buses': count,
                'passengers': total_pax,
                'bus_type': bus_type
            })

    return routes


def plot_campus_with_routes(ax, sim, routes, title, period):
    """Plot campus layout with routes as arcs for a specific period"""

    # Node positions
    positions = {}
    lot_nodes = []
    office_nodes = []

    # Extract positions from sim.nodes
    for node_name, node in sim.nodes.items():
        if node.node_type == 'parking_lot':
            positions[node_name.replace('lot_', '')] = node.position
            lot_nodes.append(node_name.replace('lot_', ''))
        elif node.node_type == 'office':
            positions['O' + node_name.replace('office_', '')] = node.position
            office_nodes.append('O' + node_name.replace('office_', ''))

    # Draw nodes
    for node, pos in positions.items():
        if node in lot_nodes:
            circle = Circle(pos, 0.25, color='#4A90E2', zorder=10, alpha=0.9, edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], node, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=11)
        else:  # office
            circle = Circle(pos, 0.25, color='#E15759', zorder=10, alpha=0.9, edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], node, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=11)

    # Draw routes as curved arcs
    max_passengers = max([r['passengers'] for r in routes]) if routes else 1

    for route in routes:
        origin = route['origin']
        dest = route['dest']

        if origin not in positions or dest not in positions:
            continue

        pos_origin = positions[origin]
        pos_dest = positions[dest]

        # Arc width based on passenger flow
        width = 1 + 8 * (route['passengers'] / max_passengers) if route['passengers'] > 0 else 1

        # Uniform color for all arcs
        color = '#4A90E2'  # Blue color
        alpha = 0.7

        # Create curved arrow
        arrow = FancyArrowPatch(
            pos_origin, pos_dest,
            arrowstyle='->', mutation_scale=20,
            linewidth=width, color=color, alpha=alpha,
            connectionstyle="arc3,rad=0.2", zorder=5
        )
        ax.add_patch(arrow)

        # Add label at midpoint (only if passengers > 0)
        if route['passengers'] > 0:
            mid_x = (pos_origin[0] + pos_dest[0]) / 2
            mid_y = (pos_origin[1] + pos_dest[1]) / 2

            # Offset label slightly
            offset_x = 0.15 * (pos_dest[1] - pos_origin[1]) / (((pos_dest[0] - pos_origin[0])**2 + (pos_dest[1] - pos_origin[1])**2)**0.5 + 0.001)
            offset_y = -0.15 * (pos_dest[0] - pos_origin[0]) / (((pos_dest[0] - pos_origin[0])**2 + (pos_dest[1] - pos_origin[1])**2)**0.5 + 0.001)

            label = f"{int(route['passengers'])}"
            ax.text(mid_x + offset_x, mid_y + offset_y, label,
                   fontsize=9, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5),
                   zorder=15)

    # Set limits and styling
    ax.set_xlim(0, 9)
    ax.set_ylim(1, 8.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#4A90E2', edgecolor='white', label='Parking Lots'),
        mpatches.Patch(facecolor='#E15759', edgecolor='white', label='Offices'),
        mpatches.Patch(facecolor='#4A90E2', alpha=0.7, label='Bus Routes (width = flow)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)


def create_visualizations():
    """Create route visualizations for all three time periods"""
    print("="*90)
    print("MULTI-TEMPORAL PERIODS VISUALIZATION")
    print("="*90)
    print()

    # Solve model
    print("Solving multi-temporal model...")
    sim = setup_network_multitemporal()
    stats = sim.get_summary_statistics()
    print(f"  Total Cost: ${stats['total_cost']:.2f}")
    print(f"  Total Fleet Size: {stats['total_fleet_size']}")
    print()

    # Extract routes for each period
    routes_morning = get_routes_for_period(sim, 'morning')
    routes_lunch = get_routes_for_period(sim, 'lunch')
    routes_evening = get_routes_for_period(sim, 'evening')

    # Count passengers per period
    pax_morning = sum(r['passengers'] for r in routes_morning)
    pax_lunch = sum(r['passengers'] for r in routes_lunch)
    pax_evening = sum(r['passengers'] for r in routes_evening)

    print(f"  Morning passengers: {pax_morning}")
    print(f"  Lunch passengers: {pax_lunch}")
    print(f"  Evening passengers: {pax_evening}")
    print()

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    plot_campus_with_routes(
        axes[0], sim, routes_morning,
        f"Morning Period\n{pax_morning} passengers, {stats['buses_by_type_period']['morning']['Long Bus'] + stats['buses_by_type_period']['morning']['Stub Bus']} buses",
        'morning'
    )

    plot_campus_with_routes(
        axes[1], sim, routes_lunch,
        f"Lunch Period\n{pax_lunch} passengers, {sum(stats['buses_by_type_period']['lunch'].values())} buses",
        'lunch'
    )

    plot_campus_with_routes(
        axes[2], sim, routes_evening,
        f"Evening Period\n{pax_evening} passengers, {stats['buses_by_type_period']['evening']['Long Bus'] + stats['buses_by_type_period']['evening']['Stub Bus']} buses",
        'evening'
    )

    plt.tight_layout()

    # Save figure
    os.makedirs('plots', exist_ok=True)
    output_path = 'plots/multitemporal_periods_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    print()

    print("="*90)
    print("VISUALIZATION COMPLETE")
    print("="*90)


if __name__ == "__main__":
    create_visualizations()
