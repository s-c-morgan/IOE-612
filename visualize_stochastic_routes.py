"""
Detailed Route Assignment Visualizations for Stochastic Capacity Model
Shows actual bus assignments and routing decisions across varying miss probabilities
"""

from shuttle_optimization import StochasticShuttleOptimization, BusType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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


def solve_and_get_routes(p):
    """
    Solve stochastic capacity model and extract route details

    Args:
        p: Miss probability

    Returns:
        dict with solution details including routes
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

    # Extract route information
    routes = {}
    for (from_node, to_node, bus_type), count in sim.bus_assignments.items():
        if count > 0:
            route = (from_node, to_node)
            if route not in routes:
                routes[route] = {}
            routes[route][bus_type] = count

    # Get passenger flows
    passenger_flows = {}
    for (from_node, to_node), pax in sim.passenger_flow.items():
        if pax > 0:
            passenger_flows[(from_node, to_node)] = pax

    stats = sim.get_summary_statistics()

    return {
        'p': p,
        'stats': stats,
        'routes': routes,
        'passenger_flows': passenger_flows,
        'bus_assignments': sim.bus_assignments
    }


def create_network_diagram(ax, solution_data, title):
    """
    Create a network diagram showing route assignments

    Args:
        ax: Matplotlib axis
        solution_data: Solution data dict
        title: Plot title
    """
    # Node positions
    positions = {
        'lot_A': (0, 3),
        'lot_B': (0, 2),
        'lot_C': (0, 1),
        'lot_D': (0, 0),
        'office_1': (3, 2.5),
        'office_2': (3, 0.5),
    }

    # Draw nodes
    for node, (x, y) in positions.items():
        if 'lot' in node:
            color = '#6A994E'
            label = node.replace('lot_', 'Lot ')
        else:
            color = '#4A4E69'
            label = node.replace('office_', 'Office ')

        bbox = FancyBboxPatch((x-0.25, y-0.15), 0.5, 0.3,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    # Draw routes with bus assignments
    routes = solution_data['routes']
    passenger_flows = solution_data['passenger_flows']

    # Color map for bus types
    bus_colors = {
        'Bike': '#F18F01',
        'Stub Bus': '#C73E1D',
        'Medium Bus': '#6A994E',
        'Long Bus': '#4A4E69'
    }

    for (from_node, to_node), bus_types_dict in routes.items():
        if from_node in positions and to_node in positions:
            x1, y1 = positions[from_node]
            x2, y2 = positions[to_node]

            # Calculate arrow offset for multiple routes between same nodes
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)

            # Perpendicular offset for curved arrows
            offset = 0.05
            mid_x = (x1 + x2) / 2 + offset * (-dy / length)
            mid_y = (y1 + y2) / 2 + offset * (dx / length)

            # Draw arrow
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->,head_width=0.3,head_length=0.2',
                connectionstyle=f"arc3,rad=.2",
                color='#2E86AB', alpha=0.6, linewidth=2.5
            )
            ax.add_patch(arrow)

            # Annotate with bus types and passengers
            bus_labels = []
            for bus_type, count in bus_types_dict.items():
                bus_abbrev = {'Bike': 'B', 'Stub Bus': 'S',
                             'Medium Bus': 'M', 'Long Bus': 'L'}[bus_type]
                bus_labels.append(f"{count}{bus_abbrev}")

            label = ", ".join(bus_labels)
            pax = passenger_flows.get((from_node, to_node), 0)

            ax.text(mid_x, mid_y, f"{label}\n{int(pax)}pax",
                   ha='center', va='center', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='#2E86AB', alpha=0.9))

    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.4, 3.4)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)


def create_route_assignment_table(ax, solution_data):
    """
    Create a detailed table of route assignments

    Args:
        ax: Matplotlib axis
        solution_data: Solution data dict
    """
    ax.axis('tight')
    ax.axis('off')

    routes = solution_data['routes']
    passenger_flows = solution_data['passenger_flows']

    # Prepare table data
    table_data = []
    table_data.append(['Route', 'Buses', 'Passengers', 'Capacity', 'Util%'])

    for (from_node, to_node), bus_types_dict in sorted(routes.items()):
        route_str = f"{from_node.replace('lot_', 'L').replace('office_', 'O')} → {to_node.replace('lot_', 'L').replace('office_', 'O')}"

        # Format bus types
        bus_strs = []
        total_capacity = 0
        for bus_type_name, count in bus_types_dict.items():
            bus_type = next(bt for bt in BUS_TYPES if bt.name == bus_type_name)
            capacity_per_bus = bus_type.capacity * (1 - solution_data['p'])
            total_capacity += capacity_per_bus * count

            abbrev = {'Bike': 'B', 'Stub Bus': 'S',
                     'Medium Bus': 'M', 'Long Bus': 'L'}[bus_type_name]
            bus_strs.append(f"{count}{abbrev}")

        buses_str = ", ".join(bus_strs)
        pax = int(passenger_flows.get((from_node, to_node), 0))
        util = (pax / total_capacity * 100) if total_capacity > 0 else 0

        table_data.append([route_str, buses_str, str(pax), f"{total_capacity:.1f}", f"{util:.0f}%"])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.25, 0.15, 0.15, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')


def create_fleet_allocation_bars(ax, solutions):
    """
    Create stacked bar chart showing fleet allocation across scenarios

    Args:
        ax: Matplotlib axis
        solutions: List of solution data dicts
    """
    p_values = [s['p'] for s in solutions]
    x = np.arange(len(p_values))

    # Extract bus counts by type
    bikes = []
    stubs = []
    mediums = []
    longs = []

    for s in solutions:
        bikes.append(s['stats']['buses_by_type'].get('Bike', 0))
        stubs.append(s['stats']['buses_by_type'].get('Stub Bus', 0))
        mediums.append(s['stats']['buses_by_type'].get('Medium Bus', 0))
        longs.append(s['stats']['buses_by_type'].get('Long Bus', 0))

    width = 0.6
    colors = ['#F18F01', '#C73E1D', '#6A994E', '#4A4E69']

    # Create stacked bars
    bottom = np.zeros(len(p_values))
    for data, color, label in zip([bikes, stubs, mediums, longs],
                                   colors,
                                   ['Bike', 'Stub Bus', 'Medium Bus', 'Long Bus']):
        bars = ax.bar(x, data, width, bottom=bottom, label=label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=2)

        # Add count labels
        for i, (val, bot) in enumerate(zip(data, bottom)):
            if val > 0:
                ax.text(i, bot + val/2, str(int(val)), ha='center', va='center',
                       fontweight='bold', fontsize=9, color='white')

        bottom += data

    ax.set_xlabel('Miss Probability (p)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Buses', fontsize=11, fontweight='bold')
    ax.set_title('Fleet Allocation by Miss Probability', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.0%}' for p in p_values])
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


def create_utilization_heatmap(ax, solutions):
    """
    Create heatmap showing route utilization across different p values

    Args:
        ax: Matplotlib axis
        solutions: List of solution data dicts
    """
    # Get all unique routes
    all_routes = set()
    for s in solutions:
        all_routes.update(s['routes'].keys())

    route_list = sorted(all_routes)
    p_values = [s['p'] for s in solutions]

    # Create utilization matrix
    util_matrix = np.zeros((len(route_list), len(p_values)))

    for j, solution in enumerate(solutions):
        routes = solution['routes']
        passenger_flows = solution['passenger_flows']

        for i, route in enumerate(route_list):
            if route in routes:
                # Calculate total capacity
                total_capacity = 0
                for bus_type_name, count in routes[route].items():
                    bus_type = next(bt for bt in BUS_TYPES if bt.name == bus_type_name)
                    capacity_per_bus = bus_type.capacity * (1 - solution['p'])
                    total_capacity += capacity_per_bus * count

                pax = passenger_flows.get(route, 0)
                util = (pax / total_capacity * 100) if total_capacity > 0 else 0
                util_matrix[i, j] = util

    # Create heatmap
    im = ax.imshow(util_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(p_values)))
    ax.set_yticks(np.arange(len(route_list)))

    ax.set_xticklabels([f'{p:.0%}' for p in p_values])
    route_labels = [f"{r[0].replace('lot_', 'L').replace('office_', 'O')} → {r[1].replace('lot_', 'L').replace('office_', 'O')}"
                    for r in route_list]
    ax.set_yticklabels(route_labels, fontsize=8)

    # Add text annotations
    for i in range(len(route_list)):
        for j in range(len(p_values)):
            if util_matrix[i, j] > 0:
                text = ax.text(j, i, f'{util_matrix[i, j]:.0f}%',
                             ha="center", va="center", color="black" if util_matrix[i, j] > 50 else "white",
                             fontsize=7, fontweight='bold')

    ax.set_title('Route Utilization (%) by Miss Probability', fontsize=12, fontweight='bold')
    ax.set_xlabel('Miss Probability (p)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Route', fontsize=11, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Utilization (%)', rotation=270, labelpad=15, fontweight='bold')


def main():
    """Main function to generate all route visualizations"""
    print("\n" + "="*90)
    print("STOCHASTIC CAPACITY MODEL - DETAILED ROUTE VISUALIZATIONS")
    print("="*90)
    print("\nGenerating route assignment visualizations...")
    print()

    # Solve for selected miss probabilities
    p_values = [0.0, 0.10, 0.20, 0.30]
    solutions = []

    for p in p_values:
        print(f"  Solving for p = {p:.0%}...")
        solution = solve_and_get_routes(p)
        solutions.append(solution)

    print("\nCreating visualizations...\n")

    # Create figure 1: Network diagrams for each p value
    fig1 = plt.figure(figsize=(16, 10))
    fig1.suptitle('Stochastic Capacity Model: Route Assignments by Miss Probability',
                  fontsize=14, fontweight='bold', y=0.98)

    for idx, solution in enumerate(solutions):
        ax = fig1.add_subplot(2, 2, idx + 1)
        title = f"p = {solution['p']:.0%} | Cost: ${solution['stats']['total_cost']:.0f} | Fleet: {solution['stats']['total_buses']} buses"
        create_network_diagram(ax, solution, title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path1 = os.path.join(output_dir, 'stochastic_route_networks.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[1/3] Network diagrams saved to: {output_path1}")
    plt.close()

    # Create figure 2: Route assignment tables
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('Stochastic Capacity Model: Detailed Route Assignments',
                  fontsize=14, fontweight='bold', y=0.98)

    for idx, solution in enumerate(solutions):
        ax = fig2.add_subplot(2, 2, idx + 1)
        ax.text(0.5, 0.95, f"p = {solution['p']:.0%} | Total Cost: ${solution['stats']['total_cost']:.2f}",
               ha='center', va='top', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.5, 0.08, f"Total Fleet: {solution['stats']['total_buses']} buses "
               f"({solution['stats']['buses_by_type']})",
               ha='center', va='bottom', fontsize=9,
               transform=ax.transAxes)
        create_route_assignment_table(ax, solution)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path2 = os.path.join(output_dir, 'stochastic_route_tables.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[2/3] Route tables saved to: {output_path2}")
    plt.close()

    # Create figure 3: Fleet allocation and utilization
    fig3 = plt.figure(figsize=(16, 6))
    fig3.suptitle('Stochastic Capacity Model: Fleet Allocation and Route Utilization',
                  fontsize=14, fontweight='bold')

    ax1 = fig3.add_subplot(1, 2, 1)
    create_fleet_allocation_bars(ax1, solutions)

    ax2 = fig3.add_subplot(1, 2, 2)
    create_utilization_heatmap(ax2, solutions)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path3 = os.path.join(output_dir, 'stochastic_fleet_utilization.png')
    plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[3/3] Fleet utilization saved to: {output_path3}")
    plt.close()

    print("\n" + "="*90)
    print("ROUTE VISUALIZATIONS COMPLETE")
    print("="*90)
    print("\nGenerated files:")
    print(f"  1. {output_path1}")
    print(f"  2. {output_path2}")
    print(f"  3. {output_path3}")
    print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    main()
