"""
Unified script to run all shuttle optimization examples
Includes: Stochastic Capacity, Multicommodity, and Multi-Temporal models
"""

from shuttle_optimization import (
    MulticommodityShuttleOptimization,
    MultiTemporalShuttleOptimization,
    StochasticShuttleOptimization,
    BusType
)
import sys


# Shared configuration for all models
BUS_TYPES = [
    BusType("Bike", 1, 1.1),           # Bikes: 1 person, 1.1x cost
    BusType("Stub Bus", 10, 1.0),
    BusType("Medium Bus", 30, 2.5),
    BusType("Long Bus", 70, 4.0)
]

DEPOT_COST_PER_DISTANCE = 1.0  # $1 per mile
DEPOT_FIXED_COST = 50.0         # $50 fixed cost per bus (bikes exempt)


def setup_network_deterministic(sim, multicommodity=False):
    """
    Common network setup for deterministic models

    Args:
        sim: Optimization model instance
        multicommodity: If True, sets demand matrix for multicommodity models
    """
    # Add 4 parking lots with depot distances and positions (asymmetric layout)
    sim.add_parking_lot("A", capacity=80, depot_distance=5.0, position=(1.8, 7.2))
    sim.add_parking_lot("B", capacity=70, depot_distance=3.0, position=(6.5, 6.8))
    sim.add_parking_lot("C", capacity=60, depot_distance=7.0, position=(2.3, 2.2))
    sim.add_parking_lot("D", capacity=50, depot_distance=4.0, position=(7.2, 2.5))

    # Add 2 offices (total: 260 workers) with positions
    sim.add_office("1", desk_count=150, position=(3.3, 4.8))
    sim.add_office("2", desk_count=110, position=(5.8, 5.2))

    # Set demand matrix for multicommodity models
    if multicommodity:
        # Lot A (80 workers)
        sim.set_demand("A", "1", 50)
        sim.set_demand("A", "2", 30)

        # Lot B (70 workers)
        sim.set_demand("B", "1", 40)
        sim.set_demand("B", "2", 30)

        # Lot C (60 workers)
        sim.set_demand("C", "1", 35)
        sim.set_demand("C", "2", 25)

        # Lot D (50 workers)
        sim.set_demand("D", "1", 25)
        sim.set_demand("D", "2", 25)

    # Generate cost matrix with distance-based lot-to-lot costs
    sim.generate_cost_matrix(
        lot_to_office_cost=10.0,
        lot_to_lot_cost_per_unit=1.0,  # $1 per unit distance
        office_to_office_cost=5.0,
        use_distance_based_lot_costs=True
    )


def setup_network_multitemporal(sim):
    """
    Network setup for multi-temporal model with demand across periods
    """
    # Add 4 parking lots with depot distances and positions (asymmetric layout)
    sim.add_parking_lot("A", capacity=80, depot_distance=5.0, position=(1.8, 7.2))
    sim.add_parking_lot("B", capacity=70, depot_distance=3.0, position=(6.5, 6.8))
    sim.add_parking_lot("C", capacity=60, depot_distance=7.0, position=(2.3, 2.2))
    sim.add_parking_lot("D", capacity=50, depot_distance=4.0, position=(7.2, 2.5))

    # Add 2 offices (total: 260 workers) with positions
    sim.add_office("1", desk_count=150, position=(3.3, 4.8))
    sim.add_office("2", desk_count=110, position=(5.8, 5.2))

    # Generate cost matrix with distance-based lot-to-lot costs
    sim.generate_cost_matrix(
        lot_to_office_cost=10.0,
        lot_to_lot_cost_per_unit=1.0,  # $1 per unit distance
        office_to_office_cost=5.0,
        office_to_lot_cost=10.0,
        use_distance_based_lot_costs=True
    )

    # Morning: Workers commute from lots to offices
    sim.set_demand("lot_A", "office_1", 50, "morning")
    sim.set_demand("lot_A", "office_2", 30, "morning")
    sim.set_demand("lot_B", "office_1", 40, "morning")
    sim.set_demand("lot_B", "office_2", 30, "morning")
    sim.set_demand("lot_C", "office_1", 35, "morning")
    sim.set_demand("lot_C", "office_2", 25, "morning")
    sim.set_demand("lot_D", "office_1", 25, "morning")
    sim.set_demand("lot_D", "office_2", 25, "morning")

    # Lunch: Some workers leave campus
    sim.set_demand("office_1", "lot_A", 10, "lunch", commodity="office_1")
    sim.set_demand("office_1", "lot_B", 8, "lunch", commodity="office_1")
    sim.set_demand("office_2", "lot_C", 5, "lunch", commodity="office_2")
    sim.set_demand("office_2", "lot_D", 7, "lunch", commodity="office_2")

    # Evening: Remaining workers return home
    sim.set_demand("office_1", "lot_A", 40, "evening", commodity="office_1")
    sim.set_demand("office_1", "lot_B", 32, "evening", commodity="office_1")
    sim.set_demand("office_1", "lot_C", 35, "evening", commodity="office_1")
    sim.set_demand("office_1", "lot_D", 25, "evening", commodity="office_1")
    sim.set_demand("office_2", "lot_A", 30, "evening", commodity="office_2")
    sim.set_demand("office_2", "lot_B", 30, "evening", commodity="office_2")
    sim.set_demand("office_2", "lot_C", 20, "evening", commodity="office_2")
    sim.set_demand("office_2", "lot_D", 18, "evening", commodity="office_2")


def run_example_1_stochastic_capacity():
    """
    Example 1: Stochastic Capacity Model
    Incorporates uncertainty in bus capacity due to passengers missing buses
    """
    print("\n" + "="*100)
    print("EXAMPLE 1: STOCHASTIC CAPACITY MODEL")
    print("="*100)
    print("\nExpected value approach with miss probability p=20%")
    print()

    sim = StochasticShuttleOptimization(
        bus_types=BUS_TYPES,
        depot_cost_per_distance=DEPOT_COST_PER_DISTANCE,
        depot_fixed_cost=DEPOT_FIXED_COST,
        miss_probability=0.20,    # 20% miss rate
        approach='expected'
    )

    # Add network with positions (asymmetric layout)
    sim.add_parking_lot("A", capacity=80, depot_distance=5.0, position=(1.8, 7.2))
    sim.add_parking_lot("B", capacity=70, depot_distance=3.0, position=(6.5, 6.8))
    sim.add_parking_lot("C", capacity=60, depot_distance=7.0, position=(2.3, 2.2))
    sim.add_parking_lot("D", capacity=50, depot_distance=4.0, position=(7.2, 2.5))
    sim.add_office("1", desk_count=150, position=(3.3, 4.8))
    sim.add_office("2", desk_count=110, position=(5.8, 5.2))

    # Generate cost matrix with distance-based lot-to-lot costs
    sim.generate_cost_matrix(
        lot_to_office_cost=10.0,
        lot_to_lot_cost_per_unit=1.0,  # $1 per unit distance
        office_to_office_cost=5.0,
        use_distance_based_lot_costs=True
    )

    sim.solve(verbose=True)

    stats = sim.get_summary_statistics()
    print(f"\n{'='*100}")
    print("EXAMPLE 1 SUMMARY:")
    print(f"{'='*100}")
    print(f"Total Cost: ${stats['total_cost']:.2f}")
    print(f"Total Buses: {stats['total_buses']}")
    print(f"Fleet Composition: {stats['buses_by_type']}")
    print(f"Miss Probability: {stats['miss_probability']:.1%}")
    print(f"Average Capacity Reduction: {stats['avg_capacity_reduction']:.1%}")
    print(f"{'='*100}\n")

    return stats


def run_example_2_multicommodity():
    """
    Example 2: Multicommodity Model
    Tracks individual worker destinations, enforces origin-destination requirements
    """
    print("\n" + "="*100)
    print("EXAMPLE 2: MULTICOMMODITY MODEL")
    print("="*100)
    print("\nTracks individual worker flows to specific office destinations")
    print()

    sim = MulticommodityShuttleOptimization(
        bus_types=BUS_TYPES,
        depot_cost_per_distance=DEPOT_COST_PER_DISTANCE,
        depot_fixed_cost=DEPOT_FIXED_COST
    )

    setup_network_deterministic(sim, multicommodity=True)
    sim.solve(verbose=True)

    stats = sim.get_summary_statistics()
    print(f"\n{'='*100}")
    print("EXAMPLE 2 SUMMARY:")
    print(f"{'='*100}")
    print(f"Total Cost: ${stats['total_cost']:.2f}")
    print(f"Total Buses: {stats['total_buses']}")
    print(f"Fleet Composition: {stats['buses_by_type']}")
    print(f"Number of Commodities: {stats['num_commodities']}")
    print(f"Direct Routes: {stats['direct_routes']}")
    print(f"Transfer Routes: {stats['transfer_routes']}")
    print(f"{'='*100}\n")

    return stats


def run_example_3_multitemporal():
    """
    Example 3: Multi-Temporal Model
    Handles multiple time periods (morning/lunch/evening) with bus inventory tracking
    """
    print("\n" + "="*100)
    print("EXAMPLE 3: MULTI-TEMPORAL MODEL")
    print("="*100)
    print("\nOptimizes bus operations across morning/lunch/evening periods")
    print()

    sim = MultiTemporalShuttleOptimization(
        bus_types=BUS_TYPES,
        depot_cost_per_distance=DEPOT_COST_PER_DISTANCE,
        depot_fixed_cost=DEPOT_FIXED_COST,
        periods=['morning', 'lunch', 'evening']
    )

    setup_network_multitemporal(sim)
    sim.solve(verbose=True)

    stats = sim.get_summary_statistics()
    print(f"\n{'='*100}")
    print("EXAMPLE 3 SUMMARY:")
    print(f"{'='*100}")
    print(f"Total Cost: ${stats['total_cost']:.2f}")
    print(f"Total Fleet Size: {stats['total_fleet_size']}")
    print(f"Fleet Composition: {stats['fleet_by_type']}")
    print(f"Buses by Period: {stats['buses_by_type_period']}")
    print(f"{'='*100}\n")

    return stats


def print_comparison_table(results):
    """
    Print a comparison table of all models
    """
    print("\n" + "="*100)
    print("COMPARISON OF ALL MODELS")
    print("="*100)
    print(f"{'Model':<30} | {'Cost':>12} | {'Buses':>8} | {'Fleet Composition'}")
    print("-"*100)

    for model_name, stats in results.items():
        cost = stats['total_cost']

        if 'total_fleet_size' in stats:
            buses = stats['total_fleet_size']
        else:
            buses = stats['total_buses']

        # Format fleet composition
        if 'fleet_by_type' in stats:
            fleet = stats['fleet_by_type']
        else:
            fleet = stats['buses_by_type']

        fleet_str = ", ".join([f"{count}{name[:1]}" for name, count in fleet.items() if count > 0])

        print(f"{model_name:<30} | ${cost:>10.2f} | {buses:>7} | {fleet_str}")

    print("="*100)
    print("\nLegend: B=Bike, S=Stub Bus, M=Medium Bus, L=Long Bus")
    print()


def main():
    """
    Main function to run all examples or specific ones
    """
    print("\n" + "="*100)
    print("SHUTTLE OPTIMIZATION - ALL EXAMPLES")
    print("="*100)
    print("\nRunning all three optimization models on 2x4 network (260 workers)")
    print("Network: 2 offices (150, 110 desks) × 4 parking lots (80, 70, 60, 50 capacity)")
    print()

    # Check command line arguments
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()

        if choice == '1' or choice == 'stochastic':
            run_example_1_stochastic_capacity()
            return
        elif choice == '2' or choice == 'multi':
            run_example_2_multicommodity()
            return
        elif choice == '3' or choice == 'temporal':
            run_example_3_multitemporal()
            return
        else:
            print(f"Unknown option: {choice}")
            print("Usage: python run_all_examples.py [1|2|3|stochastic|multi|temporal]")
            return

    # Run all examples
    results = {}

    results['Stochastic Capacity (p=20%)'] = run_example_1_stochastic_capacity()
    results['Multicommodity'] = run_example_2_multicommodity()
    results['Multi-Temporal'] = run_example_3_multitemporal()

    # Print comparison
    print_comparison_table(results)

    print("\n" + "="*100)
    print("ALL EXAMPLES COMPLETED")
    print("="*100)
    print("\nTo run individual examples:")
    print("  python run_all_examples.py 1        # Stochastic Capacity")
    print("  python run_all_examples.py 2        # Multicommodity")
    print("  python run_all_examples.py 3        # Multi-Temporal")
    print()


if __name__ == "__main__":
    main()
