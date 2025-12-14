"""
Shuttle Service Optimization Models
Unified module containing both single-commodity and multicommodity flow models
Supports multiple bus types with different capacities and costs
Includes stochastic extensions for capacity uncertainty
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pulp
from scipy import stats


@dataclass
class BusType:
    """Represents a type of bus with specific capacity and cost multiplier"""
    name: str
    capacity: int
    cost_multiplier: float  # Relative to base cost

    def __repr__(self):
        return f"{self.name} (cap={self.capacity}, cost_mult={self.cost_multiplier}x)"


@dataclass
class NetworkNode:
    """Represents a node in the shuttle network"""
    node_id: str
    node_type: str  # 'parking_lot' or 'office'
    capacity: int = 0  # Supply (parking) or demand (office)
    position: Tuple[float, float] = (0.0, 0.0)  # (x, y) coordinates for distance calculations

    def __repr__(self):
        return f"{self.node_type}_{self.node_id}"


class ShuttleOptimization:
    """
    Single-commodity (aggregate) flow model with multiple bus types.

    Decision variables:
    - X[a,b,bus_type] = number of buses of given type from node a to node b
    - P[a,b] = number of passengers from node a to node b (aggregate)

    Constraints:
    1. P[a,b] <= Σ_t (capacity_t * X[a,b,t])
    2. Office demand: inflow = desk_count
    3. Lot supply: outflow = capacity + inflow
    """

    def __init__(self, bus_types: Optional[List[BusType]] = None,
                 depot_cost_per_distance: float = 0.0,
                 depot_fixed_cost: float = 0.0):
        # Default: single bus type with capacity 30
        if bus_types is None:
            bus_types = [BusType("Standard", 30, 1.0)]

        self.bus_types = bus_types
        self.nodes: Dict[str, NetworkNode] = {}
        self.cost_matrix: Dict[Tuple[str, str], float] = {}

        # Depot parameters for bus starting costs (bikes exempt)
        self.depot_cost_per_distance = depot_cost_per_distance
        self.depot_fixed_cost = depot_fixed_cost
        self.lot_depot_distances: Dict[str, float] = {}
        self.office_depot_distances: Dict[str, float] = {}  # Distance from depot to each office

        # Performance caches
        self._lots_cache: Optional[List[str]] = None
        self._offices_cache: Optional[List[str]] = None
        self._depot_distances_cache: Optional[Dict[str, float]] = None
        self._bus_type_by_name: Dict[str, BusType] = {bt.name: bt for bt in self.bus_types}

        # X is now indexed by (from, to, bus_type)
        self.bus_assignments: Dict[Tuple[str, str, str], int] = {}
        self.passenger_flow: Dict[Tuple[str, str], int] = {}
        self.total_cost: float = 0.0
        self.depot_start_cost: float = 0.0
        self.route_cost: float = 0.0
        self.problem: Optional[pulp.LpProblem] = None

    def add_parking_lot(self, lot_id: str, capacity: int, depot_distance: float = 0.0, position: Tuple[float, float] = (0.0, 0.0)):
        """Add a parking lot to the network with optional depot distance and position."""
        node_id = f"lot_{lot_id}"
        self.nodes[node_id] = NetworkNode(lot_id, 'parking_lot', capacity, position)
        if hasattr(self, 'lot_depot_distances'):
            self.lot_depot_distances[node_id] = depot_distance

    def add_office(self, office_id: str, desk_count: int, depot_distance: float = 0.0, position: Tuple[float, float] = (0.0, 0.0)):
        """Add an office to the network with optional depot distance and position."""
        node_id = f"office_{office_id}"
        self.nodes[node_id] = NetworkNode(office_id, 'office', desk_count, position)
        if hasattr(self, 'office_depot_distances'):
            self.office_depot_distances[node_id] = depot_distance

    def set_route_cost(self, from_node: str, to_node: str, cost: float):
        """Set the cost for a shuttle route between two nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Both nodes must exist in network.")
        self.cost_matrix[(from_node, to_node)] = cost

    def _get_lots(self) -> List[str]:
        """Get cached list of parking lot node IDs."""
        if self._lots_cache is None:
            self._lots_cache = [n for n in self.nodes if self.nodes[n].node_type == 'parking_lot']
        return self._lots_cache

    def _get_offices(self) -> List[str]:
        """Get cached list of office node IDs."""
        if self._offices_cache is None:
            self._offices_cache = [n for n in self.nodes if self.nodes[n].node_type == 'office']
        return self._offices_cache

    def _get_depot_distance(self, node_id: str) -> float:
        """Get cached depot distance for a node."""
        if self._depot_distances_cache is None:
            self._depot_distances_cache = {}
            for nid in self.nodes:
                if self.nodes[nid].node_type == 'parking_lot':
                    self._depot_distances_cache[nid] = self.lot_depot_distances.get(nid, 0.0)
                else:
                    self._depot_distances_cache[nid] = self.office_depot_distances.get(nid, 0.0)
        return self._depot_distances_cache.get(node_id, 0.0)

    def _euclidean_distance(self, node1: str, node2: str) -> float:
        """Calculate Euclidean distance between two nodes."""
        import math
        pos1 = self.nodes[node1].position
        pos2 = self.nodes[node2].position
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def generate_cost_matrix(self,
                            lot_to_office_cost: float = 10.0,
                            lot_to_lot_cost_per_unit: float = 1.0,
                            office_to_office_cost: float = 5.0,
                            use_distance_based_lot_costs: bool = True):
        """Generate a complete cost matrix. If use_distance_based_lot_costs=True,
        lot-to-lot costs are based on Euclidean distance."""
        lots = self._get_lots()
        offices = self._get_offices()

        for lot in lots:
            for office in offices:
                if (lot, office) not in self.cost_matrix:
                    self.cost_matrix[(lot, office)] = lot_to_office_cost

        for lot1 in lots:
            for lot2 in lots:
                if lot1 != lot2 and (lot1, lot2) not in self.cost_matrix:
                    if use_distance_based_lot_costs:
                        distance = self._euclidean_distance(lot1, lot2)
                        self.cost_matrix[(lot1, lot2)] = distance * lot_to_lot_cost_per_unit
                    else:
                        self.cost_matrix[(lot1, lot2)] = lot_to_lot_cost_per_unit

        for office1 in offices:
            for office2 in offices:
                if office1 != office2 and (office1, office2) not in self.cost_matrix:
                    self.cost_matrix[(office1, office2)] = office_to_office_cost

    def solve(self, verbose: bool = True):
        """Solve the optimization problem with multiple bus types."""
        lots = self._get_lots()
        offices = self._get_offices()

        total_capacity = sum(self.nodes[n].capacity for n in lots)
        total_demand = sum(self.nodes[n].capacity for n in offices)
        if total_capacity != total_demand:
            raise ValueError(f"Balance violation: {total_capacity} != {total_demand}")

        self.problem = pulp.LpProblem("Shuttle_Service", pulp.LpMinimize)

        # Decision variables: X[a,b,t] for each bus type, P[a,b] for passengers
        X = {}
        P = {}

        for (from_node, to_node) in self.cost_matrix:
            # Create variable for each bus type
            for bus_type in self.bus_types:
                X[(from_node, to_node, bus_type.name)] = pulp.LpVariable(
                    f"X_{from_node}_to_{to_node}_{bus_type.name}", lowBound=0, cat='Integer'
                )

            # Passenger flow (aggregate)
            P[(from_node, to_node)] = pulp.LpVariable(
                f"P_{from_node}_to_{to_node}", lowBound=0, cat='Integer'
            )

        # Objective: minimize total cost = route costs + depot starting costs
        route_costs = pulp.lpSum([
            self.cost_matrix[(a, b)] * bus_type.cost_multiplier * X[(a, b, bus_type.name)]
            for (a, b) in self.cost_matrix
            for bus_type in self.bus_types
        ])

        # Depot costs: charged PER BUS (bikes exempt)
        # Pre-compute depot distances for all nodes (performance optimization)
        depot_dist_cache = {
            n: (self.lot_depot_distances.get(n, 0.0)
                if self.nodes[n].node_type == 'parking_lot'
                else self.office_depot_distances.get(n, 0.0))
            for n in self.nodes
        }

        depot_costs = pulp.lpSum([
            (self.depot_fixed_cost +
             depot_dist_cache[from_node] *
             self.depot_cost_per_distance * bus_type.cost_multiplier) *
            X[(from_node, to_node, bus_type.name)]
            for from_node in self.nodes
            for to_node in self.nodes
            if (from_node, to_node) in self.cost_matrix
            for bus_type in self.bus_types
            if bus_type.name != "Bike"
        ])

        self.problem += route_costs + depot_costs

        # Capacity constraint: P[a,b] <= sum over bus types of (capacity_t * X[a,b,t])
        for (a, b) in self.cost_matrix:
            total_capacity_on_route = pulp.lpSum([
                bus_type.capacity * X[(a, b, bus_type.name)]
                for bus_type in self.bus_types
            ])
            self.problem += P[(a, b)] <= total_capacity_on_route, f"Capacity_{a}_{b}"

        # Office demand constraints
        for office in offices:
            inflow = pulp.lpSum([
                P[(from_node, office)]
                for from_node in self.nodes
                if (from_node, office) in self.cost_matrix
            ])
            self.problem += inflow == self.nodes[office].capacity, f"Office_demand_{office}"

        # Lot supply constraints
        for lot in lots:
            outflow = pulp.lpSum([
                P[(lot, to_node)]
                for to_node in self.nodes
                if (lot, to_node) in self.cost_matrix
            ])
            inflow = pulp.lpSum([
                P[(from_node, lot)]
                for from_node in self.nodes
                if (from_node, lot) in self.cost_matrix
            ])
            self.problem += outflow == self.nodes[lot].capacity + inflow, f"Lot_supply_{lot}"

        # Solve with performance tuning
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=300,  # 5 minute timeout
            threads=4,      # Use multiple CPU cores
            options=['preprocess on', 'cuts on']  # Enable preprocessing and cuts
        )
        self.problem.solve(solver)

        status = pulp.LpStatus[self.problem.status]
        if status != 'Optimal':
            raise ValueError(f"Solver status: {status}")

        # Extract solution
        self.bus_assignments = {}
        self.passenger_flow = {}

        for (a, b) in self.cost_matrix:
            for bus_type in self.bus_types:
                x_val = X[(a, b, bus_type.name)].varValue
                if x_val and x_val > 0:
                    self.bus_assignments[(a, b, bus_type.name)] = int(x_val)

            p_val = P[(a, b)].varValue
            if p_val and p_val > 0:
                self.passenger_flow[(a, b)] = int(p_val)

        self.total_cost = pulp.value(self.problem.objective)

        # Calculate separate depot and route costs
        self.route_cost = pulp.value(route_costs) if route_costs else 0.0
        self.depot_start_cost = pulp.value(depot_costs) if depot_costs else 0.0

        if verbose:
            self._print_solution()

        return self.bus_assignments

    def _print_solution(self):
        """Print the solution details."""
        print("\n" + "="*80)
        print("SHUTTLE BUS ASSIGNMENT SOLUTION (Single-Commodity)")
        print("="*80)
        print(f"\nTotal Cost: ${self.total_cost:.2f}")
        if self.depot_start_cost > 0:
            print(f"  Route Cost: ${self.route_cost:.2f}")
            print(f"  Depot Start Cost: ${self.depot_start_cost:.2f}")
        print(f"Bus Types: {len(self.bus_types)}")
        for bt in self.bus_types:
            print(f"  - {bt.name}: capacity={bt.capacity}, cost_mult={bt.cost_multiplier}x")
        print(f"Total Buses: {sum(self.bus_assignments.values())}")

        print("\n" + "-"*80)
        print("ROUTES:")
        print("-"*80)

        # Group assignments by route
        routes = {}
        for (from_node, to_node, bus_type), count in self.bus_assignments.items():
            route = (from_node, to_node)
            if route not in routes:
                routes[route] = {}
            routes[route][bus_type] = count

        for (from_node, to_node) in sorted(routes.keys()):
            passengers = self.passenger_flow.get((from_node, to_node), 0)
            bus_details = []
            route_cost = 0

            for bus_type_name, count in routes[(from_node, to_node)].items():
                bus_type = self._bus_type_by_name[bus_type_name]
                cost = self.cost_matrix[(from_node, to_node)] * bus_type.cost_multiplier * count
                route_cost += cost
                bus_details.append(f"{count}x{bus_type_name}")

            bus_str = ", ".join(bus_details)
            print(f"{from_node:20} → {to_node:20}: [{bus_str}], {passengers:3} pax, ${route_cost:.2f}")

        print("="*80 + "\n")

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the solution."""
        if not self.bus_assignments:
            raise ValueError("No solution available. Run solve() first.")

        total_buses = sum(self.bus_assignments.values())

        # Count buses by type
        buses_by_type = {bt.name: 0 for bt in self.bus_types}
        for (_, _, bus_type), count in self.bus_assignments.items():
            buses_by_type[bus_type] += count

        direct_routes = sum(
            count for (from_node, to_node, bus_type), count in self.bus_assignments.items()
            if self.nodes[from_node].node_type == 'parking_lot'
            and self.nodes[to_node].node_type == 'office'
        )

        transfer_routes = sum(
            count for (from_node, to_node, bus_type), count in self.bus_assignments.items()
            if not (self.nodes[from_node].node_type == 'parking_lot'
                    and self.nodes[to_node].node_type == 'office')
        )

        return {
            'total_cost': self.total_cost,
            'total_buses': total_buses,
            'buses_by_type': buses_by_type,
            'direct_routes': direct_routes,
            'transfer_routes': transfer_routes,
            'bus_types': [{'name': bt.name, 'capacity': bt.capacity, 'cost_mult': bt.cost_multiplier}
                         for bt in self.bus_types],
            'num_parking_lots': sum(1 for n in self.nodes.values() if n.node_type == 'parking_lot'),
            'num_offices': sum(1 for n in self.nodes.values() if n.node_type == 'office'),
            'total_passengers': sum(n.capacity for n in self.nodes.values() if n.node_type == 'office')
        }


class MulticommodityShuttleOptimization:
    """
    Multicommodity flow model where each commodity represents workers
    traveling to a specific destination office. Supports multiple bus types.

    Decision variables:
    - X[a,b,t] = number of buses of type t (shared across commodities)
    - P_k[a,b] = passengers of commodity k on arc (a,b)

    Constraints:
    1. Capacity coupling: Σ_k P_k[a,b] <= Σ_t (capacity_t * X[a,b,t])
    2. Commodity flow conservation
    3. Origin-destination demand: D[lot, office]
    """

    def __init__(self, bus_types: Optional[List[BusType]] = None,
                 depot_cost_per_distance: float = 0.0,
                 depot_fixed_cost: float = 0.0):
        # Default: single bus type with capacity 30
        if bus_types is None:
            bus_types = [BusType("Standard", 30, 1.0)]

        self.bus_types = bus_types
        self.nodes: Dict[str, NetworkNode] = {}
        self.cost_matrix: Dict[Tuple[str, str], float] = {}
        self.demand_matrix: Dict[Tuple[str, str], int] = {}

        # Depot parameters for bus starting costs (bikes exempt)
        self.depot_cost_per_distance = depot_cost_per_distance
        self.depot_fixed_cost = depot_fixed_cost
        self.lot_depot_distances: Dict[str, float] = {}
        self.office_depot_distances: Dict[str, float] = {}

        # Performance caches
        self._lots_cache: Optional[List[str]] = None
        self._offices_cache: Optional[List[str]] = None
        self._bus_type_by_name: Dict[str, BusType] = {bt.name: bt for bt in self.bus_types}

        # X is now indexed by (from, to, bus_type)
        self.bus_assignments: Dict[Tuple[str, str, str], int] = {}
        self.commodity_flow: Dict[str, Dict[Tuple[str, str], int]] = {}
        self.total_cost: float = 0.0
        self.depot_start_cost: float = 0.0
        self.route_cost: float = 0.0
        self.problem: Optional[pulp.LpProblem] = None

    def add_parking_lot(self, lot_id: str, capacity: int, depot_distance: float = 0.0, position: Tuple[float, float] = (0.0, 0.0)):
        """Add a parking lot to the network with optional depot distance and position."""
        node_id = f"lot_{lot_id}"
        self.nodes[node_id] = NetworkNode(lot_id, 'parking_lot', capacity, position)
        if hasattr(self, 'lot_depot_distances'):
            self.lot_depot_distances[node_id] = depot_distance

    def add_office(self, office_id: str, desk_count: int, depot_distance: float = 0.0, position: Tuple[float, float] = (0.0, 0.0)):
        """Add an office to the network with optional depot distance and position."""
        node_id = f"office_{office_id}"
        self.nodes[node_id] = NetworkNode(office_id, 'office', desk_count, position)
        if hasattr(self, 'office_depot_distances'):
            self.office_depot_distances[node_id] = depot_distance

    def set_route_cost(self, from_node: str, to_node: str, cost: float):
        """Set the cost for a shuttle route between two nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Both nodes must exist in network.")
        self.cost_matrix[(from_node, to_node)] = cost

    def _get_lots(self) -> List[str]:
        """Get cached list of parking lot node IDs."""
        if self._lots_cache is None:
            self._lots_cache = [n for n in self.nodes if self.nodes[n].node_type == 'parking_lot']
        return self._lots_cache

    def _get_offices(self) -> List[str]:
        """Get cached list of office node IDs."""
        if self._offices_cache is None:
            self._offices_cache = [n for n in self.nodes if self.nodes[n].node_type == 'office']
        return self._offices_cache

    def set_demand(self, lot_id: str, office_id: str, num_workers: int):
        """Set the demand for workers traveling from a lot to an office."""
        lot_node = f"lot_{lot_id}"
        office_node = f"office_{office_id}"

        if lot_node not in self.nodes or office_node not in self.nodes:
            raise ValueError(f"Both lot and office must exist in network.")

        self.demand_matrix[(lot_node, office_node)] = num_workers

    def _euclidean_distance(self, node1: str, node2: str) -> float:
        """Calculate Euclidean distance between two nodes."""
        import math
        pos1 = self.nodes[node1].position
        pos2 = self.nodes[node2].position
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def generate_cost_matrix(self,
                            lot_to_office_cost: float = 10.0,
                            lot_to_lot_cost_per_unit: float = 1.0,
                            office_to_office_cost: float = 5.0,
                            use_distance_based_lot_costs: bool = True):
        """Generate a complete cost matrix. If use_distance_based_lot_costs=True,
        lot-to-lot costs are based on Euclidean distance."""
        lots = self._get_lots()
        offices = self._get_offices()

        for lot in lots:
            for office in offices:
                if (lot, office) not in self.cost_matrix:
                    self.cost_matrix[(lot, office)] = lot_to_office_cost

        for lot1 in lots:
            for lot2 in lots:
                if lot1 != lot2 and (lot1, lot2) not in self.cost_matrix:
                    if use_distance_based_lot_costs:
                        distance = self._euclidean_distance(lot1, lot2)
                        self.cost_matrix[(lot1, lot2)] = distance * lot_to_lot_cost_per_unit
                    else:
                        self.cost_matrix[(lot1, lot2)] = lot_to_lot_cost_per_unit

        for office1 in offices:
            for office2 in offices:
                if office1 != office2 and (office1, office2) not in self.cost_matrix:
                    self.cost_matrix[(office1, office2)] = office_to_office_cost

    def solve(self, verbose: bool = True):
        """Solve the multicommodity optimization problem."""
        lots = self._get_lots()
        offices = self._get_offices()

        # Verify demand balance
        total_demand_by_office = {}
        total_supply_by_lot = {}

        for office in offices:
            total_demand_by_office[office] = sum(
                self.demand_matrix.get((lot, office), 0) for lot in lots
            )

        for lot in lots:
            total_supply_by_lot[lot] = sum(
                self.demand_matrix.get((lot, office), 0) for office in offices
            )

        for office in offices:
            office_capacity = self.nodes[office].capacity
            if total_demand_by_office[office] != office_capacity:
                raise ValueError(
                    f"Office {office}: demand mismatch. "
                    f"Capacity={office_capacity}, Sum of demands={total_demand_by_office[office]}"
                )

        for lot in lots:
            lot_capacity = self.nodes[lot].capacity
            if total_supply_by_lot[lot] != lot_capacity:
                raise ValueError(
                    f"Lot {lot}: supply mismatch. "
                    f"Capacity={lot_capacity}, Sum of demands={total_supply_by_lot[lot]}"
                )

        self.problem = pulp.LpProblem("MultiCommodity_Shuttle", pulp.LpMinimize)

        # Decision variables: X[a,b,t] for each bus type, P_k[a,b] for each commodity
        X = {}
        P = {}

        for (from_node, to_node) in self.cost_matrix:
            # Create variable for each bus type
            for bus_type in self.bus_types:
                X[(from_node, to_node, bus_type.name)] = pulp.LpVariable(
                    f"X_{from_node}_to_{to_node}_{bus_type.name}", lowBound=0, cat='Integer'
                )

        # Commodity flow variables
        for office in offices:
            P[office] = {}
            for (from_node, to_node) in self.cost_matrix:
                P[office][(from_node, to_node)] = pulp.LpVariable(
                    f"P_{office}_{from_node}_to_{to_node}", lowBound=0, cat='Integer'
                )

        # Objective: minimize total cost = route costs + depot starting costs
        route_costs = pulp.lpSum([
            self.cost_matrix[(a, b)] * bus_type.cost_multiplier * X[(a, b, bus_type.name)]
            for (a, b) in self.cost_matrix
            for bus_type in self.bus_types
        ])

        # Depot costs: charged PER BUS (bikes exempt)
        # Pre-compute depot distances for all nodes (performance optimization)
        depot_dist_cache = {
            n: (self.lot_depot_distances.get(n, 0.0)
                if self.nodes[n].node_type == 'parking_lot'
                else self.office_depot_distances.get(n, 0.0))
            for n in self.nodes
        }

        depot_costs = pulp.lpSum([
            (self.depot_fixed_cost +
             depot_dist_cache[from_node] *
             self.depot_cost_per_distance * bus_type.cost_multiplier) *
            X[(from_node, to_node, bus_type.name)]
            for from_node in self.nodes
            for to_node in self.nodes
            if (from_node, to_node) in self.cost_matrix
            for bus_type in self.bus_types
            if bus_type.name != "Bike"
        ])

        self.problem += route_costs + depot_costs

        # Capacity coupling: Σ_k P_k[a,b] <= Σ_t (capacity_t * X[a,b,t])
        for (a, b) in self.cost_matrix:
            total_passengers = pulp.lpSum([
                P[office][(a, b)] for office in offices
            ])
            total_capacity_on_route = pulp.lpSum([
                bus_type.capacity * X[(a, b, bus_type.name)]
                for bus_type in self.bus_types
            ])
            self.problem += total_passengers <= total_capacity_on_route, f"Capacity_{a}_{b}"

        # Commodity flow conservation
        for office_k in offices:
            for lot in lots:
                outflow = pulp.lpSum([
                    P[office_k][(lot, to_node)]
                    for to_node in self.nodes
                    if (lot, to_node) in self.cost_matrix
                ])
                inflow = pulp.lpSum([
                    P[office_k][(from_node, lot)]
                    for from_node in self.nodes
                    if (from_node, lot) in self.cost_matrix
                ])
                demand = self.demand_matrix.get((lot, office_k), 0)
                self.problem += outflow == inflow + demand, \
                    f"Lot_commodity_{lot}_{office_k}"

            for office in offices:
                if office != office_k:
                    outflow = pulp.lpSum([
                        P[office_k][(office, to_node)]
                        for to_node in self.nodes
                        if (office, to_node) in self.cost_matrix
                    ])
                    inflow = pulp.lpSum([
                        P[office_k][(from_node, office)]
                        for from_node in self.nodes
                        if (from_node, office) in self.cost_matrix
                    ])
                    self.problem += outflow == inflow, \
                        f"Office_transfer_commodity_{office}_{office_k}"

            inflow = pulp.lpSum([
                P[office_k][(from_node, office_k)]
                for from_node in self.nodes
                if (from_node, office_k) in self.cost_matrix
            ])
            outflow = pulp.lpSum([
                P[office_k][(office_k, to_node)]
                for to_node in self.nodes
                if (office_k, to_node) in self.cost_matrix
            ])
            self.problem += inflow == self.nodes[office_k].capacity + outflow, \
                f"Destination_office_{office_k}"

        # Solve with performance tuning
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=300,  # 5 minute timeout
            threads=4,      # Use multiple CPU cores
            options=['preprocess on', 'cuts on']  # Enable preprocessing and cuts
        )
        self.problem.solve(solver)

        status = pulp.LpStatus[self.problem.status]
        if status != 'Optimal':
            raise ValueError(f"Solver status: {status}")

        # Extract solution
        self.bus_assignments = {}
        self.commodity_flow = {office: {} for office in offices}

        for (a, b) in self.cost_matrix:
            # Extract bus assignments by type
            for bus_type in self.bus_types:
                x_val = X[(a, b, bus_type.name)].varValue
                if x_val and x_val > 0:
                    self.bus_assignments[(a, b, bus_type.name)] = int(x_val)

            # Extract commodity flows
            for office in offices:
                p_val = P[office][(a, b)].varValue
                if p_val and p_val > 0:
                    self.commodity_flow[office][(a, b)] = int(p_val)

        self.total_cost = pulp.value(self.problem.objective)

        # Calculate separate depot and route costs
        self.route_cost = pulp.value(route_costs) if route_costs else 0.0
        self.depot_start_cost = pulp.value(depot_costs) if depot_costs else 0.0

        if verbose:
            self._print_solution()

        return self.bus_assignments

    def _print_solution(self):
        """Print the solution details."""
        offices = self._get_offices()
        lots = self._get_lots()

        print("\n" + "="*85)
        print("MULTICOMMODITY SHUTTLE BUS ASSIGNMENT SOLUTION")
        print("="*85)
        print(f"\nTotal Cost: ${self.total_cost:.2f}")
        if self.depot_start_cost > 0:
            print(f"  Route Cost: ${self.route_cost:.2f}")
            print(f"  Depot Start Cost: ${self.depot_start_cost:.2f}")
        print(f"Bus Types: {len(self.bus_types)}")
        for bt in self.bus_types:
            print(f"  - {bt.name}: capacity={bt.capacity}, cost_mult={bt.cost_multiplier}x")
        print(f"Total Buses: {sum(self.bus_assignments.values())}")
        print(f"Number of Commodities: {len(offices)}")

        print("\n" + "-"*85)
        print("BUS ROUTES (with commodity breakdown):")
        print("-"*85)

        # Group assignments by route
        routes = {}
        for (from_node, to_node, bus_type), count in self.bus_assignments.items():
            route = (from_node, to_node)
            if route not in routes:
                routes[route] = {}
            routes[route][bus_type] = count

        for (from_node, to_node) in sorted(routes.keys()):
            # Bus details
            bus_details = []
            route_cost = 0
            for bus_type_name, count in routes[(from_node, to_node)].items():
                bus_type = self._bus_type_by_name[bus_type_name]
                cost = self.cost_matrix[(from_node, to_node)] * bus_type.cost_multiplier * count
                route_cost += cost
                bus_details.append(f"{count}x{bus_type_name}")

            # Commodity loads
            commodity_loads = []
            total_pax = 0
            for office in offices:
                pax = self.commodity_flow[office].get((from_node, to_node), 0)
                if pax > 0:
                    office_label = office.replace('office_', 'O')
                    commodity_loads.append(f"{pax}→{office_label}")
                    total_pax += pax

            bus_str = ", ".join(bus_details)
            commodity_str = ", ".join(commodity_loads) if commodity_loads else "empty"

            print(f"{from_node:20} → {to_node:20}: [{bus_str}], "
                  f"{total_pax:3} pax [{commodity_str}], ${route_cost:.2f}")

        print("\n" + "-"*85)
        print("DEMAND MATRIX (Origin-Destination pairs):")
        print("-"*85)

        for lot in lots:
            demands = []
            for office in offices:
                demand = self.demand_matrix.get((lot, office), 0)
                if demand > 0:
                    office_label = office.replace('office_', 'O')
                    demands.append(f"{demand}→{office_label}")
            if demands:
                print(f"{lot:20}: {', '.join(demands)}")

        print("="*85 + "\n")

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the solution."""
        if not self.bus_assignments:
            raise ValueError("No solution available. Run solve() first.")

        total_buses = sum(self.bus_assignments.values())

        offices = self._get_offices()
        lots = self._get_lots()

        # Count buses by type
        buses_by_type = {bt.name: 0 for bt in self.bus_types}
        for (_, _, bus_type), count in self.bus_assignments.items():
            buses_by_type[bus_type] += count

        direct_routes = sum(
            count for (from_node, to_node, bus_type), count in self.bus_assignments.items()
            if self.nodes[from_node].node_type == 'parking_lot'
            and self.nodes[to_node].node_type == 'office'
        )

        transfer_routes = sum(
            count for (from_node, to_node, bus_type), count in self.bus_assignments.items()
            if not (self.nodes[from_node].node_type == 'parking_lot'
                    and self.nodes[to_node].node_type == 'office')
        )

        return {
            'total_cost': self.total_cost,
            'total_buses': total_buses,
            'buses_by_type': buses_by_type,
            'direct_routes': direct_routes,
            'transfer_routes': transfer_routes,
            'bus_types': [{'name': bt.name, 'capacity': bt.capacity, 'cost_mult': bt.cost_multiplier}
                         for bt in self.bus_types],
            'num_parking_lots': len(lots),
            'num_offices': len(offices),
            'num_commodities': len(offices),
            'total_passengers': sum(n.capacity for n in self.nodes.values() if n.node_type == 'office')
        }


class MultiTemporalShuttleOptimization:
    """
    Multi-temporal multicommodity flow model with three time periods:
    - Morning: Buses start from depot, workers travel lots → offices
    - Lunch: Bidirectional office ↔ lot movement, buses carry over from morning
    - Evening: Workers return offices → lots, buses return to depot

    Decision variables:
    - X[a,b,bus_type,period] = number of buses of type traveling a→b in period
    - P_k[a,b,period] = passengers of commodity k on arc (a,b) in period

    Key constraints:
    - Bus inventory linking: buses ending at location i in period t start from i in period t+1
    - Depot costs: charged for buses starting from lots (morning) and returning to lots (evening)
    - Capacity coupling per period: Σ_k P_k[a,b,t] <= Σ_bus_type (capacity × X[a,b,bus_type,t])
    """

    def __init__(self, bus_types: Optional[List[BusType]] = None,
                 depot_cost_per_distance: float = 1.0,
                 depot_fixed_cost: float = 0.0,
                 periods: Optional[List[str]] = None):
        """
        Initialize multi-temporal optimization model.

        Args:
            bus_types: List of bus types with capacities and cost multipliers
            depot_cost_per_distance: Cost per unit distance from depot
            depot_fixed_cost: Fixed cost per bus hired (e.g., driver daily wage)
            periods: List of period names (default: ['morning', 'lunch', 'evening'])
        """
        if bus_types is None:
            bus_types = [BusType("Standard", 30, 1.0)]

        if periods is None:
            periods = ['morning', 'lunch', 'evening']

        self.bus_types = bus_types
        self.periods = periods
        self.nodes: Dict[str, NetworkNode] = {}
        self.lot_depot_distances: Dict[str, float] = {}
        self.office_depot_distances: Dict[str, float] = {}
        self.depot_cost_per_distance = depot_cost_per_distance
        self.depot_fixed_cost = depot_fixed_cost

        # Performance caches
        self._lots_cache: Optional[List[str]] = None
        self._offices_cache: Optional[List[str]] = None
        self._bus_type_by_name: Dict[str, BusType] = {bt.name: bt for bt in self.bus_types}

        # Cost matrix and demand matrix indexed by period
        self.cost_matrix: Dict[str, Dict[Tuple[str, str], float]] = {p: {} for p in periods}
        # Demand matrix: {period: {(from, to, commodity): count}}
        self.demand_matrix: Dict[str, Dict[Tuple[str, str, str], int]] = {p: {} for p in periods}

        # Solution storage
        self.bus_assignments: Dict[Tuple[str, str, str, str], int] = {}  # (from, to, bus_type, period)
        self.parked_buses: Dict[Tuple[str, str, str], int] = {}  # (location, bus_type, period)
        self.commodity_flow: Dict[str, Dict[str, Dict[Tuple[str, str], int]]] = {}  # {period: {office: {(a,b): flow}}}
        self.total_cost: float = 0.0
        self.depot_start_cost: float = 0.0  # Morning
        self.depot_return_cost: float = 0.0  # Evening
        self.route_cost_by_period: Dict[str, float] = {}
        self.problem: Optional[pulp.LpProblem] = None

    def add_parking_lot(self, lot_id: str, capacity: int, depot_distance: float = 0.0, position: Tuple[float, float] = (0.0, 0.0)):
        """Add a parking lot with depot distance and position."""
        node_id = f"lot_{lot_id}"
        self.nodes[node_id] = NetworkNode(lot_id, 'parking_lot', capacity, position)
        self.lot_depot_distances[node_id] = depot_distance

    def add_office(self, office_id: str, desk_count: int, depot_distance: float = 0.0, position: Tuple[float, float] = (0.0, 0.0)):
        """Add an office to the network with optional depot distance and position."""
        node_id = f"office_{office_id}"
        self.nodes[node_id] = NetworkNode(office_id, 'office', desk_count, position)
        if hasattr(self, 'office_depot_distances'):
            self.office_depot_distances[node_id] = depot_distance

    def _get_lots(self) -> List[str]:
        """Get cached list of parking lot node IDs."""
        if self._lots_cache is None:
            self._lots_cache = [n for n in self.nodes if self.nodes[n].node_type == 'parking_lot']
        return self._lots_cache

    def _get_offices(self) -> List[str]:
        """Get cached list of office node IDs."""
        if self._offices_cache is None:
            self._offices_cache = [n for n in self.nodes if self.nodes[n].node_type == 'office']
        return self._offices_cache

    def set_route_cost(self, from_node: str, to_node: str, cost: float, period: Optional[str] = None):
        """
        Set route cost for specific period or all periods.

        Args:
            from_node: Origin node
            to_node: Destination node
            cost: Route cost
            period: Specific period (if None, applies to all periods)
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Both nodes must exist in network.")

        periods_to_set = [period] if period else self.periods
        for p in periods_to_set:
            self.cost_matrix[p][(from_node, to_node)] = cost

    def set_demand(self, from_node: str, to_node: str, num_workers: int, period: str, commodity: Optional[str] = None):
        """
        Set demand for workers traveling from→to in specific period.

        Args:
            from_node: Origin (lot or office)
            to_node: Destination (office or lot)
            num_workers: Number of workers
            period: Time period
            commodity: Which office these workers belong to (auto-inferred if None)
                      For morning lot→office: commodity = destination office
                      For lunch/evening: must specify which office workers belong to
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Both nodes must exist in network.")
        if period not in self.periods:
            raise ValueError(f"Period {period} not in {self.periods}")

        # Auto-infer commodity for morning lot→office flows
        if commodity is None:
            if (self.nodes[from_node].node_type == 'parking_lot' and
                self.nodes[to_node].node_type == 'office'):
                commodity = to_node  # Workers going to office become that commodity
            else:
                raise ValueError(
                    f"Cannot auto-infer commodity for {from_node}→{to_node}. "
                    "Please specify commodity parameter (which office these workers belong to)."
                )

        # Store as (from, to, commodity) tuple
        key = (from_node, to_node, commodity)
        if key not in self.demand_matrix[period]:
            self.demand_matrix[period][key] = 0
        self.demand_matrix[period][key] = num_workers

    def _euclidean_distance(self, node1: str, node2: str) -> float:
        """Calculate Euclidean distance between two nodes."""
        import math
        pos1 = self.nodes[node1].position
        pos2 = self.nodes[node2].position
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def generate_cost_matrix(self,
                            lot_to_office_cost: float = 10.0,
                            lot_to_lot_cost_per_unit: float = 1.0,
                            office_to_office_cost: float = 5.0,
                            office_to_lot_cost: float = 10.0,
                            use_distance_based_lot_costs: bool = True):
        """Generate complete cost matrices for all periods. If use_distance_based_lot_costs=True,
        lot-to-lot costs are based on Euclidean distance."""
        lots = self._get_lots()
        offices = self._get_offices()

        for period in self.periods:
            # Lot → Office
            for lot in lots:
                for office in offices:
                    if (lot, office) not in self.cost_matrix[period]:
                        self.cost_matrix[period][(lot, office)] = lot_to_office_cost

            # Lot → Lot
            for lot1 in lots:
                for lot2 in lots:
                    if lot1 != lot2 and (lot1, lot2) not in self.cost_matrix[period]:
                        if use_distance_based_lot_costs:
                            distance = self._euclidean_distance(lot1, lot2)
                            self.cost_matrix[period][(lot1, lot2)] = distance * lot_to_lot_cost_per_unit
                        else:
                            self.cost_matrix[period][(lot1, lot2)] = lot_to_lot_cost_per_unit

            # Office → Office
            for office1 in offices:
                for office2 in offices:
                    if office1 != office2 and (office1, office2) not in self.cost_matrix[period]:
                        self.cost_matrix[period][(office1, office2)] = office_to_office_cost

            # Office → Lot (for lunch and evening)
            for office in offices:
                for lot in lots:
                    if (office, lot) not in self.cost_matrix[period]:
                        self.cost_matrix[period][(office, lot)] = office_to_lot_cost

    def solve(self, verbose: bool = True):
        """Solve the multi-temporal multicommodity optimization problem."""
        lots = self._get_lots()
        offices = self._get_offices()

        self.problem = pulp.LpProblem("MultiTemporal_Shuttle", pulp.LpMinimize)

        # Decision variables
        X = {}  # X[from, to, bus_type, period]
        P = {}  # P[office_commodity][from, to, period]
        Park = {}  # Park[location, bus_type, period] = number of buses parked

        # Create bus assignment variables for each period
        for period in self.periods:
            for (from_node, to_node) in self.cost_matrix[period]:
                for bus_type in self.bus_types:
                    X[(from_node, to_node, bus_type.name, period)] = pulp.LpVariable(
                        f"X_{from_node}_{to_node}_{bus_type.name}_{period}",
                        lowBound=0, cat='Integer'
                    )

        # Create parking variables (for all intermediate periods, not first or last)
        all_locations = list(lots) + list(offices)
        for period_idx in range(len(self.periods)):
            period = self.periods[period_idx]
            for location in all_locations:
                for bus_type in self.bus_types:
                    Park[(location, bus_type.name, period)] = pulp.LpVariable(
                        f"Park_{location}_{bus_type.name}_{period}",
                        lowBound=0, cat='Integer'
                    )

        # Create commodity flow variables for each period
        # Optimization: Only create variables for feasible commodity-route combinations
        for period in self.periods:
            P[period] = {}
            for office in offices:
                P[period][office] = {}
                for (from_node, to_node) in self.cost_matrix[period]:
                    from_type = self.nodes[from_node].node_type
                    to_type = self.nodes[to_node].node_type

                    # Skip lot-to-lot flows (office workers don't transfer between lots)
                    if from_type == 'parking_lot' and to_type == 'parking_lot':
                        continue

                    # Create variable for this feasible commodity-route pair
                    P[period][office][(from_node, to_node)] = pulp.LpVariable(
                        f"P_{office}_{from_node}_{to_node}_{period}",
                        lowBound=0, cat='Integer'
                    )

        # OBJECTIVE FUNCTION
        # Route costs across all periods
        route_costs = {}
        for period in self.periods:
            route_costs[period] = pulp.lpSum([
                self.cost_matrix[period][(a, b)] * bus_type.cost_multiplier *
                X[(a, b, bus_type.name, period)]
                for (a, b) in self.cost_matrix[period]
                for bus_type in self.bus_types
            ])

        total_route_costs = pulp.lpSum([route_costs[p] for p in self.periods])

        # Pre-compute depot distances for all nodes (performance optimization)
        depot_dist_cache = {
            n: (self.lot_depot_distances.get(n, 0.0)
                if self.nodes[n].node_type == 'parking_lot'
                else self.office_depot_distances.get(n, 0.0))
            for n in self.nodes
        }

        # Depot starting costs (morning) - charged PER BUS, bikes exempt
        # All buses in the system must come from depot in the morning
        depot_start_costs = pulp.lpSum([
            (self.depot_fixed_cost +
             depot_dist_cache[from_node] *
             self.depot_cost_per_distance * bus_type.cost_multiplier) *
            X[(from_node, to_node, bus_type.name, self.periods[0])]
            for from_node in self.nodes
            for to_node in self.nodes
            if (from_node, to_node) in self.cost_matrix[self.periods[0]]
            for bus_type in self.bus_types
            if bus_type.name != "Bike"  # Bikes don't incur depot costs
        ])

        # Depot return costs (evening) - charged PER BUS, bikes exempt (no fixed cost, only distance cost)
        # All buses must return to depot in the evening
        depot_return_costs = pulp.lpSum([
            depot_dist_cache[to_node] *
            self.depot_cost_per_distance * bus_type.cost_multiplier *
            X[(from_node, to_node, bus_type.name, self.periods[-1])]
            for from_node in self.nodes
            for to_node in self.nodes
            if (from_node, to_node) in self.cost_matrix[self.periods[-1]]
            for bus_type in self.bus_types
            if bus_type.name != "Bike"  # Bikes don't incur depot costs
        ])

        # No parking costs - buses can remain where they are for free
        self.problem += total_route_costs + depot_start_costs + depot_return_costs

        # CONSTRAINTS

        # 1. Capacity coupling per period
        for period in self.periods:
            for (a, b) in self.cost_matrix[period]:
                # Only sum over commodities that have variables for this route
                total_passengers = pulp.lpSum([
                    P[period][office][(a, b)]
                    for office in offices
                    if (a, b) in P[period][office]
                ])
                total_capacity = pulp.lpSum([
                    bus_type.capacity * X[(a, b, bus_type.name, period)]
                    for bus_type in self.bus_types
                ])
                self.problem += total_passengers <= total_capacity, \
                    f"Capacity_{a}_{b}_{period}"

        # 2. Commodity flow conservation for each period
        # Commodity k = workers assigned to office_k
        # Demand format: (from, to, commodity_office_k)

        # Pre-compute supply dictionary for performance (avoids O(n^4) loop)
        supply_by_period_commodity_node = {}
        for period in self.periods:
            for office_k in offices:
                supply_by_period_commodity_node[(period, office_k)] = {}
                for (from_n, to_n, comm), demand in self.demand_matrix[period].items():
                    if comm == office_k:
                        # Add supply at origin node
                        if from_n not in supply_by_period_commodity_node[(period, office_k)]:
                            supply_by_period_commodity_node[(period, office_k)][from_n] = 0
                        supply_by_period_commodity_node[(period, office_k)][from_n] += demand

                        # Subtract demand at destination node
                        if to_n not in supply_by_period_commodity_node[(period, office_k)]:
                            supply_by_period_commodity_node[(period, office_k)][to_n] = 0
                        supply_by_period_commodity_node[(period, office_k)][to_n] -= demand

        # Build flow conservation constraints
        for period in self.periods:
            for office_k in offices:
                # At each node: inflow + supply = outflow
                for node in self.nodes:
                    # Inflow of commodity k to this node (only for routes with variables)
                    inflow = pulp.lpSum([
                        P[period][office_k][(from_node, node)]
                        for from_node in self.nodes
                        if (from_node, node) in self.cost_matrix[period]
                        and (from_node, node) in P[period][office_k]
                    ])

                    # Outflow of commodity k from this node (only for routes with variables)
                    outflow = pulp.lpSum([
                        P[period][office_k][(node, to_node)]
                        for to_node in self.nodes
                        if (node, to_node) in self.cost_matrix[period]
                        and (node, to_node) in P[period][office_k]
                    ])

                    # Get pre-computed supply for this node
                    supply = supply_by_period_commodity_node.get((period, office_k), {}).get(node, 0)

                    # Flow conservation: inflow + supply = outflow
                    self.problem += inflow + supply == outflow, \
                        f"Flow_conservation_{node}_{office_k}_{period}"

        # 3. Bus inventory linking between periods (with parking)
        # General formula: arriving[t] + parked_from_previous[t-1] = departing[t] + parked_to_next[t]
        # Special cases:
        #  - First period: buses can originate from depot (at lots only)
        #  - Last period: buses can terminate at depot (at lots only)

        for period_idx in range(len(self.periods)):
            period = self.periods[period_idx]
            is_first = (period_idx == 0)
            is_last = (period_idx == len(self.periods) - 1)

            for location in all_locations:
                is_lot = (self.nodes[location].node_type == 'parking_lot')

                for bus_type in self.bus_types:
                    # Buses arriving at this location in this period
                    arriving = pulp.lpSum([
                        X[(from_node, location, bus_type.name, period)]
                        for from_node in self.nodes
                        if (from_node, location) in self.cost_matrix[period]
                    ])

                    # Buses departing from this location in this period
                    departing = pulp.lpSum([
                        X[(location, to_node, bus_type.name, period)]
                        for to_node in self.nodes
                        if (location, to_node) in self.cost_matrix[period]
                    ])

                    # Buses parked from previous period
                    if period_idx > 0:
                        prev_period = self.periods[period_idx - 1]
                        parked_from_prev = Park[(location, bus_type.name, prev_period)]
                    else:
                        parked_from_prev = 0

                    # Buses parked for next period
                    parked_to_next = Park[(location, bus_type.name, period)]

                    # Conservation: what arrives + what was parked = what departs + what parks
                    # SPECIAL: In first period at lots, buses can come from depot (not modeled as arriving)
                    # SPECIAL: In last period at lots, buses can go to depot (not modeled as departing)

                    if is_first and is_lot:
                        # First period at lot: buses may start from depot, so don't enforce arriving = departing
                        # Just ensure parked buses are accounted for
                        self.problem += arriving + parked_from_prev >= parked_to_next, \
                            f"Bus_inventory_first_{location}_{bus_type.name}_{period}"
                    elif is_last and is_lot:
                        # Last period at lot: buses may return to depot, so allow departing without arriving
                        self.problem += arriving + parked_from_prev >= departing + parked_to_next, \
                            f"Bus_inventory_last_{location}_{bus_type.name}_{period}"
                    else:
                        # Normal inventory conservation
                        self.problem += arriving + parked_from_prev == departing + parked_to_next, \
                            f"Bus_inventory_{location}_{bus_type.name}_{period}"

        # 4. Fleet balance: buses starting in morning = buses returning in evening
        for bus_type in self.bus_types:
            total_starting = pulp.lpSum([
                X[(lot, to_node, bus_type.name, self.periods[0])]
                for lot in lots
                for to_node in self.nodes
                if (lot, to_node) in self.cost_matrix[self.periods[0]]
            ])
            total_returning = pulp.lpSum([
                X[(from_node, lot, bus_type.name, self.periods[-1])]
                for lot in lots
                for from_node in self.nodes
                if (from_node, lot) in self.cost_matrix[self.periods[-1]]
            ])
            self.problem += total_starting == total_returning, \
                f"Fleet_balance_{bus_type.name}"

        # SOLVE with performance tuning
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=600,  # 10 minute timeout (multi-temporal is more complex)
            threads=4,      # Use multiple CPU cores
            options=['preprocess on', 'cuts on']  # Enable preprocessing and cuts
        )
        self.problem.solve(solver)

        status = pulp.LpStatus[self.problem.status]
        if status != 'Optimal':
            raise ValueError(f"Solver status: {status}")

        # EXTRACT SOLUTION
        self.bus_assignments = {}
        self.parked_buses = {}
        self.commodity_flow = {period: {office: {} for office in offices} for period in self.periods}

        for period in self.periods:
            for (a, b) in self.cost_matrix[period]:
                # Extract bus assignments
                for bus_type in self.bus_types:
                    x_val = X[(a, b, bus_type.name, period)].varValue
                    if x_val and x_val > 0:
                        self.bus_assignments[(a, b, bus_type.name, period)] = int(x_val)

                # Extract commodity flows (only for routes with variables)
                for office in offices:
                    if (a, b) in P[period][office]:
                        p_val = P[period][office][(a, b)].varValue
                        if p_val and p_val > 0:
                            self.commodity_flow[period][office][(a, b)] = int(p_val)

        # Extract parked buses
        for location in all_locations:
            for bus_type in self.bus_types:
                for period in self.periods:
                    park_val = Park[(location, bus_type.name, period)].varValue
                    if park_val and park_val > 0:
                        self.parked_buses[(location, bus_type.name, period)] = int(park_val)

        self.total_cost = pulp.value(self.problem.objective)

        # Calculate costs by component
        self.route_cost_by_period = {
            period: pulp.value(route_costs[period]) if route_costs[period] else 0.0
            for period in self.periods
        }
        self.depot_start_cost = pulp.value(depot_start_costs) if depot_start_costs else 0.0
        self.depot_return_cost = pulp.value(depot_return_costs) if depot_return_costs else 0.0

        if verbose:
            self._print_solution()

        return self.bus_assignments

    def _print_solution(self):
        """Print the multi-temporal solution details."""
        offices = self._get_offices()

        print("\n" + "="*90)
        print("MULTI-TEMPORAL SHUTTLE BUS ASSIGNMENT SOLUTION")
        print("="*90)
        print(f"\nTotal Cost: ${self.total_cost:.2f}")
        print(f"  Depot Start Cost (Morning): ${self.depot_start_cost:.2f}")
        total_route_cost = sum(self.route_cost_by_period.values())
        print(f"  Route Costs (All Periods): ${total_route_cost:.2f}")
        for period in self.periods:
            print(f"    {period.capitalize()}: ${self.route_cost_by_period[period]:.2f}")
        print(f"  Depot Return Cost (Evening): ${self.depot_return_cost:.2f}")

        print(f"\nBus Types: {len(self.bus_types)}")
        for bt in self.bus_types:
            print(f"  - {bt.name}: capacity={bt.capacity}, cost_mult={bt.cost_multiplier}x")

        # Calculate total fleet size (buses used in morning = buses returned in evening)
        total_fleet = sum(
            count for (from_node, _, _, period), count in self.bus_assignments.items()
            if period == self.periods[0] and self.nodes[from_node].node_type == 'parking_lot'
        )
        print(f"Total Fleet Size: {total_fleet} buses")

        # Print solution by period
        for period in self.periods:
            print("\n" + "-"*90)
            print(f"PERIOD: {period.upper()}")
            print("-"*90)

            # Group assignments by route for this period
            period_assignments = {
                (a, b, bt): count
                for (a, b, bt, p), count in self.bus_assignments.items()
                if p == period
            }

            if not period_assignments:
                print("  (No bus movements in this period)")
                continue

            # Group by route
            routes = {}
            for (from_node, to_node, bus_type), count in period_assignments.items():
                route = (from_node, to_node)
                if route not in routes:
                    routes[route] = {}
                routes[route][bus_type] = count

            for (from_node, to_node) in sorted(routes.keys()):
                # Bus details
                bus_details = []
                route_cost = 0
                for bus_type_name, count in routes[(from_node, to_node)].items():
                    bus_type = self._bus_type_by_name[bus_type_name]
                    cost = self.cost_matrix[period][(from_node, to_node)] * bus_type.cost_multiplier * count
                    route_cost += cost
                    bus_details.append(f"{count}x{bus_type_name}")

                # Commodity loads
                commodity_loads = []
                total_pax = 0
                for office in offices:
                    pax = self.commodity_flow[period][office].get((from_node, to_node), 0)
                    if pax > 0:
                        office_label = office.replace('office_', 'O')
                        commodity_loads.append(f"{pax}→{office_label}")
                        total_pax += pax

                bus_str = ", ".join(bus_details)
                commodity_str = ", ".join(commodity_loads) if commodity_loads else "empty"

                print(f"  {from_node:18} → {to_node:18}: [{bus_str:20}], "
                      f"{total_pax:3} pax [{commodity_str:25}], ${route_cost:.2f}")

            # Show parked buses for this period
            period_parked = {
                (loc, bt): count
                for (loc, bt, p), count in self.parked_buses.items()
                if p == period
            }

            if period_parked:
                print(f"\n  PARKED BUSES ({period}):")
                for (location, bus_type), count in sorted(period_parked.items()):
                    loc_label = location.replace('office_', 'O').replace('lot_', 'L')
                    print(f"    {loc_label:15}: {count}x{bus_type}")

        # Print demand summary
        print("\n" + "-"*90)
        print("DEMAND SUMMARY BY PERIOD:")
        print("-"*90)
        for period in self.periods:
            print(f"\n{period.upper()}:")
            demands_exist = False
            for (from_node, to_node, commodity), demand in self.demand_matrix[period].items():
                if demand > 0:
                    demands_exist = True
                    to_label = to_node.replace('office_', 'O').replace('lot_', 'L')
                    from_label = from_node.replace('office_', 'O').replace('lot_', 'L')
                    comm_label = commodity.replace('office_', 'O')
                    print(f"  {from_label:15} → {to_label:15}: {demand:3} workers (commodity: {comm_label})")
            if not demands_exist:
                print("  (No demand in this period)")

        print("="*90 + "\n")

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the multi-temporal solution."""
        if not self.bus_assignments:
            raise ValueError("No solution available. Run solve() first.")

        offices = self._get_offices()
        lots = self._get_lots()

        # Total fleet (count buses starting in morning)
        total_fleet = sum(
            count for (from_node, _, _, period), count in self.bus_assignments.items()
            if period == self.periods[0] and self.nodes[from_node].node_type == 'parking_lot'
        )

        # Buses by type (across all periods)
        buses_by_type_period = {}
        for period in self.periods:
            buses_by_type_period[period] = {bt.name: 0 for bt in self.bus_types}
            for (_, _, bus_type, p), count in self.bus_assignments.items():
                if p == period:
                    buses_by_type_period[period][bus_type] += count

        # Fleet composition
        fleet_by_type = {bt.name: 0 for bt in self.bus_types}
        for (from_node, _, bus_type, period), count in self.bus_assignments.items():
            if period == self.periods[0] and self.nodes[from_node].node_type == 'parking_lot':
                fleet_by_type[bus_type] += count

        return {
            'total_cost': self.total_cost,
            'depot_start_cost': self.depot_start_cost,
            'depot_return_cost': self.depot_return_cost,
            'route_cost_by_period': self.route_cost_by_period,
            'total_fleet_size': total_fleet,
            'fleet_by_type': fleet_by_type,
            'buses_by_type_period': buses_by_type_period,
            'periods': self.periods,
            'bus_types': [{'name': bt.name, 'capacity': bt.capacity, 'cost_mult': bt.cost_multiplier}
                         for bt in self.bus_types],
            'num_parking_lots': len(lots),
            'num_offices': len(offices),
            'num_commodities': len(offices),
        }


class StochasticShuttleOptimization(ShuttleOptimization):
    """
    Stochastic single-commodity flow model with passenger miss probability.

    Extends ShuttleOptimization by incorporating stochasticity in bus capacity.
    Passengers have probability p of missing the bus, creating capacity uncertainty.

    Two formulation options:
    1. Expected value: Use expected effective capacity = capacity × (1 - p)
    2. Chance constraint: Ensure capacity is sufficient with probability α

    Mathematical formulation:
    - For each bus type t with nominal capacity C_t:
      * Expected passengers boarding = min(demand, C_t × (1 - p))
      * With binomial model: X ~ Binomial(C_t, 1-p)
      * For large C_t: X ≈ Normal(C_t(1-p), C_t*p*(1-p))

    Decision variables (inherited from ShuttleOptimization):
    - X[a,b,t] = number of buses of type t from node a to node b
    - P[a,b] = aggregate passenger flow from a to b

    New constraints:
    - Capacity with chance constraint: P[a,b] ≤ Σ_t (effective_capacity_t(α) × X[a,b,t])
      where effective_capacity_t(α) = C_t(1-p) - z_α√(C_t*p*(1-p))
    """

    def __init__(self,
                 bus_types: Optional[List[BusType]] = None,
                 depot_cost_per_distance: float = 0.0,
                 depot_fixed_cost: float = 0.0,
                 miss_probability: float = 0.1,
                 approach: str = 'expected',
                 confidence_level: float = 0.95):
        """
        Initialize stochastic shuttle optimization model.

        Args:
            bus_types: List of bus types with capacities and cost multipliers
            depot_cost_per_distance: Cost per unit distance from depot
            depot_fixed_cost: Fixed cost per bus hired
            miss_probability: Probability p that each passenger misses the bus (0 ≤ p < 1)
            approach: 'expected' (use expected capacity) or 'chance' (chance constraint)
            confidence_level: For chance constraint approach, probability that capacity is sufficient (α)
        """
        super().__init__(bus_types, depot_cost_per_distance, depot_fixed_cost)

        if not 0 <= miss_probability < 1:
            raise ValueError("miss_probability must be in [0, 1)")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if approach not in ['expected', 'chance']:
            raise ValueError("approach must be 'expected' or 'chance'")

        self.miss_probability = miss_probability
        self.approach = approach
        self.confidence_level = confidence_level
        self.effective_capacities: Dict[str, float] = {}

        # Calculate effective capacities for each bus type
        self._calculate_effective_capacities()

    def _calculate_effective_capacities(self):
        """
        Calculate effective capacity for each bus type based on approach.

        Expected value approach:
            effective_capacity = C × (1 - p)

        Chance constraint approach:
            Use normal approximation to binomial:
            P(successful_passengers ≥ k) ≥ α
            => k ≤ C(1-p) - z_α√(C*p*(1-p))

            where z_α is the α-quantile of standard normal
        """
        p = self.miss_probability

        for bus_type in self.bus_types:
            C = bus_type.capacity

            if self.approach == 'expected':
                # Expected value: E[X] = C(1-p) where X ~ Binomial(C, 1-p)
                self.effective_capacities[bus_type.name] = C * (1 - p)

            elif self.approach == 'chance':
                # Chance constraint with normal approximation
                # Number of passengers successfully boarding ~ Binomial(C, 1-p)
                # Approximated as Normal(μ, σ²) where:
                #   μ = C(1-p)
                #   σ² = C*p*(1-p)

                mean = C * (1 - p)
                variance = C * p * (1 - p)
                std_dev = np.sqrt(variance)

                # For capacity to be sufficient with probability α,
                # we need P(X ≥ demand) ≥ α
                # Using normal approximation: demand ≤ μ - z_(1-α) * σ
                # where z_(1-α) is the (1-α) quantile

                z_score = stats.norm.ppf(self.confidence_level)
                effective_cap = mean - z_score * std_dev

                # Ensure non-negative
                self.effective_capacities[bus_type.name] = max(0, effective_cap)

    def solve(self, verbose: bool = True):
        """
        Solve the stochastic optimization problem using effective capacities.

        The solve method is nearly identical to the deterministic version,
        but uses effective_capacities instead of nominal capacities in the
        capacity constraints.
        """
        lots = self._get_lots()
        offices = self._get_offices()

        total_capacity = sum(self.nodes[n].capacity for n in lots)
        total_demand = sum(self.nodes[n].capacity for n in offices)
        if total_capacity != total_demand:
            raise ValueError(f"Balance violation: {total_capacity} != {total_demand}")

        self.problem = pulp.LpProblem("Stochastic_Shuttle_Service", pulp.LpMinimize)

        # Decision variables: X[a,b,t] for each bus type, P[a,b] for passengers
        X = {}
        P = {}

        for (from_node, to_node) in self.cost_matrix:
            # Create variable for each bus type
            for bus_type in self.bus_types:
                X[(from_node, to_node, bus_type.name)] = pulp.LpVariable(
                    f"X_{from_node}_to_{to_node}_{bus_type.name}", lowBound=0, cat='Integer'
                )

            # Passenger flow (aggregate)
            P[(from_node, to_node)] = pulp.LpVariable(
                f"P_{from_node}_to_{to_node}", lowBound=0, cat='Integer'
            )

        # Objective: minimize total cost = route costs + depot starting costs
        route_costs = pulp.lpSum([
            self.cost_matrix[(a, b)] * bus_type.cost_multiplier * X[(a, b, bus_type.name)]
            for (a, b) in self.cost_matrix
            for bus_type in self.bus_types
        ])

        # Depot costs: charged PER BUS (bikes exempt)
        depot_dist_cache = {
            n: (self.lot_depot_distances.get(n, 0.0)
                if self.nodes[n].node_type == 'parking_lot'
                else self.office_depot_distances.get(n, 0.0))
            for n in self.nodes
        }

        depot_costs = pulp.lpSum([
            (self.depot_fixed_cost +
             depot_dist_cache[from_node] *
             self.depot_cost_per_distance * bus_type.cost_multiplier) *
            X[(from_node, to_node, bus_type.name)]
            for from_node in self.nodes
            for to_node in self.nodes
            if (from_node, to_node) in self.cost_matrix
            for bus_type in self.bus_types
            if bus_type.name != "Bike"
        ])

        self.problem += route_costs + depot_costs

        # STOCHASTIC CAPACITY CONSTRAINT: Use effective capacities
        for (a, b) in self.cost_matrix:
            total_capacity_on_route = pulp.lpSum([
                self.effective_capacities[bus_type.name] * X[(a, b, bus_type.name)]
                for bus_type in self.bus_types
            ])
            self.problem += P[(a, b)] <= total_capacity_on_route, f"Stochastic_Capacity_{a}_{b}"

        # Office demand constraints (unchanged)
        for office in offices:
            inflow = pulp.lpSum([
                P[(from_node, office)]
                for from_node in self.nodes
                if (from_node, office) in self.cost_matrix
            ])
            self.problem += inflow == self.nodes[office].capacity, f"Office_demand_{office}"

        # Lot supply constraints (unchanged)
        for lot in lots:
            outflow = pulp.lpSum([
                P[(lot, to_node)]
                for to_node in self.nodes
                if (lot, to_node) in self.cost_matrix
            ])
            inflow = pulp.lpSum([
                P[(from_node, lot)]
                for from_node in self.nodes
                if (from_node, lot) in self.cost_matrix
            ])
            self.problem += outflow == self.nodes[lot].capacity + inflow, f"Lot_supply_{lot}"

        # Solve with performance tuning
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=300,
            threads=4,
            options=['preprocess on', 'cuts on']
        )
        self.problem.solve(solver)

        status = pulp.LpStatus[self.problem.status]
        if status != 'Optimal':
            raise ValueError(f"Solver status: {status}")

        # Extract solution
        self.bus_assignments = {}
        self.passenger_flow = {}

        for (a, b) in self.cost_matrix:
            for bus_type in self.bus_types:
                x_val = X[(a, b, bus_type.name)].varValue
                if x_val and x_val > 0:
                    self.bus_assignments[(a, b, bus_type.name)] = int(x_val)

            p_val = P[(a, b)].varValue
            if p_val and p_val > 0:
                self.passenger_flow[(a, b)] = int(p_val)

        self.total_cost = pulp.value(self.problem.objective)

        # Calculate separate depot and route costs
        self.route_cost = pulp.value(route_costs) if route_costs else 0.0
        self.depot_start_cost = pulp.value(depot_costs) if depot_costs else 0.0

        if verbose:
            self._print_solution()

        return self.bus_assignments

    def _print_solution(self):
        """Print the stochastic solution details with capacity information."""
        print("\n" + "="*80)
        print("STOCHASTIC SHUTTLE BUS ASSIGNMENT SOLUTION")
        print("="*80)
        print(f"\nMiss Probability (p): {self.miss_probability:.2%}")
        print(f"Approach: {self.approach}")
        if self.approach == 'chance':
            print(f"Confidence Level (α): {self.confidence_level:.2%}")

        print(f"\nTotal Cost: ${self.total_cost:.2f}")
        if self.depot_start_cost > 0:
            print(f"  Route Cost: ${self.route_cost:.2f}")
            print(f"  Depot Start Cost: ${self.depot_start_cost:.2f}")

        print(f"\nBus Types (with effective capacities):")
        for bt in self.bus_types:
            nominal = bt.capacity
            effective = self.effective_capacities[bt.name]
            reduction = (1 - effective/nominal) * 100 if nominal > 0 else 0
            print(f"  - {bt.name}: nominal_cap={nominal}, effective_cap={effective:.1f} "
                  f"({reduction:.1f}% reduction), cost_mult={bt.cost_multiplier}x")

        print(f"Total Buses: {sum(self.bus_assignments.values())}")

        print("\n" + "-"*80)
        print("ROUTES:")
        print("-"*80)

        # Group assignments by route
        routes = {}
        for (from_node, to_node, bus_type), count in self.bus_assignments.items():
            route = (from_node, to_node)
            if route not in routes:
                routes[route] = {}
            routes[route][bus_type] = count

        for (from_node, to_node) in sorted(routes.keys()):
            passengers = self.passenger_flow.get((from_node, to_node), 0)
            bus_details = []
            route_cost = 0
            nominal_cap = 0
            effective_cap = 0

            for bus_type_name, count in routes[(from_node, to_node)].items():
                bus_type = self._bus_type_by_name[bus_type_name]
                cost = self.cost_matrix[(from_node, to_node)] * bus_type.cost_multiplier * count
                route_cost += cost
                nominal_cap += bus_type.capacity * count
                effective_cap += self.effective_capacities[bus_type_name] * count
                bus_details.append(f"{count}x{bus_type_name}")

            bus_str = ", ".join(bus_details)
            print(f"{from_node:20} → {to_node:20}: [{bus_str}], {passengers:3} pax, "
                  f"cap: {effective_cap:.1f}/{nominal_cap}, ${route_cost:.2f}")

        print("="*80 + "\n")

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics including stochastic parameters."""
        if not self.bus_assignments:
            raise ValueError("No solution available. Run solve() first.")

        stats_dict = super().get_summary_statistics()

        # Add stochastic-specific information
        stats_dict['miss_probability'] = self.miss_probability
        stats_dict['approach'] = self.approach
        stats_dict['confidence_level'] = self.confidence_level if self.approach == 'chance' else None
        stats_dict['effective_capacities'] = self.effective_capacities.copy()

        # Calculate average capacity reduction
        total_nominal = sum(bt.capacity for bt in self.bus_types)
        total_effective = sum(self.effective_capacities.values())
        stats_dict['avg_capacity_reduction'] = (1 - total_effective / total_nominal) if total_nominal > 0 else 0

        return stats_dict
