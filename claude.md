# Shuttle Service Optimization - Complete Project Documentation

**Course:** IOE 612 - Stochastic Processes in Operations Research
**Last Updated:** December 9, 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current Model Configuration](#current-model-configuration)
3. [Quick Start](#quick-start)
4. [Model Variants](#model-variants)
5. [Mathematical Formulations](#mathematical-formulations)
6. [Multiple Bus Types](#multiple-bus-types)
7. [Multi-Temporal Optimization](#multi-temporal-optimization)
8. [Stochastic Optimization](#stochastic-optimization)
9. [Depot Cost Modeling](#depot-cost-modeling)
10. [Usage Examples](#usage-examples)
11. [Future Work](#future-work)

---

## Project Overview

This project implements **three optimization models** for shuttle bus routing and assignment in an office park setting, using minimum cost flow formulations:

1. **Stochastic Capacity Model**: Incorporates uncertainty in bus capacity due to passengers missing buses (expected value approach)
2. **Multicommodity Model**: Tracks individual worker destinations, enforces origin-destination requirements
3. **Multi-Temporal Model**: Handles multiple time periods (morning/lunch/evening) with bus inventory tracking

All models minimize total bus operating costs while satisfying capacity constraints and demand requirements.

---

## Current Model Configuration

### 2x4 Network (2 Offices × 4 Parking Lots)

The project focuses on a standard test case with 260 workers:

| Component | Configuration |
|-----------|---------------|
| **Offices** | Office 1 (150 desks), Office 2 (110 desks) |
| **Parking Lots** | Lot A (80 cap, 5mi), Lot B (70 cap, 3mi), Lot C (60 cap, 7mi), Lot D (50 cap, 4mi) |
| **Bus Types** | Bike (1 pax, 1.1×), Stub (10 pax, 1.0×), Medium (30 pax, 2.5×), Long (70 pax, 4.0×) |
| **Cost Structure** | Lot↔Office: $10, Lot↔Lot: $5, Office↔Office: $5 |
| **Depot Costs** | Fixed: $50/bus, Variable: $1/mi × cost_multiplier |

### Model Comparison Results

| Model | Total Cost | Fleet Size | Fleet Composition |
|-------|------------|------------|-------------------|
| **Stochastic Capacity (p=20%)** | $634 | 11 vehicles | 5 bikes, 2 medium, 4 long buses |
| **Multicommodity** | $624 | 11 vehicles | 5 bikes, 6 long buses |
| **Multi-Temporal** | $1017 | 6 vehicles | 1 bike, 1 stub, 4 long |

**Key Insights:**
- Stochastic capacity handles 20% miss rate by reducing effective capacity
- Multicommodity enforces specific origin-destination constraints
- Multi-temporal uses only 6 vehicles through reuse across morning/lunch/evening periods

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
pulp>=2.7
numpy>=1.20
matplotlib>=3.5
networkx>=2.6
```

### Run All Models

```bash
# Run ALL three examples with comparison (RECOMMENDED)
python run_all_examples.py

# Run individual examples
python run_all_examples.py 1        # Example 1: Stochastic Capacity (p=20%)
python run_all_examples.py 2        # Example 2: Multicommodity
python run_all_examples.py 3        # Example 3: Multi-Temporal
```

### Generate Visualizations

```bash
# Generate all visualizations at once (RECOMMENDED)
python generate_all_plots.py                  # All 9 visualizations

# Example 1: Stochastic capacity visualizations
python visualize_stochastic.py                # Sensitivity analysis (p ∈ [0%, 30%])
python visualize_stochastic_routes.py         # Route assignments and utilization

# Other visualizations
python visualize_2x4_campus.py                # Campus layout diagram
python visualize_routing_matrices_clean.py    # Routing matrices comparison
python create_cost_heatmaps.py                # Cost matrices heatmaps
```

---

## Model Variants

### When to Use Each Model

**Use Stochastic Capacity Model When:**
- Passengers have uncertain arrival times (miss probability p)
- Bus capacity is effectively reduced due to no-shows
- Want to plan for expected capacity utilization
- Cost minimization is the primary objective
- Occasional capacity shortages are acceptable

**Use Multicommodity Model When:**
- Workers have fixed office assignments
- Need to track individual O-D flows
- Feasibility is critical (must satisfy specific demands)
- Generating realistic operational plans

**Use Multi-Temporal Model When:**
- Need to plan across multiple time periods
- Bus inventory and positioning matters
- Morning/lunch/evening demand patterns vary
- Want to minimize total daily fleet size

---

## Mathematical Formulations

### Stochastic Capacity Model

**Key Feature:** Bus capacity is uncertain - effective capacity = nominal × (1-p)

See [Stochastic Optimization](#stochastic-optimization) section for complete mathematical formulation.

### Multicommodity Model

**Decision Variables:**
- `X[a,b,t]` = number of buses of type t (shared across commodities)
- `P_k[a,b]` = flow of commodity k (workers going to office k)

**Objective:**
```
minimize: Σ_a Σ_b Σ_t (c[a,b] × cost_mult[t] × X[a,b,t])
        + depot_costs
```

**Constraints:**

1. **Capacity coupling:**
```
Σ_k P_k[a,b] ≤ Σ_t (capacity[t] × X[a,b,t])
```

2. **Commodity flow conservation** (for each node n, commodity k):
```
Σ_a P_k[a,n] + supply_k[n] = Σ_b P_k[n,b]
```

3. **Origin-destination demand:**
```
D[lot, office] workers from each lot to each office
```

4. **Non-negativity:**
```
X[a,b,t] ≥ 0, integer
P_k[a,b] ≥ 0
```

**Key Difference:** Multicommodity tracks where each worker wants to go via demand matrix `D[lot, office]`.

### Multi-Temporal Model

**Decision Variables:**
- `X[a,b,t,p]` = number of buses of type t on route (a,b) in period p
- `P_k[a,b,p]` = passengers of commodity k on route (a,b) in period p

**Objective:**
```
minimize:
  // Route costs across all periods
  Σ_p Σ_a Σ_b Σ_t (c[a,b] × cost_mult[t] × X[a,b,t,p])

  // Depot starting costs (morning)
  + Σ_lot Σ_t (depot_fixed + depot_dist[lot] × depot_rate × cost_mult[t]) × X[lot,*,t,morning]

  // Depot return costs (evening)
  + Σ_lot Σ_t (depot_dist[lot] × depot_rate × cost_mult[t]) × X[*,lot,t,evening]
```

**Constraints:**

1. **Capacity coupling (per period):**
```
Σ_k P_k[a,b,p] ≤ Σ_t (capacity[t] × X[a,b,t,p])
```

2. **Commodity flow conservation (per period, per commodity):**
```
For each node n, commodity k, period p:
  inflow[n,k,p] + supply[n,k,p] = outflow[n,k,p]
```

3. **Bus inventory linking (between periods):**
```
For each location i, bus_type t:
  Σ_a X[a,i,t,morning] = Σ_b X[i,b,t,lunch]
  Σ_a X[a,i,t,lunch] = Σ_b X[i,b,t,evening]
```
Buses arriving at location i in period p must depart from i in period p+1.

4. **Fleet balance:**
```
For each bus_type t:
  (buses starting morning) = (buses returning in evening)
```

---

## Multiple Bus Types

### Bus Type Definition

Each bus type has three properties:
- **Name**: Descriptive identifier (e.g., "Stub Bus", "Medium Bus", "Long Bus")
- **Capacity**: Number of passengers the bus can carry
- **Cost Multiplier**: Relative operating cost (multiplied by base route cost)

### Standard Configuration

```python
from shuttle_optimization import BusType

bus_types = [
    BusType("Bike", 1, 1.1),           # Individual transport, 1.1x cost
    BusType("Stub Bus", 10, 1.0),      # Small bus, base cost
    BusType("Medium Bus", 30, 2.5),    # Medium capacity, 2.5x cost
    BusType("Long Bus", 70, 4.0)       # Large capacity, 4x cost
]
```

**Capacity per Dollar Analysis:**
- Bike: 1 pax / $1.1 = 0.91 pax/$ ← Least efficient
- Stub Bus: 10 pax / $1.0 = 10.0 pax/$
- Medium Bus: 30 pax / $2.5 = 12.0 pax/$
- Long Bus: 70 pax / $4.0 = 17.5 pax/$ ← Most efficient!

**Note:** Bikes are exempt from depot costs (presumed already distributed on campus).

### Implementation

The optimizer automatically selects the most cost-effective mix of bus types:

```python
# Decision variable: X[a,b,bus_type]
# Objective coefficient: base_cost[a,b] × cost_multiplier[bus_type]
# Capacity contribution: capacity[bus_type] × X[a,b,bus_type]
```

---

## Multi-Temporal Optimization

### Three Time Periods

**Morning Period:**
- Workers commute from parking lots → offices
- Buses start from depot, travel to lots, then to offices
- High volume, mostly direct lot→office routes

**Lunch Period:**
- Some workers leave campus (office→lot) for lunch/errands
- Some workers travel between offices for meetings
- Lower volume, bidirectional flows

**Evening Period:**
- Workers return from offices → parking lots
- Buses eventually return to depot from lots
- High volume, reverse of morning pattern

### Commodity Tracking

**Commodity k = workers assigned to office k**

This definition remains consistent across all periods:
- Morning: Office_1 workers travel lot→office_1 (commodity = office_1)
- Lunch: Office_1 workers might leave (office_1→lot) but still commodity = office_1
- Evening: Office_1 workers return home (office_1→lot) still commodity = office_1

### Demand Specification

```python
# Morning lot→office flows: commodity auto-inferred
sim.set_demand("lot_A", "office_1", 120, "morning")
# Commodity automatically set to "office_1"

# All other flows: must explicitly specify commodity
sim.set_demand("office_1", "lot_A", 20, "lunch", commodity="office_1")
sim.set_demand("office_1", "lot_A", 100, "evening", commodity="office_1")
```

### Demand Balance

Commodity flows must balance across the day:

```python
# Morning: 100 workers arrive at office_1
sim.set_demand("lot_A", "office_1", 100, "morning")

# Lunch: 20 leave campus
sim.set_demand("office_1", "lot_A", 20, "lunch", commodity="office_1")

# Evening: Only 80 return (100 arrived - 20 left = 80 remain)
sim.set_demand("office_1", "lot_A", 80, "evening", commodity="office_1")  # ✓ Balanced
```

---

## Stochastic Optimization

### Overview

The **Stochastic Shuttle Optimization** model extends the single-commodity model by incorporating **capacity uncertainty** due to passengers missing buses. This introduces realistic operational risk where not all passengers show up on time.

**Key Feature:** Tuneable miss probability parameter `p` (0 ≤ p < 1)
- Each passenger has probability `p` of missing the bus
- Number of successful boardings follows a binomial distribution
- Model adapts fleet size to handle uncertainty

### Mathematical Foundation

#### Binomial Passenger Arrival Model

For a bus with nominal capacity `C`, if each passenger misses with probability `p`:
- Number of successful boardings: `X ~ Binomial(C, 1-p)`
- Expected value: `E[X] = C(1-p)`
- Variance: `Var[X] = C·p·(1-p)`

For large `C`, use **normal approximation**:
```
X ≈ Normal(μ = C(1-p), σ² = C·p·(1-p))
```

### Modeling Approach: Expected Value

**Concept:** Use expected effective capacity in constraints

**Formulation:**
```
Effective capacity: C_eff = C × (1 - p)
```

**Characteristics:**
- Risk-neutral: plans for average performance
- Optimizes for expected passenger arrivals
- Suitable when occasional capacity shortages are acceptable
- Computationally efficient

**Modified Constraint:**
```
P[a,b] ≤ Σ_t (C_t(1-p) × X[a,b,t])
```

### Complete Mathematical Formulation

#### Sets and Indices
- `N`: Set of nodes (parking lots ∪ offices)
- `T`: Set of bus types
- `(a,b) ∈ A`: Set of arcs (routes)

#### Parameters
- `C_t`: Nominal capacity of bus type t
- `c_{ab}`: Base cost of route (a,b)
- `m_t`: Cost multiplier for bus type t
- `p`: Miss probability (0 ≤ p < 1)
- `d_i^{supply}`: Supply at parking lot i
- `d_j^{demand}`: Demand at office j

#### Decision Variables
- `X_{ab}^t ∈ ℤ_+`: Number of buses of type t on route (a,b)
- `P_{ab} ∈ ℝ_+`: Passenger flow on route (a,b)

#### Objective Function
```
minimize: Σ_{(a,b)∈A} Σ_{t∈T} c_{ab} · m_t · X_{ab}^t + depot_costs
```

#### Constraints

**1. Stochastic Capacity Constraint (Expected Value):**
```
P_{ab} ≤ Σ_{t∈T} [C_t(1-p)] · X_{ab}^t     ∀(a,b) ∈ A
```

**2. Office Demand:**
```
Σ_{a∈N} P_{a,office_j} = d_j^{demand}     ∀ office_j
```

**3. Lot Supply:**
```
Σ_{b∈N} P_{lot_i,b} = d_i^{supply} + Σ_{a∈N} P_{a,lot_i}     ∀ lot_i
```

**4. Non-negativity:**
```
X_{ab}^t ≥ 0, integer     ∀(a,b) ∈ A, ∀t ∈ T
P_{ab} ≥ 0                ∀(a,b) ∈ A
```

### Performance Analysis: 2x4 Network

Using the standard test case (260 workers, p varying from 0% to 30%):

| Miss Prob | Cost    | Buses  | Fleet Composition       | Δ Cost vs p=0% |
|-----------|---------|--------|-------------------------|----------------|
| 0%        | $501    | 5      | 1S, 4L                  | -              |
| 5%        | $551    | 10     | 5B, 1M, 4L              | +10.0%         |
| 10%       | $568    | 13     | 8B, 1M, 4L              | +13.4%         |
| 15%       | $614    | 11     | 5B, 1S, 1M, 4L          | +22.6%         |
| 20%       | $634    | 11     | 5B, 2M, 4L              | +26.5%         |
| 25%       | $683    | 7      | 1S, 1M, 5L              | +36.3%         |
| 30%       | $732    | 9      | 2B, 2M, 5L              | +46.1%         |

**Legend:** B=Bike, S=Stub Bus, M=Medium Bus, L=Long Bus

**Key Insights:**
1. **Cost increases steadily** with miss probability (10% to 46% increase)
2. **Fleet size varies** as optimizer balances smaller vs larger buses
3. **Fleet composition changes** - mix of bus types adapts to effective capacity
4. **Bikes and medium buses** become more common at intermediate miss probabilities
5. At higher miss probabilities, **larger buses are preferred** despite reduced capacity

### Effective Capacity Reductions

Example with p=15% (expected value approach):

| Bus Type   | Nominal Capacity | Effective Capacity | Capacity Reduction |
|------------|------------------|--------------------|--------------------|
| Bike       | 1                | 0.85               | 15.0%              |
| Stub Bus   | 10               | 8.50               | 15.0%              |
| Medium Bus | 30               | 25.50              | 15.0%              |
| Long Bus   | 70               | 59.50              | 15.0%              |

**Observation:** Expected value approach reduces all capacities proportionally by p%.

### Implementation

```python
from shuttle_optimization import StochasticShuttleOptimization, BusType

bus_types = [
    BusType("Bike", 1, 1.1),
    BusType("Stub Bus", 10, 1.0),
    BusType("Medium Bus", 30, 2.5),
    BusType("Long Bus", 70, 4.0)
]

# Expected value approach
sim = StochasticShuttleOptimization(
    bus_types=bus_types,
    depot_cost_per_distance=1.0,
    depot_fixed_cost=50.0,
    miss_probability=0.15,  # 15% miss rate
    approach='expected'
)

# Build network
sim.add_parking_lot("A", capacity=80, depot_distance=5.0)
sim.add_parking_lot("B", capacity=70, depot_distance=3.0)
sim.add_parking_lot("C", capacity=60, depot_distance=7.0)
sim.add_parking_lot("D", capacity=50, depot_distance=4.0)
sim.add_office("1", desk_count=150)
sim.add_office("2", desk_count=110)

sim.generate_cost_matrix(
    lot_to_office_cost=10.0,
    lot_to_lot_cost=5.0,
    office_to_office_cost=5.0
)
sim.solve(verbose=True)

# Access results
stats = sim.get_summary_statistics()
print(f"Total Cost: ${stats['total_cost']:.2f}")
print(f"Total Buses: {stats['total_buses']}")
print(f"Fleet: {stats['buses_by_type']}")
print(f"Average capacity reduction: {stats['avg_capacity_reduction']:.1%}")
```

### Running the Demonstration

```bash
# Run comprehensive stochastic analysis
python stochastic_demo.py
```

This script:
1. Tests expected value approach across p ∈ [0%, 5%, ..., 30%]
2. Shows cost and fleet size progression
3. Displays detailed solution for p=15%
4. Demonstrates how capacity uncertainty affects optimization

### When to Use This Approach

**Use Stochastic Capacity Model When:**
- Passengers have uncertain arrival times (miss probability p)
- Bus capacity is effectively reduced due to no-shows
- You want to plan for expected capacity utilization
- Occasional capacity shortages are acceptable
- Operating in non-critical transportation applications
- Cost minimization is the primary objective

**Don't Use When:**
- All passengers are guaranteed to arrive on time
- Capacity must be guaranteed with high confidence
- Operating critical transportation (medical, emergency)
- Regulatory requirements mandate strict capacity guarantees

### Relation to Course Material (IOE 612)

This stochastic extension demonstrates:
1. **Stochastic Programming**: Optimization under capacity uncertainty
2. **Expected Value Approach**: Risk-neutral decision making
3. **Binomial Processes**: Passenger arrival modeling
4. **Network Flow under Uncertainty**: Stochastic capacity constraints

---

## Depot Cost Modeling

### Per-Bus Linear Approach

Depot costs are charged **per individual bus** (not per location/bus-type combination).

**Formula:**
```
depot_cost_per_bus = fixed_cost + distance × cost_per_distance × cost_multiplier
```

**Where:**
- `fixed_cost` = $50 per bus (morning starts only)
- `cost_per_distance` = $1 per mile
- `cost_multiplier` = bus type's cost multiplier
- `distance` = miles from lot to depot

### Example Calculation

**Scenario:** 2 Long Buses and 1 Medium Bus start from Lot A (5 miles from depot)
- Long Bus: cost_multiplier = 4.0
- Medium Bus: cost_multiplier = 2.5

```
Long Bus #1:  ($50 + 5×$1×4.0) = $70.00
Long Bus #2:  ($50 + 5×$1×4.0) = $70.00
Medium Bus:   ($50 + 5×$1×2.5) = $62.50
Total:                          $202.50
```

### Multi-Temporal Depot Costs

**Morning (Start):**
```
depot_start_cost = Σ (fixed_cost + distance × rate × cost_mult) × X[lot, *, bus_type, morning]
```

**Evening (Return):**
```
depot_return_cost = Σ (distance × rate × cost_mult) × X[*, lot, bus_type, evening]
```
(No fixed cost on return - just distance cost)

**Special Case - Bikes:**
Bikes do NOT incur depot costs (presumed already distributed on campus).

---

## Usage Examples

### Example 1: Stochastic Capacity Model

```python
from shuttle_optimization import StochasticShuttleOptimization, BusType

# Define bus types
bus_types = [
    BusType("Bike", 1, 1.1),
    BusType("Stub Bus", 10, 1.0),
    BusType("Medium Bus", 30, 2.5),
    BusType("Long Bus", 70, 4.0)
]

# Create model with stochastic capacity (expected value approach)
sim = StochasticShuttleOptimization(
    bus_types=bus_types,
    depot_cost_per_distance=1.0,
    depot_fixed_cost=50.0,
    miss_probability=0.20,    # 20% miss rate reduces capacity
    approach='expected'
)

# Add network
sim.add_parking_lot("A", capacity=80, depot_distance=5.0)
sim.add_parking_lot("B", capacity=70, depot_distance=3.0)
sim.add_parking_lot("C", capacity=60, depot_distance=7.0)
sim.add_parking_lot("D", capacity=50, depot_distance=4.0)
sim.add_office("1", desk_count=150)
sim.add_office("2", desk_count=110)

# Generate cost matrix
sim.generate_cost_matrix(
    lot_to_office_cost=10.0,
    lot_to_lot_cost=5.0,
    office_to_office_cost=5.0
)

# Solve
sim.solve(verbose=True)

# Get results
stats = sim.get_summary_statistics()
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"Total buses: {stats['total_buses']}")
print(f"Fleet: {stats['buses_by_type']}")
print(f"Capacity reduction: {stats['avg_capacity_reduction']:.1%}")
```

### Example 2: Multicommodity Model

```python
from shuttle_optimization import MulticommodityShuttleOptimization, BusType

bus_types = [
    BusType("Bike", 1, 1.1),
    BusType("Stub Bus", 10, 1.0),
    BusType("Medium Bus", 30, 2.5),
    BusType("Long Bus", 70, 4.0)
]

sim = MulticommodityShuttleOptimization(
    bus_types=bus_types,
    depot_cost_per_distance=1.0,
    depot_fixed_cost=50.0
)

# Add network (same as before)
sim.add_parking_lot("A", capacity=80, depot_distance=5.0)
sim.add_office("1", desk_count=150)
sim.add_office("2", desk_count=110)

# Set demand matrix (where workers want to go)
sim.set_demand("A", "1", 50)  # 50 workers from A to Office 1
sim.set_demand("A", "2", 30)  # 30 workers from A to Office 2

# Generate costs and solve
sim.generate_cost_matrix(
    lot_to_office_cost=10.0,
    lot_to_lot_cost=5.0,
    office_to_office_cost=5.0
)

sim.solve(verbose=True)

# Results include commodity breakdown
print(f"Commodities: {stats['num_commodities']}")
print(f"Direct routes: {stats['direct_routes']}")
print(f"Transfer routes: {stats['transfer_routes']}")
```

### Example 3: Multi-Temporal Model

```python
from shuttle_optimization import MultiTemporalShuttleOptimization, BusType

bus_types = [
    BusType("Bike", 1, 1.1),
    BusType("Stub Bus", 10, 1.0),
    BusType("Medium Bus", 30, 2.5),
    BusType("Long Bus", 70, 4.0)
]

sim = MultiTemporalShuttleOptimization(
    bus_types=bus_types,
    depot_cost_per_distance=1.0,
    depot_fixed_cost=50.0,
    periods=['morning', 'lunch', 'evening']
)

# Add network
sim.add_parking_lot("A", capacity=80, depot_distance=5.0)
sim.add_office("1", desk_count=150)

# Generate costs
sim.generate_cost_matrix(
    lot_to_office_cost=10.0,
    lot_to_lot_cost=5.0,
    office_to_office_cost=5.0,
    office_to_lot_cost=10.0
)

# Set demand for MORNING (lot → office, commodity auto-inferred)
sim.set_demand("lot_A", "office_1", 80, "morning")

# Set demand for LUNCH (must specify commodity)
sim.set_demand("office_1", "lot_A", 10, "lunch", commodity="office_1")

# Set demand for EVENING (must specify commodity)
sim.set_demand("office_1", "lot_A", 70, "evening", commodity="office_1")
# Note: 80 arrived - 10 left = 70 remain (balanced!)

# Solve
sim.solve(verbose=True)

# Results show per-period breakdown
print(f"Total cost: ${sim.total_cost:.2f}")
print(f"Depot start cost (morning): ${sim.depot_start_cost:.2f}")
print(f"Route costs by period: {sim.route_cost_by_period}")
print(f"Depot return cost (evening): ${sim.depot_return_cost:.2f}")

stats = sim.get_summary_statistics()
print(f"Total fleet size: {stats['total_fleet_size']}")
print(f"Fleet by type: {stats['fleet_by_type']}")
print(f"Buses by type per period: {stats['buses_by_type_period']}")
```

---

## Future Work

### Completed Enhancements

#### ✓ Stochasticity in Shuttle Optimization (COMPLETED)

**Status:** Fully implemented with stochastic capacity model

**Implementation Details:**
- Binomial passenger arrival model with miss probability `p`
- Expected value approach: Effective capacity = Nominal × (1-p)
- Full mathematical formulation documented
- Demonstration script with sensitivity analysis across p ∈ [0%, 30%]

**See:** [Stochastic Optimization](#stochastic-optimization) section for complete details

**Files:**
- `shuttle_optimization.py` - StochasticShuttleOptimization class
- `stochastic_demo.py` - Comprehensive demonstration and analysis

### Planned Enhancements (from Project Feedback)

#### 1. Explicit Mathematical Formulation for Final Project

**Requirement:** Write complete mathematical formulations in optimization notation

**Needed for:**
- Multicommodity model
- Multi-temporal model

**Format Requirements:**
- **Sets and Indices**: Define all problem entities
- **Parameters**: List all input data
- **Decision Variables**: Define all variables with domains
- **Objective Function**: Complete mathematical expression
- **Constraints**: All constraints with clear notation
- **Additional Notes**: Sign restrictions, integrality, etc.

### Additional Extensions

1. **Vehicle routing**: Individual bus tracking with route sequences
2. **Bus availability constraints**: Limited fleet size per type
3. **Time windows**: Specific arrival/departure times
4. **Travel time modeling**: Account for bus travel duration
5. **Asymmetric demand**: Handle cases where morning ≠ evening patterns
6. **Multi-day planning**: Optimize schedules over multiple days

---

## File Structure

### Core Implementation
```
shuttle_optimization.py          # Main optimization module
├── BusType                      # Dataclass for bus specifications
├── MulticommodityShuttleOptimization  # Multicommodity model (Example 2)
├── MultiTemporalShuttleOptimization   # Multi-temporal model (Example 3)
└── StochasticShuttleOptimization      # Stochastic capacity model (Example 1)
```

### Main Simulation Script
```
run_all_examples.py             # PRIMARY: Unified script for all 3 examples
├── run_example_1_stochastic_capacity()  # $634, 11 vehicles (p=20%)
├── run_example_2_multicommodity()       # $624, 11 vehicles
└── run_example_3_multitemporal()        # $1017, 6 vehicles
```

### Visualization Scripts
```
generate_all_plots.py                  # Master script - generates all 9 plots

Individual visualization scripts:
├── visualize_stochastic.py            # Sensitivity analysis (Example 1)
├── visualize_stochastic_routes.py     # Route assignments (Example 1)
├── visualize_2x4_campus.py            # Campus layout diagram
├── visualize_routing_matrices_clean.py # Routing comparison (all 3 models)
└── create_cost_heatmaps.py            # Cost matrices heatmaps
```

### Documentation
```
claude.md                      # This file (complete documentation)
README.md                      # Quick reference guide
project_report.tex             # LaTeX report with all formulations and results
requirements.txt              # Python dependencies
```

### Archive
```
archive/
├── project-specs.md                    # Original project specifications
├── initial-approach.md                 # Initial design approach
├── shuttle_simulation.py               # First working implementation
├── two_stage_stochastic.py             # Two-stage stochastic (infeasible)
├── stochastic_demand_model.py          # Stochastic demand model (infeasible)
├── TWO_STAGE_STOCHASTIC_SUMMARY.md     # Two-stage documentation
├── STOCHASTIC_DEMAND_SUMMARY.md        # Stochastic demand documentation
├── CLEANUP_SUMMARY.md                  # Historical cleanup notes
├── STOCHASTIC_MODEL_SUMMARY.md         # Historical stochastic model notes
├── run_all_models.py                   # Old simulation (duplicate)
├── stochastic_demo.py                  # Old demo (replaced by visualize_stochastic.py)
├── visualization_unified.py            # Unused utility functions
└── visualize_multitemporal.py          # Unused utility functions
```

---

## Computational Performance

### Solve Times (Approximate)

| Problem Size | Single-Commodity | Multicommodity | Multi-Temporal |
|--------------|------------------|----------------|----------------|
| Small (2-3 lots, 2 offices) | < 1 sec | < 2 sec | 2-5 sec |
| Medium (4-5 lots, 2-3 offices) | 1-3 sec | 3-5 sec | 5-15 sec |
| Large (6+ lots, 4+ offices) | 5-10 sec | 10-30 sec | 30-60+ sec |

**Solvers Used:** CBC (default) or GLPK via PuLP

---

## Common Pitfalls

### Single-Commodity / Multicommodity

❌ **Supply ≠ Demand**
```python
sim.add_parking_lot("A", capacity=200)
sim.add_office("1", desk_count=300)  # MISMATCH!
```

✅ **Balanced**
```python
sim.add_parking_lot("A", capacity=200)
sim.add_office("1", desk_count=200)  # Balanced
```

❌ **Forgot to generate cost matrix**
```python
# Missing: sim.generate_cost_matrix()
```

✅ **Complete setup**
```python
sim.generate_cost_matrix()
sim.solve()
```

### Multi-Temporal

❌ **Forgot to specify commodity**
```python
# ERROR: This will fail for office→lot flows
sim.set_demand("office_1", "lot_A", 60, "evening")
```

✅ **Specify commodity**
```python
sim.set_demand("office_1", "lot_A", 60, "evening", commodity="office_1")
```

❌ **Demand imbalance**
```python
sim.set_demand("lot_A", "office_1", 100, "morning")
sim.set_demand("office_1", "lot_A", 80, "evening", commodity="office_1")  # Unbalanced!
```

✅ **Account for lunch departures**
```python
sim.set_demand("lot_A", "office_1", 100, "morning")
sim.set_demand("office_1", "lot_A", 20, "lunch", commodity="office_1")  # 20 leave
sim.set_demand("office_1", "lot_A", 80, "evening", commodity="office_1")  # 80 remain ✓
```

---

## References

### Network Flow Theory
- Ford, L. R., & Fulkerson, D. R. (1962). *Flows in Networks*. Princeton University Press.
- Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). *Network Flows*. Prentice Hall.

### Time-Expanded Networks
- Assad, A. A. (1980). "Models for rail transportation." *Transportation Research Part A*, 14(3), 205-220.

### Integer Programming
- Wolsey, L. A. (1998). *Integer Programming*. Wiley-Interscience.

---

## License

MIT License - Academic/Educational use

---

**End of Documentation**

*For questions or issues, refer to the individual markdown files in the project root or run the test examples to see the models in action.*
