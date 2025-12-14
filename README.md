# Shuttle Service Optimization

Optimization models for shuttle bus routing and assignment in an office park setting, using minimum cost flow formulations.

## Overview

This project implements two approaches to shuttle service optimization:

1. **Single-Commodity (Aggregate) Model**: Treats all passengers as homogeneous, optimizes total flow
2. **Multicommodity Model**: Tracks individual worker destinations, enforces origin-destination requirements

Both models minimize total bus operating costs while satisfying capacity constraints and demand requirements.

## Current Model Configuration (2x4 Network)

The project currently focuses on a **2 Offices × 4 Parking Lots** network with 260 workers:

| Component | Configuration |
|-----------|---------------|
| **Offices** | Office 1 (150 desks), Office 2 (110 desks) |
| **Parking Lots** | Lot A (80 cap, 5mi), Lot B (70 cap, 3mi), Lot C (60 cap, 7mi), Lot D (50 cap, 4mi) |
| **Bus Types** | Bike (1 pax, 0.6×), Stub (10 pax, 1.0×), Medium (30 pax, 2.5×), Long (70 pax, 4.0×) |
| **Cost Structure** | Lot↔Office: $10, Lot↔Lot: $5, Office↔Office: $5, Depot: $50 + $1/mi × mult |

**Model Comparison Results:**
- **Single-Commodity**: $496 total, 14 vehicles (10 bikes, 4 long buses)
- **Multicommodity**: $606 total, 16 vehicles (10 bikes, 6 long buses)
- **Multi-Temporal**: $992 total, 11 vehicles (5 bikes, 1 stub, 2 medium, 3 long) across 3 periods

Run the test files to reproduce these results.

**Features:**
- **Multiple bus types** with different capacities and cost multipliers. See [MULTI_BUS_TYPES.md](MULTI_BUS_TYPES.md) for details.
- **Multi-temporal optimization** with three time periods (morning/lunch/evening) and bus inventory tracking. See [MULTITEMPORAL.md](MULTITEMPORAL.md) for details.
- **Depot cost modeling** with fixed and distance-based costs. See [DEPOT_COST_UPDATE.md](DEPOT_COST_UPDATE.md) for details.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all three models (single-commodity, multicommodity, multi-temporal)
python run_all_models.py

# Run individual models
python run_all_models.py single     # Single-commodity only
python run_all_models.py multi      # Multicommodity only
python run_all_models.py temporal   # Multi-temporal only

# Generate all visualizations at once
python generate_all_plots.py

# Or generate individually
python visualize_2x4_campus.py                # Campus layout
python visualize_routing_matrices_clean.py    # Routing matrices comparison
python create_cost_heatmaps.py                # Cost matrices heatmaps

# All plots are saved to plots/ directory
```

## Files

### Core Module

- **`shuttle_optimization.py`** - Main optimization models
  - `ShuttleOptimization` - Single-commodity flow model
  - `MulticommodityShuttleOptimization` - Multicommodity flow model
  - `MultiTemporalShuttleOptimization` - Multi-period multicommodity model

### Current Working Model (2 Offices × 4 Parking Lots)

**Main Simulation:**
- **`run_all_models.py`** - Unified simulation file that runs all three model variants
  - Single-commodity: $496 cost, 14 vehicles
  - Multicommodity: $606 cost, 16 vehicles
  - Multi-temporal: $992 cost, 11 vehicles (3 periods)

**Visualization Scripts:**
- **`generate_all_plots.py`** - Generate all visualizations at once (recommended)
- **`visualize_2x4_campus.py`** - Campus layout diagram
- **`visualize_routing_matrices_clean.py`** - Routing matrices comparison (all 3 models)
- **`create_cost_heatmaps.py`** - Route and depot cost matrix heatmaps
- **`visualize_multitemporal.py`** - Generic multitemporal visualization utilities
- **`visualization_unified.py`** - Generic visualization utilities (works with any model)

**Generated Outputs (saved to plots/ directory):**
- `campus_layout_2x4.png` - Campus network layout
- `routing_matrices_clean.png` - Routing comparison (all 3 models)
- `cost_matrices_heatmap.png` - Combined cost matrices
- `route_cost_matrix.png` - Detailed route costs
- `depot_cost_matrix.png` - Detailed depot costs

### Documentation

- **`README.md`** - This file (quick reference)
- **`claude.md`** - Complete project documentation (read this for full details)
- **`CLEANUP_SUMMARY.md`** - Codebase cleanup summary
- **`requirements.txt`** - Python dependencies

### Directories

- **`archive/`** - Minimal history (3 files: original project specs, initial approach, first implementation)
- **`plots/`** - Generated visualizations (5 PNG files created by visualization scripts)

## Mathematical Formulation

### Single-Commodity Model

**Decision Variables:**
- `X[a,b]` = number of buses from node a to node b
- `P[a,b]` = aggregate passenger flow from a to b

**Objective:**
```
minimize: Σ c[a,b] × X[a,b]
```

**Constraints:**
1. Capacity: `P[a,b] ≤ bus_capacity × X[a,b]`
2. Office demand: `Σ P[*, office] = desk_count`
3. Lot supply: `Σ P[lot, *] = capacity + Σ P[*, lot]`

### Multicommodity Model

**Decision Variables:**
- `X[a,b]` = number of buses (shared across commodities)
- `P_k[a,b]` = flow of commodity k (workers going to office k)

**Objective:**
```
minimize: Σ c[a,b] × X[a,b]
```

**Constraints:**
1. Capacity coupling: `Σ_k P_k[a,b] ≤ bus_capacity × X[a,b]`
2. Commodity flow conservation at all nodes
3. O-D demand: `D[lot, office]` workers from each lot to each office

**Key Difference:** Multicommodity model tracks where each worker wants to go via demand matrix `D[lot, office]`.

## Usage Examples

### Single-Commodity Model

```python
from shuttle_optimization import ShuttleOptimization

# Create model
sim = ShuttleOptimization(bus_capacity=30)

# Add network
sim.add_parking_lot("A", capacity=150)
sim.add_parking_lot("B", capacity=100)
sim.add_office("1", desk_count=180)
sim.add_office("2", desk_count=120)

# Set costs
sim.set_route_cost("lot_A", "office_1", 8.0)
sim.set_route_cost("lot_A", "office_2", 12.0)
sim.generate_cost_matrix()  # Fill in defaults for other routes

# Solve
sim.solve(verbose=True)

# Get results
print(f"Total cost: ${sim.total_cost}")
stats = sim.get_summary_statistics()
```

### Multicommodity Model

```python
from shuttle_optimization import MulticommodityShuttleOptimization

# Create model
sim = MulticommodityShuttleOptimization(bus_capacity=30)

# Add network
sim.add_parking_lot("A", capacity=150)
sim.add_office("1", desk_count=180)
sim.add_office("2", desk_count=120)

# Set demand matrix (where workers want to go)
sim.set_demand("A", "1", 120)  # 120 workers from A to Office 1
sim.set_demand("A", "2", 30)   # 30 workers from A to Office 2

# Set costs and solve
sim.generate_cost_matrix()
sim.solve(verbose=True)

# Results include commodity breakdown
print(f"Commodity flows: {sim.commodity_flow}")
```

### Visualization

```python
from visualization_unified import (visualize_network, plot_demand_matrix,
                                   plot_commodity_flows)
import matplotlib.pyplot as plt

# Visualize network
fig, ax = visualize_network(sim, show_assignments=True)
plt.savefig('plots/network.png', dpi=300, bbox_inches='tight')

# For multicommodity: visualize demand matrix
fig, ax = plot_demand_matrix(sim)
plt.savefig('plots/demand.png', dpi=300, bbox_inches='tight')

# Visualize commodity-specific flows
fig = plot_commodity_flows(sim)
plt.savefig('plots/flows.png', dpi=300, bbox_inches='tight')
```

## Example Output

### Single-Commodity Model

```
Total Cost: $89.00
Total Buses: 11

ROUTES:
lot_A → office_1: 5 buses, 150 pax, $40.00
lot_B → office_2: 2 buses,  60 pax, $20.00
lot_C → office_2: 2 buses,  60 pax, $14.00
...
```

### Multicommodity Model

```
Total Cost: $96.00
Total Buses: 11
Commodities: 2

BUS ROUTES (with commodity breakdown):
lot_A → office_1: 4 buses, 120 pax [120→O1], $32.00
lot_A → office_2: 1 bus,    30 pax [30→O2],  $12.00
lot_B → office_1: 2 buses,  60 pax [60→O1],  $20.00
lot_C → lot_B:    1 bus,    20 pax [20→O1],  $5.00
...

DEMAND MATRIX:
lot_A: 120→O1, 30→O2
lot_B:  40→O1, 60→O2
lot_C:  20→O1, 30→O2
```

**Note:** The commodity breakdown `[120→O1, 30→O2]` shows which passengers are on each bus.

## When to Use Each Model

### Use Single-Commodity Model When:
- Workers are flexible about destinations
- Quick approximation is needed
- Demand pattern is unknown
- Problem size is very large

### Use Multicommodity Model When:
- Workers have fixed office assignments
- Need to track individual O-D flows
- Feasibility is critical (must satisfy specific demands)
- Generating realistic operational plans

## Key Features

### Both Models
✓ Minimize total bus operating costs
✓ Respect bus capacity constraints
✓ Support transfers (lot-to-lot, office-to-office routes)
✓ Integer programming formulation (exact solutions)
✓ Fast solving with CBC/GLPK
✓ **NEW: Multiple bus types** with different capacities and costs

### Multicommodity Model Additional Features
✓ Track commodity-specific flows
✓ Buses can carry mixed commodities
✓ Origin-destination demand matrix
✓ Realistic worker destination requirements

### Multiple Bus Types (NEW)
✓ Define custom bus fleet (e.g., Stub Bus, Medium Bus, Long Bus)
✓ Different capacities and cost multipliers per bus type
✓ Optimizer automatically selects cost-effective mix
✓ Visualizations show bus type breakdown per route
✓ Works with both single-commodity and multicommodity models

## Computational Performance

- **Small problems** (2-3 lots, 2-3 offices): < 1 second
- **Medium problems** (5 lots, 5 offices): ~ 2-5 seconds
- **Large problems** (10+ lots/offices): May require specialized solvers

## Requirements

```
pulp>=2.7
numpy>=1.20
matplotlib>=3.5
networkx>=2.6
```

Install with:
```bash
pip install -r requirements.txt
```

## Extensions and Future Work

Possible extensions:
1. ~~Time-expanded network for scheduling~~ ✓ **IMPLEMENTED** (multi-temporal model)
2. Vehicle routing with individual bus tracking
3. Stochastic demand models
4. ~~Multiple bus sizes and types~~ ✓ **IMPLEMENTED**
5. Fixed costs per bus (in addition to variable route costs)
6. Bus availability constraints (limited fleet size per type)
7. Geographical routing with real distances
8. ~~Multi-period planning horizon~~ ✓ **IMPLEMENTED** (3 periods: morning/lunch/evening)
9. Additional time periods (4+ periods throughout the day)
10. Travel time and scheduling constraints

## References

- Ford, L. R., & Fulkerson, D. R. (1962). *Flows in Networks*. Princeton University Press.
- Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). *Network Flows*. Prentice Hall.

## License

MIT License - Academic/Educational use

## Author

Created for IOE 612: Stochastic Processes in Operations Research
