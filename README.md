# Shuttle Service Optimization - IOE 612 Final Project

Three optimization models for shuttle bus routing: Stochastic Capacity, Multicommodity Flow, and Multi-Temporal.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all simulations
python run_all_examples.py

# Generate all visualizations
python generate_all_plots.py
```

## Reproduce All Results

### 1. Run All Three Models
```bash
python run_all_examples.py
```

This runs:
- Example 1: Stochastic Capacity (p=20%)
- Example 2: Multicommodity
- Example 3: Multi-Temporal (morning/lunch/evening)

Run individual examples:
```bash
python run_all_examples.py 1    # Stochastic only
python run_all_examples.py 2    # Multicommodity only
python run_all_examples.py 3    # Multi-temporal only
```

### 2. Generate All Visualizations
```bash
python generate_all_plots.py
```

This creates 11 PNG files in `plots/` directory:
- `stochastic_capacity_analysis.png` - Sensitivity analysis (p ∈ [0%, 30%])
- `stochastic_route_networks.png` - Route diagrams for different miss probabilities
- `stochastic_route_tables.png` - Detailed route tables
- `stochastic_fleet_utilization.png` - Fleet allocation
- `campus_layout_2x4.png` - Campus network layout
- `routes_on_campus_map.png` - Route comparison across all models
- `multitemporal_periods_comparison.png` - Morning/lunch/evening periods
- `routing_matrices_clean.png` - Routing matrices heatmaps
- `cost_matrices_heatmap.png` - Cost matrices
- `route_cost_matrix.png` - Route costs
- `depot_cost_matrix.png` - Depot costs

## File Structure

```
.
├── run_all_examples.py              # Run all optimization models
├── generate_all_plots.py            # Generate all visualizations
├── shuttle_optimization.py          # Core optimization classes
│
├── visualize_stochastic.py          # Stochastic sensitivity analysis
├── visualize_stochastic_routes.py   # Stochastic route diagrams
├── visualize_routes_on_map.py       # Model comparison on campus map
├── visualize_multitemporal_periods.py # Multi-temporal periods
├── visualize_2x4_campus.py          # Campus layout
├── visualize_routing_matrices_clean.py # Routing matrices
├── create_cost_heatmaps.py          # Cost heatmaps
│
├── claude.md                        # Complete project documentation
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
└── plots/                           # Generated visualizations (11 PNG files)
```

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

## Expected Results

**Stochastic Capacity (p=20%):** $633, 11 vehicles (5 bikes, 2 medium, 4 long)
**Multicommodity:** $631, 16 vehicles (10 bikes, 6 long)
**Multi-Temporal:** $994, 5 vehicles (1 stub, 4 long) across 3 periods

## Documentation

See `claude.md` for complete project documentation including:
- Mathematical formulations for all three models
- Detailed model descriptions and usage examples
- Network configuration and parameters
- Cost structures and depot modeling

## License

MIT License - Academic/Educational Use
