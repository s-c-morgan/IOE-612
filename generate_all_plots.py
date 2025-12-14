"""
Generate all visualization plots for the shuttle optimization models
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid display issues

print("\n" + "="*80)
print("GENERATING ALL VISUALIZATIONS")
print("Shuttle Optimization Models")
print("="*80 + "\n")

# Create plots directory
os.makedirs('plots', exist_ok=True)
print("✓ Created plots/ directory\n")

# Import and run each visualization
print("Generating visualizations...\n")

# 1. Stochastic Capacity Analysis (Example 1)
print("[1/5] Stochastic capacity sensitivity analysis...")
try:
    exec(open('visualize_stochastic.py').read(), {'__name__': '__main__'})
    print()
except Exception as e:
    print(f"✗ Error generating stochastic analysis: {e}\n")

# 2. Stochastic Route Assignments
print("[2/5] Stochastic capacity route assignments...")
try:
    exec(open('visualize_stochastic_routes.py').read(), {'__name__': '__main__'})
    print()
except Exception as e:
    print(f"✗ Error generating stochastic route visualizations: {e}\n")

# 3. Campus Layout
print("[3/5] Campus layout diagram...")
try:
    exec(open('visualize_2x4_campus.py').read(), {'__name__': '__main__'})
    print()
except Exception as e:
    print(f"✗ Error generating campus layout: {e}\n")

# 4. Routes on Campus Map
print("[4/7] Routes on campus map...")
try:
    exec(open('visualize_routes_on_map.py').read(), {'__name__': '__main__'})
    print()
except Exception as e:
    print(f"✗ Error generating routes on campus map: {e}\n")

# 5. Multi-Temporal Periods Comparison
print("[5/7] Multi-temporal periods comparison...")
try:
    exec(open('visualize_multitemporal_periods.py').read(), {'__name__': '__main__'})
    print()
except Exception as e:
    print(f"✗ Error generating multi-temporal periods: {e}\n")

# 6. Routing Matrices
print("[6/7] Routing matrices comparison...")
try:
    exec(open('visualize_routing_matrices_clean.py').read(), {'__name__': '__main__'})
    print()
except Exception as e:
    print(f"✗ Error generating routing matrices: {e}\n")

# 7. Cost Heatmaps
print("[7/7] Cost heatmaps...")
try:
    exec(open('create_cost_heatmaps.py').read(), {'__name__': '__main__'})
    print()
except Exception as e:
    print(f"✗ Error generating cost heatmaps: {e}\n")

# Summary
print("="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files in plots/ directory:")
print("\n  Example 1 - Stochastic Capacity:")
print("    1. stochastic_capacity_analysis.png - Sensitivity analysis (p ∈ [0%, 30%])")
print("    2. stochastic_route_networks.png    - Network diagrams for p=0%, 10%, 20%, 30%")
print("    3. stochastic_route_tables.png      - Detailed route assignment tables")
print("    4. stochastic_fleet_utilization.png - Fleet allocation and utilization")
print("\n  General:")
print("    5. campus_layout_2x4.png                  - Campus network layout")
print("    6. routes_on_campus_map.png               - Bus routes on spatial layout (all models)")
print("    7. multitemporal_periods_comparison.png   - Multi-temporal periods (morning/lunch/evening)")
print("    8. routing_matrices_clean.png             - Routing matrices comparison (all models)")
print("    9. cost_matrices_heatmap.png              - Combined cost matrices")
print("   10. route_cost_matrix.png                  - Detailed route costs")
print("   11. depot_cost_matrix.png                  - Detailed depot costs")
print("\nTotal: 11 PNG files")
print("="*80 + "\n")
