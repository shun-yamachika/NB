"""
Extract and create a single plot: Cost vs Uncertainty: OLS vs WLS-MN
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

# Load results
with open('wls_mn_cost_analysis_results.pickle', 'rb') as f:
    results = pk.load(f)

# Extract data
cost_array = np.array(results['costs'])
ols_mean_array = np.array(results['ols']['mean_uncertainty']) * 100
ols_std_array = np.array(results['ols']['std_uncertainty']) * 100
wls_mn_mean_array = np.array(results['wls_mn']['mean_uncertainty']) * 100
wls_mn_std_array = np.array(results['wls_mn']['std_uncertainty']) * 100

# Create single plot
fig, ax = plt.subplots(figsize=(12, 8))

ax.errorbar(cost_array, ols_mean_array, yerr=ols_std_array,
            fmt='o-', color='blue', linewidth=2, markersize=8,
            capsize=5, capthick=2, label='OLS', alpha=0.7)
ax.errorbar(cost_array, wls_mn_mean_array, yerr=wls_mn_std_array,
            fmt='s-', color='red', linewidth=2, markersize=8,
            capsize=5, capthick=2, label='WLS-MN', alpha=0.7)

ax.set_xlabel('Total Cost (Σ m × n)', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative Uncertainty (%)', fontsize=14, fontweight='bold')
ax.set_title('Cost vs Uncertainty: OLS vs WLS-MN', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Highlight cost-efficient region
ax.axvspan(0, 3000, alpha=0.1, color='green', label='Cost-efficient region')

plt.tight_layout()
plt.savefig('cost_vs_uncertainty_single.pdf', bbox_inches='tight')
print("PDF created: cost_vs_uncertainty_single.pdf")
