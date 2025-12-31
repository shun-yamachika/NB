"""
Analyze the relationship between sequence length (m) and standard deviation.
This helps determine if noise depends on m.
"""

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

# Load 2-node data
print("Loading 2-node simulation data...")
with open('AB_decay_50samples.pickle', 'rb') as f:
    data = pk.load(f)

endpoints = data["endpoints"]
min_m, max_m = endpoints
m_values = list(range(min_m, max_m + 1))

# Extract raw data
fid_raw_data = data["decay data"][1]
fid_means = data["decay data"][0]

# Compute statistics for each m (using all 50 samples)
statistics = {
    'm': [],
    'mean': [],
    'std': [],
    'cv': [],  # Coefficient of variation = std/mean
    'n_samples': []
}

print("\nStatistics per sequence length (using all 50 samples):")
print("="*70)
print(f"{'m':<4} {'Mean':<10} {'Std':<10} {'CV':<10} {'Std/√50':<10}")
print("="*70)

for m in m_values:
    raw_data = fid_raw_data[m]
    mean_val = np.mean(raw_data)
    std_val = np.std(raw_data, ddof=1)
    cv_val = std_val / mean_val if mean_val > 0 else 0
    sem_val = std_val / np.sqrt(50)

    statistics['m'].append(m)
    statistics['mean'].append(mean_val)
    statistics['std'].append(std_val)
    statistics['cv'].append(cv_val)
    statistics['n_samples'].append(50)

    print(f"{m:<4} {mean_val:<10.6f} {std_val:<10.6f} {cv_val:<10.6f} {sem_val:<10.6f}")

print("="*70)

# Statistical analysis
print("\nCorrelation analysis:")
print(f"Correlation(m, std):  {np.corrcoef(statistics['m'], statistics['std'])[0,1]:.4f}")
print(f"Correlation(m, CV):   {np.corrcoef(statistics['m'], statistics['cv'])[0,1]:.4f}")
print(f"Correlation(mean, std): {np.corrcoef(statistics['mean'], statistics['std'])[0,1]:.4f}")

# Simple linear model: std = a + b*m
from scipy.stats import linregress
slope_m, intercept_m, r_m, p_m, se_m = linregress(statistics['m'], statistics['std'])
print(f"\nLinear model: std = {intercept_m:.6f} + {slope_m:.6f} * m")
print(f"  R² = {r_m**2:.4f}, p-value = {p_m:.4e}")

# Model: std = c * mean
slope_mean, intercept_mean_forced, r_mean, p_mean, se_mean = linregress(statistics['mean'], statistics['std'])
print(f"\nLinear model: std = {intercept_mean_forced:.6f} + {slope_mean:.6f} * mean")
print(f"  R² = {r_mean**2:.4f}, p-value = {p_mean:.4e}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: std vs m
ax1 = axes[0, 0]
ax1.scatter(statistics['m'], statistics['std'], alpha=0.6)
ax1.plot(statistics['m'], np.array(statistics['m']) * slope_m + intercept_m,
         'r--', label=f'Linear fit (R²={r_m**2:.3f})')
ax1.set_xlabel('Sequence length (m)', fontsize=12)
ax1.set_ylabel('Standard deviation', fontsize=12)
ax1.set_title('Standard Deviation vs Sequence Length', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: CV vs m
ax2 = axes[0, 1]
ax2.scatter(statistics['m'], statistics['cv'], alpha=0.6, color='green')
ax2.set_xlabel('Sequence length (m)', fontsize=12)
ax2.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
ax2.set_title('Relative Noise vs Sequence Length', fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot 3: std vs mean
ax3 = axes[1, 0]
ax3.scatter(statistics['mean'], statistics['std'], alpha=0.6, color='orange')
ax3.plot(statistics['mean'], np.array(statistics['mean']) * slope_mean + intercept_mean_forced,
         'r--', label=f'Linear fit (R²={r_mean**2:.3f})')
ax3.set_xlabel('Mean fidelity', fontsize=12)
ax3.set_ylabel('Standard deviation', fontsize=12)
ax3.set_title('Standard Deviation vs Mean Fidelity', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: mean vs m (exponential decay)
ax4 = axes[1, 1]
ax4.scatter(statistics['m'], statistics['mean'], alpha=0.6, color='purple')
ax4.set_xlabel('Sequence length (m)', fontsize=12)
ax4.set_ylabel('Mean fidelity', fontsize=12)
ax4.set_title('Fidelity Decay', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('std_vs_m_analysis.png', dpi=150)
print("\nPlot saved to: std_vs_m_analysis.png")

# Save statistics
with open('std_vs_m_statistics.pickle', 'wb') as f:
    pk.dump(statistics, f)
print("Statistics saved to: std_vs_m_statistics.pickle")
