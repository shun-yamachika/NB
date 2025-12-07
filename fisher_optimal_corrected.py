"""
Fisher-Optimal Design Using ACTUAL Variance from Real Data

This version uses the actual variance observed in the Uniform 40 experiment
to calculate the optimal sample allocation.
"""

import numpy as np
import pickle as pk
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load actual variance data from Uniform 40
def load_actual_variances(filename='AB_decay_uniform_40.pickle'):
    with open(filename, 'rb') as f:
        data = pk.load(f)
        endpoints = data["endpoints"]
        fid_data = data["decay data"][1]

        variances = {}
        for m in range(endpoints[0], endpoints[1] + 1):
            samples = fid_data[m]
            variance = np.var(samples, ddof=1) if len(samples) > 1 else 0.001
            variances[m] = max(variance, 0.0001)  # Avoid zero variance

        return variances

# Load and fit the model
def load_and_fit(filename='AB_decay_uniform_40.pickle'):
    with open(filename, 'rb') as f:
        data = pk.load(f)
        endpoints = data["endpoints"]
        fid_means = data["decay data"][0]

        m_values = np.array(range(endpoints[0], endpoints[1] + 1))
        fid_values = np.array([fid_means[m] for m in m_values])

        # Fit: log(fid) = log(A) + m*log(f)
        valid = fid_values > 0
        log_fid = np.log(fid_values[valid])
        m_valid = m_values[valid]

        coeffs = np.polyfit(m_valid, log_fid, 1)
        f_est = np.exp(coeffs[0])
        A_est = np.exp(coeffs[1])

        return A_est, f_est

def sensitivity(m, A, f):
    """Sensitivity df/dm"""
    return A * m * f**(m-1)

def optimize_with_actual_variance(bounce_numbers, total_samples, A, f, actual_variances,
                                   min_samples=25, max_samples=80):
    """
    Optimize using ACTUAL measured variances
    """
    n_bounces = len(bounce_numbers)

    # Compute Fisher Information per sample using ACTUAL variances
    FI_per_sample = []
    for m in bounce_numbers:
        sens = sensitivity(m, A, f)
        var = actual_variances[m]
        FI_ps = (sens**2) / var
        FI_per_sample.append(FI_ps)

    # Objective: maximize total Fisher Information
    def objective(n):
        return -np.sum(n * np.array(FI_per_sample))

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda n: np.sum(n) - total_samples},
    ]

    # Bounds
    bounds = [(min_samples, max_samples) for _ in range(n_bounces)]

    # Initial guess: proportional to FI
    weights = np.array(FI_per_sample)
    n0 = weights / np.sum(weights) * total_samples
    n0 = np.clip(n0, min_samples, max_samples)

    # Optimize
    result = minimize(objective, n0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"Warning: {result.message}")

    # Round to integers
    continuous = result.x
    allocation = np.round(continuous).astype(int)

    # Adjust to exact total
    diff = total_samples - np.sum(allocation)
    if diff != 0:
        errors = continuous - allocation
        indices = np.argsort(errors if diff > 0 else -errors)
        for i in range(abs(diff)):
            new_val = allocation[indices[i]] + np.sign(diff)
            if min_samples <= new_val <= max_samples:
                allocation[indices[i]] = new_val

    return allocation, FI_per_sample

# Main
print("="*80)
print("Fisher-Optimal Design Using ACTUAL Variance")
print("="*80)

# Load parameters
A_est, f_est = load_and_fit()
actual_variances = load_actual_variances()

print(f"\nEstimated parameters: A = {A_est:.4f}, f = {f_est:.4f}")

# Optimize
bounce_numbers = list(range(2, 21))
total_samples = 760
min_samples = 25
max_samples = 80

optimal_allocation, FI_per_sample = optimize_with_actual_variance(
    bounce_numbers, total_samples, A_est, f_est, actual_variances,
    min_samples, max_samples
)

# Calculate total Fisher Information
total_FI_optimal = np.sum(optimal_allocation * np.array(FI_per_sample))
uniform_allocation = np.ones(len(bounce_numbers)) * 40
total_FI_uniform = np.sum(uniform_allocation * np.array(FI_per_sample))

print("\n" + "="*80)
print("Corrected Optimal Allocation (Using Actual Variance)")
print("="*80)

print(f"\n{'Bounce':<8}{'FI/sample':<15}{'Actual Var':<15}{'Optimal n':<12}{'Uniform n':<12}")
print("-"*70)

for m, FI_ps, n_opt in zip(bounce_numbers, FI_per_sample, optimal_allocation):
    var = actual_variances[m]
    print(f"{m:<8}{FI_ps:<15.2f}{var:<15.6f}{n_opt:<12}{40:<12}")

print("-"*70)
print(f"{'Total':<8}{'':<15}{'':<15}{np.sum(optimal_allocation):<12}{np.sum(uniform_allocation):<12}")

print("\n" + "="*80)
print(f"Total Fisher Information (Corrected Optimal): {total_FI_optimal:.2f}")
print(f"Total Fisher Information (Uniform 40):         {total_FI_uniform:.2f}")
improvement = ((total_FI_optimal - total_FI_uniform) / total_FI_uniform) * 100
print(f"Improvement: {improvement:.2f}%")
print("="*80)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Fisher Information per sample
ax1 = axes[0, 0]
ax1.bar(bounce_numbers, FI_per_sample, color='steelblue', alpha=0.7)
ax1.set_xlabel('Bounce number m', fontsize=12)
ax1.set_ylabel('Fisher Information per sample', fontsize=12)
ax1.set_title('FI per Sample (Using Actual Variance)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(bounce_numbers)

# Highlight top 5
top5_indices = np.argsort(FI_per_sample)[-5:]
for idx in top5_indices:
    ax1.bar(bounce_numbers[idx], FI_per_sample[idx], color='red', alpha=0.7)

# Plot 2: Optimal vs Uniform allocation
ax2 = axes[0, 1]
x = np.arange(len(bounce_numbers))
width = 0.35
ax2.bar(x - width/2, uniform_allocation, width, label='Uniform', color='gray', alpha=0.7)
ax2.bar(x + width/2, optimal_allocation, width, label='Corrected Optimal', color='darkgreen', alpha=0.7)
ax2.set_xlabel('Bounce number m', fontsize=12)
ax2.set_ylabel('Number of samples', fontsize=12)
ax2.set_title('Sample Allocation Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(bounce_numbers)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Actual variance distribution
ax3 = axes[1, 0]
variances_list = [actual_variances[m] for m in bounce_numbers]
ax3.plot(bounce_numbers, variances_list, 'o-', linewidth=2, markersize=8, color='red')
ax3.set_xlabel('Bounce number m', fontsize=12)
ax3.set_ylabel('Actual Variance', fontsize=12)
ax3.set_title('Variance Across Bounce Numbers', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(bounce_numbers)

# Plot 4: Fisher Information contribution
ax4 = axes[1, 1]
FI_contrib_uniform = uniform_allocation * np.array(FI_per_sample)
FI_contrib_optimal = optimal_allocation * np.array(FI_per_sample)
ax4.bar(x - width/2, FI_contrib_uniform, width, label='Uniform', color='gray', alpha=0.7)
ax4.bar(x + width/2, FI_contrib_optimal, width, label='Corrected Optimal', color='darkgreen', alpha=0.7)
ax4.set_xlabel('Bounce number m', fontsize=12)
ax4.set_ylabel('FI Contribution', fontsize=12)
ax4.set_title('Fisher Information Contribution', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(bounce_numbers)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fisher_optimal_corrected.pdf', dpi=150)
print("\nVisualization saved as: fisher_optimal_corrected.pdf")

# Save optimal allocation
optimal_dict = {m: int(n) for m, n in zip(bounce_numbers, optimal_allocation)}
print("\nCorrected optimal allocation dictionary:")
print(f"samples_per_bounce = {optimal_dict}")

# Save to file
with open('fisher_optimal_corrected.pkl', 'wb') as f:
    pk.dump({
        'optimal_allocation': optimal_dict,
        'FI_per_sample': {m: fi for m, fi in zip(bounce_numbers, FI_per_sample)},
        'actual_variances': actual_variances,
        'total_FI': total_FI_optimal,
        'improvement': improvement
    }, f)
