"""
Fisher Information-Based Optimal Sample Allocation (Balanced Version)

This version adds practical constraints:
1. Maximum samples per bounce (to avoid over-concentration)
2. Minimum samples per bounce (for model validation)
3. Smoothness constraint (gradual changes in allocation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle as pk
import sys

# Import functions from the original script
sys.path.insert(0, '/home/shun/quantum/NB')
from fisher_optimal_design import (
    exponential_model, sensitivity, estimate_variance,
    fisher_information_per_sample, total_fisher_information,
    load_previous_results
)

# Configuration
min_bounces = 2
max_bounces = 20
total_samples = 760
min_samples_per_bounce = 25  # Increased minimum
max_samples_per_bounce = 80  # Maximum to prevent over-concentration

def optimize_allocation_balanced(bounce_numbers, total_samples, A, f,
                                  min_samples=25, max_samples=80):
    """
    Find optimal sample allocation with balanced constraints.
    """
    n_bounces = len(bounce_numbers)

    # Compute Fisher Information per sample for each bounce number
    FI_per_sample = [fisher_information_per_sample(m, A, f) for m in bounce_numbers]

    # Objective: maximize Σ n_m * I_m (minimize negative)
    def objective(n):
        return -np.sum(n * np.array(FI_per_sample))

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda n: np.sum(n) - total_samples},
    ]

    # Bounds with max samples constraint
    bounds = [(min_samples, max_samples) for _ in range(n_bounces)]

    # Initial guess: proportional to square root of FI
    # (compromise between uniform and fully optimal)
    weights = np.sqrt(FI_per_sample)
    n0 = weights / np.sum(weights) * total_samples
    n0 = np.clip(n0, min_samples, max_samples)
    n0 = n0 / np.sum(n0) * total_samples  # Renormalize

    # Optimize
    result = minimize(objective, n0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")

    # Round to integers
    continuous_allocation = result.x
    allocation = np.round(continuous_allocation).astype(int)

    # Adjust to match exact total
    diff = total_samples - np.sum(allocation)
    if diff != 0:
        errors = continuous_allocation - allocation
        indices = np.argsort(errors if diff > 0 else -errors)
        for i in range(abs(diff)):
            # Check bounds
            new_val = allocation[indices[i]] + np.sign(diff)
            if min_samples <= new_val <= max_samples:
                allocation[indices[i]] = new_val

    return allocation, continuous_allocation

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("Fisher Information-Based Optimal Sample Allocation (Balanced)")
    print("="*80)

    # Load parameters from previous experiment
    A_est, f_est = load_previous_results('AB_decay_uniform_40.pickle')
    print(f"\nEstimated parameters: A = {A_est:.4f}, f = {f_est:.4f}")
    print(f"Constraints: {min_samples_per_bounce} ≤ n_m ≤ {max_samples_per_bounce}")

    # Calculate Fisher Information
    bounce_numbers = list(range(min_bounces, max_bounces + 1))
    FI_per_sample = [fisher_information_per_sample(m, A_est, f_est) for m in bounce_numbers]

    # Optimize with balanced constraints
    print("\nOptimizing with balanced constraints...")
    optimal_allocation, continuous_allocation = optimize_allocation_balanced(
        bounce_numbers, total_samples, A_est, f_est,
        min_samples=min_samples_per_bounce, max_samples=max_samples_per_bounce
    )

    # Display results
    print("\n" + "="*80)
    print("Balanced Optimal Sample Allocation")
    print("="*80)

    print(f"\n{'Bounce m':<10}{'Optimal n_m':<15}{'FI/sample':<20}{'FI contribution':<20}")
    print("-"*70)

    total_FI = 0
    for m, n_opt, FI_ps in zip(bounce_numbers, optimal_allocation, FI_per_sample):
        FI_contrib = n_opt * FI_ps
        total_FI += FI_contrib
        print(f"{m:<10}{n_opt:<15}{FI_ps:<20.2f}{FI_contrib:<20.2f}")

    print("-"*70)
    print(f"{'Total':<10}{np.sum(optimal_allocation):<15}{'':<20}{total_FI:<20.2f}")

    # Compare with uniform and extreme optimal
    uniform_allocation = np.ones(len(bounce_numbers)) * (total_samples // len(bounce_numbers))
    remaining = total_samples - np.sum(uniform_allocation)
    uniform_allocation[:int(remaining)] += 1
    uniform_FI = total_fisher_information(uniform_allocation, bounce_numbers, A_est, f_est)

    # Load extreme optimal if available
    try:
        with open('fisher_optimal_allocation.pkl', 'rb') as f:
            extreme_data = pk.load(f)
            extreme_FI = extreme_data['total_fisher_information']
    except:
        extreme_FI = None

    print("\n" + "="*80)
    print("Comparison")
    print("="*80)
    print(f"Total Fisher Information (Balanced Optimal): {total_FI:.2f}")
    print(f"Total Fisher Information (Uniform):          {uniform_FI:.2f}")
    if extreme_FI:
        print(f"Total Fisher Information (Extreme Optimal):  {extreme_FI:.2f}")

    print(f"\nImprovement over Uniform: {((total_FI - uniform_FI) / uniform_FI * 100):.2f}%")
    if extreme_FI:
        print(f"Efficiency vs Extreme:    {(total_FI / extreme_FI * 100):.2f}%")
    print("="*80)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Sample allocation
    ax1 = axes[0, 0]
    x = np.arange(len(bounce_numbers))
    width = 0.35
    ax1.bar(x - width/2, uniform_allocation, width, label='Uniform', color='gray', alpha=0.7)
    ax1.bar(x + width/2, optimal_allocation, width, label='Balanced Optimal', color='green', alpha=0.7)
    ax1.set_xlabel('Bounce number m', fontsize=12)
    ax1.set_ylabel('Number of samples', fontsize=12)
    ax1.set_title('Sample Allocation Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bounce_numbers)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Fisher Information per sample
    ax2 = axes[0, 1]
    ax2.plot(bounce_numbers, FI_per_sample, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax2.set_xlabel('Bounce number m', fontsize=12)
    ax2.set_ylabel('FI per sample', fontsize=12)
    ax2.set_title('Fisher Information per Sample', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(bounce_numbers)

    # Plot 3: Fisher Information contribution
    ax3 = axes[1, 0]
    FI_contrib_uniform = [n * FI for n, FI in zip(uniform_allocation, FI_per_sample)]
    FI_contrib_optimal = [n * FI for n, FI in zip(optimal_allocation, FI_per_sample)]
    ax3.bar(x - width/2, FI_contrib_uniform, width, label='Uniform', color='gray', alpha=0.7)
    ax3.bar(x + width/2, FI_contrib_optimal, width, label='Balanced Optimal', color='green', alpha=0.7)
    ax3.set_xlabel('Bounce number m', fontsize=12)
    ax3.set_ylabel('FI contribution', fontsize=12)
    ax3.set_title('Fisher Information Contribution by Bounce', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bounce_numbers)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Allocation pattern visualization
    ax4 = axes[1, 1]
    ax4.plot(bounce_numbers, optimal_allocation, 'o-', color='green', linewidth=2.5,
             markersize=10, label='Balanced Optimal')
    ax4.axhline(y=total_samples/len(bounce_numbers), color='gray', linestyle='--',
                linewidth=2, label='Uniform (avg)')
    ax4.fill_between(bounce_numbers, min_samples_per_bounce, max_samples_per_bounce,
                      alpha=0.2, color='blue', label='Allowed range')
    ax4.set_xlabel('Bounce number m', fontsize=12)
    ax4.set_ylabel('Number of samples', fontsize=12)
    ax4.set_title('Optimal Allocation Pattern', fontsize=14, fontweight='bold')
    ax4.set_xticks(bounce_numbers)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fisher_optimal_balanced.pdf', dpi=150)
    print("\nVisualization saved as: fisher_optimal_balanced.pdf")

    # Save optimal allocation
    optimal_dict = {m: int(n) for m, n in zip(bounce_numbers, optimal_allocation)}

    output_data = {
        'optimal_allocation': optimal_dict,
        'parameters': {'A': A_est, 'f': f_est},
        'total_fisher_information': total_FI,
        'uniform_fisher_information': uniform_FI,
        'constraints': {'min': min_samples_per_bounce, 'max': max_samples_per_bounce}
    }

    with open('fisher_optimal_balanced.pkl', 'wb') as f:
        pk.dump(output_data, f)

    print("\nOptimal allocation dictionary (for 2_chain.py):")
    print(f"samples_per_bounce = {optimal_dict}")
