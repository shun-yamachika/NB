"""
Fisher Information-Based Optimal Sample Allocation for Network RB

This script calculates the optimal allocation of samples across different
bounce numbers to maximize the Fisher Information for estimating the
network fidelity parameter f.

Theory:
For the exponential decay model: b_m = A * f^m
where m is the number of bounces and f is the network fidelity.

The Fisher Information for parameter f at bounce number m is:
I_m(f) = n_m * (∂b_m/∂f)^2 / σ_m^2

where:
- n_m is the number of samples at bounce m
- ∂b_m/∂f = A * m * f^(m-1) is the sensitivity
- σ_m^2 is the variance of measurements at bounce m

For optimal design, we maximize total Fisher Information:
I_total = Σ_m I_m(f)

subject to: Σ_m n_m = N (total sample budget)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
import pickle as pk

# Configuration
min_bounces = 2
max_bounces = 20
total_samples = 760  # Total sample budget
min_samples_per_bounce = 20  # Minimum samples per bounce (for stability)

def exponential_model(m, A, f):
    """Exponential decay model: b_m = A * f^m"""
    return A * f**m

def sensitivity(m, A, f):
    """
    Sensitivity of the model to parameter f: ∂b_m/∂f
    """
    return A * m * f**(m-1)

def estimate_variance(m, A, f, noise_factor=0.1):
    """
    Estimate the measurement variance at bounce number m.

    In practice, variance depends on:
    1. Quantum measurement noise
    2. Statistical fluctuations
    3. Systematic errors

    Simple model: σ_m^2 ≈ b_m * (1 - b_m) * noise_factor
    This accounts for binomial-like statistics.
    """
    b_m = exponential_model(m, A, f)
    # Binomial-like variance + minimum floor
    variance = max(b_m * (1 - b_m) * noise_factor + 0.001, 0.001)
    return variance

def fisher_information_per_sample(m, A, f):
    """
    Fisher Information per sample at bounce number m for parameter f.

    I_m^(1)(f) = (∂b_m/∂f)^2 / σ_m^2
    """
    sens = sensitivity(m, A, f)
    var = estimate_variance(m, A, f)
    return (sens**2) / var

def total_fisher_information(sample_allocation, bounce_numbers, A, f):
    """
    Total Fisher Information for a given sample allocation.

    I_total = Σ_m n_m * I_m^(1)(f)
    """
    total_FI = 0
    for m, n_m in zip(bounce_numbers, sample_allocation):
        FI_per_sample = fisher_information_per_sample(m, A, f)
        total_FI += n_m * FI_per_sample
    return total_FI

def optimize_allocation_continuous(bounce_numbers, total_samples, A, f, min_samples=20):
    """
    Find optimal sample allocation using continuous optimization,
    then round to integers.

    We want to maximize: Σ_m n_m * I_m^(1)(f)
    subject to:
    - Σ_m n_m = total_samples
    - n_m >= min_samples for all m
    """
    n_bounces = len(bounce_numbers)

    # Compute Fisher Information per sample for each bounce number
    FI_per_sample = [fisher_information_per_sample(m, A, f) for m in bounce_numbers]

    # Objective: maximize Σ n_m * I_m (equivalent to minimizing negative)
    def objective(n):
        return -np.sum(n * np.array(FI_per_sample))

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda n: np.sum(n) - total_samples},  # Total samples
    ]

    # Bounds: minimum samples per bounce
    bounds = [(min_samples, total_samples) for _ in range(n_bounces)]

    # Initial guess: uniform distribution
    n0 = np.ones(n_bounces) * (total_samples / n_bounces)

    # Optimize
    result = minimize(objective, n0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")

    # Round to integers while preserving total
    continuous_allocation = result.x
    allocation = np.round(continuous_allocation).astype(int)

    # Adjust to match exact total
    diff = total_samples - np.sum(allocation)
    if diff != 0:
        # Add/subtract samples to/from positions with largest rounding error
        errors = continuous_allocation - allocation
        indices = np.argsort(errors if diff > 0 else -errors)
        for i in range(abs(diff)):
            allocation[indices[i]] += np.sign(diff)

    return allocation, continuous_allocation

def load_previous_results(filename='AB_decay_uniform_40.pickle'):
    """
    Load results from a previous experiment to estimate A and f.
    """
    try:
        with open(filename, 'rb') as file:
            data = pk.load(file)
            fid_means = data["decay data"][0]
            endpoints = data["endpoints"]

            # Estimate A and f from the data
            # Using simple linear regression on log-transformed data
            # log(b_m) = log(A) + m*log(f)
            m_values = np.array(range(endpoints[0], endpoints[1]+1))
            b_values = np.array([fid_means[i] for i in m_values])

            # Remove any zeros or negative values
            valid = b_values > 0
            m_values = m_values[valid]
            b_values = b_values[valid]

            log_b = np.log(b_values)

            # Linear fit
            coeffs = np.polyfit(m_values, log_b, 1)
            log_f = coeffs[0]
            log_A = coeffs[1]

            f_est = np.exp(log_f)
            A_est = np.exp(log_A)

            return A_est, f_est
    except Exception as e:
        print(f"Could not load previous results: {e}")
        print("Using default values")
        return 0.5, 0.90  # Default values

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("Fisher Information-Based Optimal Sample Allocation")
    print("="*80)

    # Step 1: Estimate parameters from previous experiment
    print("\nStep 1: Estimating parameters from previous experiments...")
    A_est, f_est = load_previous_results('AB_decay_uniform_40.pickle')
    print(f"Estimated parameters: A = {A_est:.4f}, f = {f_est:.4f}")

    # Step 2: Calculate Fisher Information for each bounce number
    print("\nStep 2: Computing Fisher Information per sample at each bounce number...")
    bounce_numbers = list(range(min_bounces, max_bounces + 1))
    FI_per_sample = [fisher_information_per_sample(m, A_est, f_est) for m in bounce_numbers]

    print(f"\n{'Bounce m':<10}{'FI per sample':<20}{'Sensitivity':<20}{'Variance':<20}")
    print("-"*70)
    for m, FI in zip(bounce_numbers, FI_per_sample):
        sens = sensitivity(m, A_est, f_est)
        var = estimate_variance(m, A_est, f_est)
        print(f"{m:<10}{FI:<20.6f}{sens:<20.6f}{var:<20.6f}")

    # Step 3: Optimize sample allocation
    print("\nStep 3: Optimizing sample allocation...")
    optimal_allocation, continuous_allocation = optimize_allocation_continuous(
        bounce_numbers, total_samples, A_est, f_est, min_samples=min_samples_per_bounce
    )

    # Step 4: Display results
    print("\n" + "="*80)
    print("Optimal Sample Allocation")
    print("="*80)

    print(f"\n{'Bounce m':<10}{'Optimal n_m':<15}{'Continuous':<15}{'FI contribution':<20}")
    print("-"*70)

    total_FI = 0
    for m, n_opt, n_cont, FI_ps in zip(bounce_numbers, optimal_allocation, continuous_allocation, FI_per_sample):
        FI_contrib = n_opt * FI_ps
        total_FI += FI_contrib
        print(f"{m:<10}{n_opt:<15}{n_cont:<15.2f}{FI_contrib:<20.6f}")

    print("-"*70)
    print(f"{'Total':<10}{np.sum(optimal_allocation):<15}{np.sum(continuous_allocation):<15.2f}{total_FI:<20.6f}")

    # Compare with uniform allocation
    uniform_allocation = np.ones(len(bounce_numbers)) * (total_samples // len(bounce_numbers))
    remaining = total_samples - np.sum(uniform_allocation)
    uniform_allocation[:int(remaining)] += 1
    uniform_FI = total_fisher_information(uniform_allocation, bounce_numbers, A_est, f_est)

    print("\n" + "="*80)
    print(f"Total Fisher Information (Optimal): {total_FI:.6f}")
    print(f"Total Fisher Information (Uniform): {uniform_FI:.6f}")
    print(f"Improvement: {((total_FI - uniform_FI) / uniform_FI * 100):.2f}%")
    print("="*80)

    # Step 5: Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

    # Plot 1: Fisher Information per sample
    ax1.bar(bounce_numbers, FI_per_sample, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Bounce number m', fontsize=14)
    ax1.set_ylabel('Fisher Information per sample', fontsize=14)
    ax1.set_title('Fisher Information per Sample at Each Bounce Number', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(bounce_numbers)

    # Plot 2: Sample allocation comparison
    x = np.arange(len(bounce_numbers))
    width = 0.35

    ax2.bar(x - width/2, uniform_allocation, width, label='Uniform', color='gray', alpha=0.7)
    ax2.bar(x + width/2, optimal_allocation, width, label='Fisher Optimal', color='green', alpha=0.7)
    ax2.set_xlabel('Bounce number m', fontsize=14)
    ax2.set_ylabel('Number of samples', fontsize=14)
    ax2.set_title('Sample Allocation: Uniform vs Fisher Optimal', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bounce_numbers)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Fisher Information contribution
    FI_contributions = [n * FI for n, FI in zip(optimal_allocation, FI_per_sample)]
    ax3.bar(bounce_numbers, FI_contributions, color='darkgreen', alpha=0.7)
    ax3.set_xlabel('Bounce number m', fontsize=14)
    ax3.set_ylabel('Fisher Information contribution', fontsize=14)
    ax3.set_title('Fisher Information Contribution by Bounce Number (Optimal Allocation)',
                  fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(bounce_numbers)

    plt.tight_layout()
    plt.savefig('fisher_optimal_allocation.pdf', dpi=150)
    print("\nVisualization saved as: fisher_optimal_allocation.pdf")

    # Step 6: Save optimal allocation for simulation
    optimal_dict = {m: int(n) for m, n in zip(bounce_numbers, optimal_allocation)}

    output_data = {
        'optimal_allocation': optimal_dict,
        'parameters': {'A': A_est, 'f': f_est},
        'total_fisher_information': total_FI,
        'uniform_fisher_information': uniform_FI
    }

    with open('fisher_optimal_allocation.pkl', 'wb') as f:
        pk.dump(output_data, f)

    print("\nOptimal allocation saved to: fisher_optimal_allocation.pkl")
    print("\nOptimal allocation dictionary (for copy-paste into 2_chain.py):")
    print(f"samples_per_bounce = {optimal_dict}")
