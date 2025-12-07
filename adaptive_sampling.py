"""
Adaptive/Online Fisher-Optimal Sampling for Network RB

This script simulates an adaptive experimental design where:
1. Start with initial exploration (uniform sampling)
2. Estimate variance and parameters from collected data
3. Compute optimal allocation for next batch
4. Repeat until total budget is exhausted

This approach learns the variance structure online instead of
assuming it's known beforehand.
"""

import numpy as np
import pickle as pk
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from scipy.stats import t

def exp(m, A, f):
    """Exponential decay model"""
    return A * f**m

def sensitivity(m, A, f):
    """Sensitivity to parameter f"""
    return A * m * f**(m-1)

class AdaptiveSampler:
    """
    Adaptive sampler that learns variance online and optimizes allocation
    """
    def __init__(self, bounce_numbers, total_budget, batch_size,
                 min_samples=5, max_samples_per_batch=20):
        self.bounce_numbers = bounce_numbers
        self.total_budget = total_budget
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.max_samples_per_batch = max_samples_per_batch

        # Accumulated data
        self.samples_collected = {m: 0 for m in bounce_numbers}
        self.all_measurements = {m: [] for m in bounce_numbers}

        # Current estimates
        self.mean_estimates = {m: None for m in bounce_numbers}
        self.variance_estimates = {m: 0.001 for m in bounce_numbers}  # Initial guess
        self.A_est = 0.5
        self.f_est = 0.9

        # History
        self.allocation_history = []
        self.parameter_history = []
        self.variance_history = []

    def initial_exploration(self, n_per_bounce=5):
        """
        Phase 1: Uniform exploration to get initial variance estimates
        """
        allocation = {m: n_per_bounce for m in self.bounce_numbers}
        return allocation

    def update_statistics(self, new_measurements):
        """
        Update mean and variance estimates with new measurements

        new_measurements: {bounce_number: [list of fidelity values]}
        """
        for m, measurements in new_measurements.items():
            if measurements:
                self.all_measurements[m].extend(measurements)
                self.samples_collected[m] += len(measurements)

                # Update mean estimate
                self.mean_estimates[m] = np.mean(self.all_measurements[m])

                # Update variance estimate (with minimum floor)
                if len(self.all_measurements[m]) > 1:
                    var = np.var(self.all_measurements[m], ddof=1)
                    self.variance_estimates[m] = max(var, 0.0001)

    def update_model_parameters(self):
        """
        Fit exponential decay model to current data
        """
        # Get current means for bounces with data
        valid_bounces = [m for m in self.bounce_numbers
                        if self.mean_estimates[m] is not None and self.mean_estimates[m] > 0]

        if len(valid_bounces) < 3:
            return  # Not enough data yet

        m_values = np.array(valid_bounces)
        fid_values = np.array([self.mean_estimates[m] for m in valid_bounces])

        try:
            # Fit: log(fid) = log(A) + m*log(f)
            log_fid = np.log(fid_values)
            coeffs = np.polyfit(m_values, log_fid, 1)
            self.f_est = np.exp(coeffs[0])
            self.A_est = np.exp(coeffs[1])

            # Clip to reasonable ranges
            self.f_est = np.clip(self.f_est, 0.5, 0.99)
            self.A_est = np.clip(self.A_est, 0.1, 1.0)

        except:
            pass  # Keep previous estimates if fitting fails

    def compute_fisher_information_per_sample(self):
        """
        Compute Fisher Information per sample for each bounce
        using current parameter and variance estimates
        """
        FI_per_sample = {}
        for m in self.bounce_numbers:
            sens = sensitivity(m, self.A_est, self.f_est)
            var = self.variance_estimates[m]
            FI_per_sample[m] = (sens**2) / var if var > 0 else 0

        return FI_per_sample

    def optimize_next_batch(self, remaining_budget):
        """
        Optimize allocation for next batch using current estimates
        """
        FI_per_sample = self.compute_fisher_information_per_sample()

        # Determine batch size
        actual_batch_size = min(self.batch_size, remaining_budget)

        # Optimization: maximize sum of n_m * FI_m
        # subject to: sum(n_m) = batch_size
        #             0 <= n_m <= max_samples_per_batch

        n_bounces = len(self.bounce_numbers)
        FI_array = np.array([FI_per_sample[m] for m in self.bounce_numbers])

        # Objective: minimize negative FI
        def objective(n):
            return -np.sum(n * FI_array)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda n: np.sum(n) - actual_batch_size}
        ]

        # Bounds
        bounds = [(0, self.max_samples_per_batch) for _ in range(n_bounces)]

        # Initial guess: proportional to FI
        if np.sum(FI_array) > 0:
            weights = FI_array / np.sum(FI_array)
            n0 = weights * actual_batch_size
        else:
            n0 = np.ones(n_bounces) * (actual_batch_size / n_bounces)

        # Optimize
        result = minimize(objective, n0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 100})

        if result.success:
            continuous = result.x
        else:
            # Fallback to proportional allocation
            continuous = n0

        # Round to integers
        allocation_array = np.round(continuous).astype(int)

        # Adjust to exact total
        diff = actual_batch_size - np.sum(allocation_array)
        if diff != 0:
            errors = continuous - allocation_array
            indices = np.argsort(errors if diff > 0 else -errors)
            for i in range(abs(diff)):
                if 0 <= allocation_array[indices[i]] + np.sign(diff) <= self.max_samples_per_batch:
                    allocation_array[indices[i]] += int(np.sign(diff))

        # Convert to dict
        allocation = {m: int(n) for m, n in zip(self.bounce_numbers, allocation_array)}

        return allocation

    def get_total_samples(self):
        """Total samples collected so far"""
        return sum(self.samples_collected.values())

    def save_state(self):
        """Save current state to history"""
        self.allocation_history.append(self.samples_collected.copy())
        self.parameter_history.append((self.A_est, self.f_est))
        self.variance_history.append(self.variance_estimates.copy())


def simulate_adaptive_experiment(real_data_file='AB_decay_uniform_40.pickle',
                                 total_budget=760,
                                 initial_samples=10,
                                 batch_size=50):
    """
    Simulate an adaptive experiment using real data

    We'll pretend we're doing the experiment adaptively by:
    1. Drawing samples from the real data pool
    2. Using those to estimate variance
    3. Deciding next allocation
    4. Repeat
    """

    # Load real data (this is our "ground truth" distribution)
    with open(real_data_file, 'rb') as f:
        real_data = pk.load(f)
        real_fid_data = real_data['decay data'][1]

    bounce_numbers = list(range(2, 21))
    sampler = AdaptiveSampler(bounce_numbers, total_budget, batch_size,
                             min_samples=5, max_samples_per_batch=20)

    print("="*80)
    print("Adaptive/Online Fisher-Optimal Sampling Simulation")
    print("="*80)
    print(f"\nTotal budget: {total_budget} samples")
    print(f"Batch size: {batch_size} samples")
    print(f"Initial exploration: {initial_samples} samples per bounce")
    print()

    # Phase 1: Initial uniform exploration
    print("Phase 1: Initial exploration...")
    initial_allocation = sampler.initial_exploration(n_per_bounce=initial_samples)

    # "Perform" initial measurements (sample from real data)
    initial_measurements = {}
    for m, n_samples in initial_allocation.items():
        # Randomly sample from real data
        available_samples = real_fid_data[m]
        if n_samples <= len(available_samples):
            sampled = np.random.choice(available_samples, size=n_samples, replace=False)
        else:
            sampled = available_samples
        initial_measurements[m] = list(sampled)

    sampler.update_statistics(initial_measurements)
    sampler.update_model_parameters()
    sampler.save_state()

    print(f"Initial samples collected: {sampler.get_total_samples()}")
    print(f"Initial parameter estimates: A = {sampler.A_est:.4f}, f = {sampler.f_est:.4f}")

    # Phase 2: Adaptive batches
    print("\nPhase 2: Adaptive optimization...")
    batch_num = 1

    while sampler.get_total_samples() < total_budget and batch_num <= 50:
        remaining = total_budget - sampler.get_total_samples()

        # Compute optimal allocation for next batch
        next_allocation = sampler.optimize_next_batch(remaining)

        # Filter out zero allocations
        next_allocation = {m: n for m, n in next_allocation.items() if n > 0}

        batch_total = sum(next_allocation.values())
        print(f"\nBatch {batch_num}: {batch_total} samples")
        print(f"  Allocation: {next_allocation}")

        # "Perform" measurements
        new_measurements = {}
        for m, n_samples in next_allocation.items():
            # Sample from real data (avoid reusing samples if possible)
            available_samples = real_fid_data[m]
            already_used = len(sampler.all_measurements[m])

            # Try to use new samples if available
            if already_used + n_samples <= len(available_samples):
                # Use sequential samples (simulating new experiments)
                sampled = available_samples[already_used:already_used + n_samples]
            else:
                # Resample if we've exhausted the pool
                sampled = np.random.choice(available_samples, size=n_samples, replace=True)

            new_measurements[m] = list(sampled)

        # Update statistics
        sampler.update_statistics(new_measurements)
        sampler.update_model_parameters()
        sampler.save_state()

        print(f"  Total collected: {sampler.get_total_samples()}/{total_budget}")
        print(f"  Updated estimates: A = {sampler.A_est:.4f}, f = {sampler.f_est:.4f}")

        batch_num += 1

    # Ensure we don't exceed budget - remove excess if needed
    total_collected = sampler.get_total_samples()
    if total_collected > total_budget:
        excess = total_collected - total_budget
        print(f"\nRemoving {excess} excess samples to match budget...")

        # Remove from bounces with most samples
        sorted_bounces = sorted(sampler.samples_collected.items(),
                               key=lambda x: x[1], reverse=True)

        for bounce, count in sorted_bounces:
            if excess <= 0:
                break

            to_remove = min(excess, count - 5)  # Keep at least 5 samples
            if to_remove > 0:
                # Remove from the end
                sampler.all_measurements[bounce] = sampler.all_measurements[bounce][:-to_remove]
                sampler.samples_collected[bounce] -= to_remove
                excess -= to_remove

        # Recalculate statistics
        for m in sampler.bounce_numbers:
            if len(sampler.all_measurements[m]) > 0:
                sampler.mean_estimates[m] = np.mean(sampler.all_measurements[m])
                if len(sampler.all_measurements[m]) > 1:
                    var = np.var(sampler.all_measurements[m], ddof=1)
                    sampler.variance_estimates[m] = max(var, 0.0001)

        sampler.update_model_parameters()
        print(f"Final total: {sampler.get_total_samples()} samples")

    return sampler


def analyze_adaptive_results(sampler, comparison_file='AB_decay_uniform_40.pickle'):
    """
    Analyze the results of adaptive sampling
    """
    print("\n" + "="*80)
    print("Final Results")
    print("="*80)

    # Final fit
    m_values = np.array(sampler.bounce_numbers)
    fid_values = np.array([sampler.mean_estimates[m] for m in m_values
                          if sampler.mean_estimates[m] is not None])
    m_values_valid = np.array([m for m in m_values
                               if sampler.mean_estimates[m] is not None])

    # Weighted fit using actual sample sizes and variance
    std_errors = []
    for m in m_values_valid:
        n = sampler.samples_collected[m]
        var = sampler.variance_estimates[m]
        std_err = np.sqrt(var / n) if n > 0 else 1.0
        std_errors.append(std_err)

    popt, pcov = curve_fit(exp, m_values_valid, fid_values,
                          sigma=std_errors, absolute_sigma=True)

    h = t.ppf((1 + 0.95) / 2., len(m_values_valid) - 2)
    f_adaptive = popt[1]
    f_unc_adaptive = h * np.sqrt(pcov[1,1])
    rel_unc_adaptive = (f_unc_adaptive / f_adaptive) * 100

    print(f"\nAdaptive Sampling:")
    print(f"  Estimated fidelity: {f_adaptive:.4f} ± {f_unc_adaptive:.4f}")
    print(f"  Relative uncertainty: {rel_unc_adaptive:.2f}%")
    print(f"  Total samples: {sampler.get_total_samples()}")
    print(f"  Final allocation: {sampler.samples_collected}")

    # Compare with uniform
    with open(comparison_file, 'rb') as f:
        uniform_data = pk.load(f)

    endpoints = uniform_data["endpoints"]
    fid_means = uniform_data["decay data"][0]
    fid_data = uniform_data["decay data"][1]

    m_uni = np.array(range(endpoints[0], endpoints[1]+1))
    fid_uni = np.array([fid_means[i] for i in m_uni])
    popt_uni, pcov_uni = curve_fit(exp, m_uni, fid_uni)

    f_uniform = popt_uni[1]
    f_unc_uniform = h * np.sqrt(pcov_uni[1,1])
    rel_unc_uniform = (f_unc_uniform / f_uniform) * 100

    print(f"\nUniform 40 (baseline):")
    print(f"  Estimated fidelity: {f_uniform:.4f} ± {f_unc_uniform:.4f}")
    print(f"  Relative uncertainty: {rel_unc_uniform:.2f}%")

    improvement = ((rel_unc_uniform - rel_unc_adaptive) / rel_unc_uniform) * 100

    print("\n" + "="*80)
    print(f"Improvement: {improvement:.2f}%")
    if improvement > 0:
        print(f"✓ Adaptive sampling achieved {improvement:.2f}% better precision!")
    else:
        print(f"✗ No improvement ({abs(improvement):.2f}% worse)")
    print("="*80)

    return sampler, f_adaptive, f_unc_adaptive


def visualize_adaptive_process(sampler):
    """
    Visualize the adaptive sampling process
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Evolution of sample allocation
    ax1 = axes[0, 0]
    n_iterations = len(sampler.allocation_history)

    # Stack plot showing cumulative allocation
    allocations_matrix = np.zeros((n_iterations, len(sampler.bounce_numbers)))
    for i, alloc in enumerate(sampler.allocation_history):
        for j, m in enumerate(sampler.bounce_numbers):
            allocations_matrix[i, j] = alloc[m]

    for j, m in enumerate(sampler.bounce_numbers[::3]):  # Plot every 3rd for clarity
        ax1.plot(range(n_iterations), allocations_matrix[:, sampler.bounce_numbers.index(m)],
                label=f'Bounce {m}', marker='o', markersize=4)

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Cumulative samples', fontsize=12)
    ax1.set_title('Evolution of Sample Allocation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter convergence
    ax2 = axes[0, 1]
    A_history = [p[0] for p in sampler.parameter_history]
    f_history = [p[1] for p in sampler.parameter_history]

    ax2_twin = ax2.twinx()
    line1 = ax2.plot(range(n_iterations), A_history, 'b-o', label='A', markersize=6)
    line2 = ax2_twin.plot(range(n_iterations), f_history, 'r-s', label='f', markersize=6)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Parameter A', fontsize=12, color='b')
    ax2_twin.set_ylabel('Parameter f', fontsize=12, color='r')
    ax2.set_title('Parameter Convergence', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')

    # Plot 3: Final allocation distribution
    ax3 = axes[1, 0]
    final_allocation = [sampler.samples_collected[m] for m in sampler.bounce_numbers]
    uniform_allocation = [40] * len(sampler.bounce_numbers)

    x = np.arange(len(sampler.bounce_numbers))
    width = 0.35
    ax3.bar(x - width/2, uniform_allocation, width, label='Uniform', color='gray', alpha=0.7)
    ax3.bar(x + width/2, final_allocation, width, label='Adaptive', color='green', alpha=0.7)
    ax3.set_xlabel('Bounce number', fontsize=12)
    ax3.set_ylabel('Number of samples', fontsize=12)
    ax3.set_title('Final Allocation: Adaptive vs Uniform', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sampler.bounce_numbers)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Variance evolution for selected bounces
    ax4 = axes[1, 1]
    selected_bounces = [2, 7, 12, 17, 20]

    for m in selected_bounces:
        var_history = [vh[m] for vh in sampler.variance_history]
        ax4.plot(range(n_iterations), var_history, '-o', label=f'Bounce {m}', markersize=5)

    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Variance estimate', fontsize=12)
    ax4.set_title('Variance Estimation Over Time', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig('adaptive_sampling_process.pdf', dpi=150)
    print("\nVisualization saved as: adaptive_sampling_process.pdf")


# Run adaptive experiment
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    sampler = simulate_adaptive_experiment(
        real_data_file='AB_decay_uniform_40.pickle',
        total_budget=760,
        initial_samples=10,  # Start with 10 samples per bounce
        batch_size=50        # Add 50 samples per batch
    )

    sampler_results, f_adaptive, f_unc_adaptive = analyze_adaptive_results(sampler)
    visualize_adaptive_process(sampler)

    # Save results in the same format as other experiments
    output_data = {
        "decay data": [sampler.mean_estimates, sampler.all_measurements],
        "endpoints": [2, 20],
        "alpha": 0.95,
        "samples_per_bounce": sampler.samples_collected
    }

    with open('AB_decay_adaptive.pickle', 'wb') as f:
        pk.dump(output_data, f)

    print("\nResults saved to: AB_decay_adaptive.pickle")
