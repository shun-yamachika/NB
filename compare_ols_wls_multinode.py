"""
Experiment to compare Ordinary Least Squares (OLS) vs Weighted Least Squares (WLS)
for multi-node network RB fitting with random trial counts per sequence length.

This script:
1. Loads full simulation data for n=2,3,4,5,6 nodes (50 trials per m)
2. Randomly subsamples different trial counts for each m (5-50 trials)
3. Compares OLS vs WLS (method 1: sigma = 1/sqrt(n_trials))
4. Repeats x times and computes average relative uncertainty for each node count
"""

import pickle as pk
import numpy as np
from scipy.optimize import curve_fit

def exp(m, A, f):
    """Exponential decay model: A * f^m"""
    return A * f**m

def compute_relative_uncertainty(popt, pcov):
    """Compute relative uncertainty: b/a where a=f, b=std(f)"""
    f_value = popt[1]
    f_std = np.sqrt(pcov[1, 1])
    return f_std / f_value

def run_single_experiment(full_data, endpoints, n_samples_per_m, use_weights=False):
    """
    Run a single fitting experiment with given trial counts.

    Parameters:
    -----------
    full_data : dict
        Full simulation data with structure {"decay data": [means, raw_data], "endpoints": [min, max]}
    endpoints : list
        [min_bounces, max_bounces]
    n_samples_per_m : dict
        {m: n_samples} - number of trials to use for each m
    use_weights : bool
        If True, use weighted least squares with sigma = 1/sqrt(n_trials)

    Returns:
    --------
    relative_uncertainty : float
        Relative uncertainty of fitted fidelity parameter
    """
    min_m, max_m = endpoints
    m_values = list(range(min_m, max_m + 1))

    # Extract raw data for each m
    fid_raw_data = full_data["decay data"][1]

    # Subsample according to n_samples_per_m
    fid_means = []
    n_samples_list = []

    for m in m_values:
        n_samples = n_samples_per_m[m]
        subset = fid_raw_data[m][:n_samples]  # Take first n_samples trials
        fid_means.append(np.mean(subset))
        n_samples_list.append(n_samples)

    # Perform curve fitting
    if use_weights:
        # Weighted Least Squares: sigma = 1 / sqrt(n_trials)
        sigma = 1.0 / np.sqrt(np.array(n_samples_list))
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)
    else:
        # Ordinary Least Squares
        popt, pcov = curve_fit(exp, m_values, fid_means)

    return compute_relative_uncertainty(popt, pcov)

def main():
    # Experimental parameters
    x_iterations = 10  # Number of repetitions
    n_samples_min = 5
    n_samples_max = 50
    n_nodes_list = [2, 3, 4, 5, 6]

    # Storage for results
    results = {
        "ols": {n: [] for n in n_nodes_list},
        "wls": {n: [] for n in n_nodes_list},
        "settings": {
            "iterations": x_iterations,
            "n_samples_min": n_samples_min,
            "n_samples_max": n_samples_max,
            "n_nodes": n_nodes_list
        }
    }

    # Load all simulation data
    print("Loading multi-node simulation data...")
    full_data_dict = {}
    for n_nodes in n_nodes_list:
        with open(f'{n_nodes}_RB_decay_50samples.pickle', 'rb') as f:
            full_data_dict[n_nodes] = pk.load(f)
        endpoints = full_data_dict[n_nodes]["endpoints"]
        print(f"  {n_nodes}-node: m ∈ {endpoints}")

    print(f"\nRunning {x_iterations} experiments for each node count...")

    # Run experiments for each node count
    for n_nodes in n_nodes_list:
        print(f"\n{'='*60}")
        print(f"Processing {n_nodes}-node network")
        print(f"{'='*60}")

        full_data = full_data_dict[n_nodes]
        endpoints = full_data["endpoints"]
        min_m, max_m = endpoints
        m_values = list(range(min_m, max_m + 1))

        for iteration in range(x_iterations):
            # Generate random trial counts for each m
            np.random.seed(iteration)  # Same seed for all node counts for consistency
            n_samples_per_m = {m: np.random.randint(n_samples_min, n_samples_max + 1)
                              for m in m_values}

            print(f"\nIteration {iteration + 1}/{x_iterations}")
            print(f"  Trial counts: {list(n_samples_per_m.values())}")

            # Run OLS
            rel_unc_ols = run_single_experiment(full_data, endpoints, n_samples_per_m, use_weights=False)
            results["ols"][n_nodes].append(rel_unc_ols)
            print(f"  OLS relative uncertainty: {rel_unc_ols:.6f}")

            # Run WLS (same data, different fitting method)
            rel_unc_wls = run_single_experiment(full_data, endpoints, n_samples_per_m, use_weights=True)
            results["wls"][n_nodes].append(rel_unc_wls)
            print(f"  WLS relative uncertainty: {rel_unc_wls:.6f}")

    # Compute and display statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY (Multi-node)")
    print("="*70)
    print(f"Number of iterations: {x_iterations}")
    print(f"Trial count range: [{n_samples_min}, {n_samples_max}]")
    print()

    # Store summary statistics
    results["summary"] = {}

    for n_nodes in n_nodes_list:
        endpoints = full_data_dict[n_nodes]["endpoints"]
        mean_ols = np.mean(results["ols"][n_nodes])
        std_ols = np.std(results["ols"][n_nodes])
        mean_wls = np.mean(results["wls"][n_nodes])
        std_wls = np.std(results["wls"][n_nodes])
        improvement = (mean_ols - mean_wls) / mean_ols * 100

        results["summary"][n_nodes] = {
            "mean_ols": mean_ols,
            "std_ols": std_ols,
            "mean_wls": mean_wls,
            "std_wls": std_wls,
            "improvement_percent": improvement,
            "m_range": endpoints
        }

        print(f"\n{n_nodes}-node network (m ∈ {endpoints}):")
        print(f"  OLS: {mean_ols:.6f} ± {std_ols:.6f}")
        print(f"  WLS: {mean_wls:.6f} ± {std_wls:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")

    print("="*70)

    # Save results
    with open('ols_vs_wls_multinode_results.pickle', 'wb') as f:
        pk.dump(results, f)

    print("\nResults saved to: ols_vs_wls_multinode_results.pickle")

if __name__ == "__main__":
    main()
