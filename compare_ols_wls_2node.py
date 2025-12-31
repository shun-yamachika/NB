"""
Experiment to compare Ordinary Least Squares (OLS) vs Weighted Least Squares (WLS)
for 2-node network RB fitting with random trial counts per sequence length.

This script:
1. Loads full simulation data (50 trials per m)
2. Randomly subsamples different trial counts for each m (5-50 trials)
3. Compares OLS vs WLS (method 1: sigma = 1/sqrt(n_trials))
4. Repeats x times and computes average relative uncertainty
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
    use_weights : str or bool
        False: OLS (no weights)
        "theoretical": WLS with sigma = 1/sqrt(n_trials)
        "empirical": WLS with sigma = std(data)/sqrt(n_trials)

    Returns:
    --------
    relative_uncertainty : float
        Relative uncertainty of fitted fidelity parameter
    popt : array
        Fitted parameters
    pcov : array
        Covariance matrix
    """
    min_m, max_m = endpoints
    m_values = list(range(min_m, max_m + 1))

    # Extract raw data for each m
    fid_raw_data = full_data["decay data"][1]

    # Subsample according to n_samples_per_m
    fid_means = []
    fid_stds = []
    n_samples_list = []

    for m in m_values:
        n_samples = n_samples_per_m[m]
        subset = fid_raw_data[m][:n_samples]  # Take first n_samples trials
        fid_means.append(np.mean(subset))
        fid_stds.append(np.std(subset, ddof=1))  # Sample standard deviation
        n_samples_list.append(n_samples)

    # Perform curve fitting
    if use_weights == "theoretical":
        # Method 1: Theoretical WLS with sigma = 1 / sqrt(n_trials)
        sigma = 1.0 / np.sqrt(np.array(n_samples_list))
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)
    elif use_weights == "empirical":
        # Method 2: Empirical WLS with sigma = std(data) / sqrt(n_trials)
        sigma = np.array(fid_stds) / np.sqrt(np.array(n_samples_list))
        # Handle cases where std is 0 (shouldn't happen with n>=5, but just in case)
        sigma = np.where(sigma > 0, sigma, 1e-10)
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)
    else:
        # Ordinary Least Squares
        popt, pcov = curve_fit(exp, m_values, fid_means)

    return compute_relative_uncertainty(popt, pcov), popt, pcov

def main():
    # Experimental parameters
    x_iterations = 10  # Number of repetitions
    n_samples_min = 5
    n_samples_max = 50

    # Load full simulation data
    print("Loading 2-node simulation data...")
    with open('AB_decay_50samples.pickle', 'rb') as f:
        full_data = pk.load(f)

    endpoints = full_data["endpoints"]
    min_m, max_m = endpoints
    m_values = list(range(min_m, max_m + 1))

    print(f"Data loaded: m ∈ [{min_m}, {max_m}] ({len(m_values)} points)")

    # Storage for results
    results_ols = []
    results_wls_theoretical = []
    results_wls_empirical = []
    params_ols = []
    params_wls_theoretical = []
    params_wls_empirical = []

    # Run experiments
    print(f"\nRunning {x_iterations} experiments...")
    for iteration in range(x_iterations):
        # Generate random trial counts for each m
        np.random.seed(iteration)  # For reproducibility
        n_samples_per_m = {m: np.random.randint(n_samples_min, n_samples_max + 1)
                          for m in m_values}

        print(f"\nIteration {iteration + 1}/{x_iterations}")
        print(f"  Trial counts: {list(n_samples_per_m.values())}")

        # Run OLS
        rel_unc_ols, popt_ols, pcov_ols = run_single_experiment(full_data, endpoints, n_samples_per_m, use_weights=False)
        results_ols.append(rel_unc_ols)
        params_ols.append(popt_ols)
        print(f"  OLS:             f={popt_ols[1]:.6f}, rel_unc={rel_unc_ols:.6f}")

        # Run WLS with theoretical weights (Method 1)
        rel_unc_wls_th, popt_wls_th, pcov_wls_th = run_single_experiment(full_data, endpoints, n_samples_per_m, use_weights="theoretical")
        results_wls_theoretical.append(rel_unc_wls_th)
        params_wls_theoretical.append(popt_wls_th)
        print(f"  WLS (Method 1):  f={popt_wls_th[1]:.6f}, rel_unc={rel_unc_wls_th:.6f}")

        # Run WLS with empirical weights (Method 2)
        rel_unc_wls_emp, popt_wls_emp, pcov_wls_emp = run_single_experiment(full_data, endpoints, n_samples_per_m, use_weights="empirical")
        results_wls_empirical.append(rel_unc_wls_emp)
        params_wls_empirical.append(popt_wls_emp)
        print(f"  WLS (Method 2):  f={popt_wls_emp[1]:.6f}, rel_unc={rel_unc_wls_emp:.6f}")

    # Compute statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY (2-node)")
    print("="*70)
    print(f"Number of iterations: {x_iterations}")
    print(f"Trial count range: [{n_samples_min}, {n_samples_max}]")
    print(f"Sequence length range: m ∈ [{min_m}, {max_m}]")
    print()

    # Extract fidelity parameters
    f_ols = [p[1] for p in params_ols]
    f_wls_th = [p[1] for p in params_wls_theoretical]
    f_wls_emp = [p[1] for p in params_wls_empirical]

    print(f"Ordinary Least Squares (OLS):")
    print(f"  Mean fidelity f:           {np.mean(f_ols):.6f} ± {np.std(f_ols):.6f}")
    print(f"  Mean relative uncertainty: {np.mean(results_ols):.6f} ± {np.std(results_ols):.6f}")
    print()

    print(f"Weighted Least Squares - Method 1 (sigma = 1/sqrt(n)):")
    print(f"  Mean fidelity f:           {np.mean(f_wls_th):.6f} ± {np.std(f_wls_th):.6f}")
    print(f"  Mean relative uncertainty: {np.mean(results_wls_theoretical):.6f} ± {np.std(results_wls_theoretical):.6f}")
    improvement_th = (np.mean(results_ols) - np.mean(results_wls_theoretical)) / np.mean(results_ols) * 100
    print(f"  vs OLS improvement:        {improvement_th:+.2f}%")
    print()

    print(f"Weighted Least Squares - Method 2 (sigma = std/sqrt(n)):")
    print(f"  Mean fidelity f:           {np.mean(f_wls_emp):.6f} ± {np.std(f_wls_emp):.6f}")
    print(f"  Mean relative uncertainty: {np.mean(results_wls_empirical):.6f} ± {np.std(results_wls_empirical):.6f}")
    improvement_emp = (np.mean(results_ols) - np.mean(results_wls_empirical)) / np.mean(results_ols) * 100
    print(f"  vs OLS improvement:        {improvement_emp:+.2f}%")
    print("="*70)

    # Save results
    results = {
        "ols": results_ols,
        "wls_theoretical": results_wls_theoretical,
        "wls_empirical": results_wls_empirical,
        "mean_ols": np.mean(results_ols),
        "mean_wls_theoretical": np.mean(results_wls_theoretical),
        "mean_wls_empirical": np.mean(results_wls_empirical),
        "std_ols": np.std(results_ols),
        "std_wls_theoretical": np.std(results_wls_theoretical),
        "std_wls_empirical": np.std(results_wls_empirical),
        "improvement_theoretical_percent": improvement_th,
        "improvement_empirical_percent": improvement_emp,
        "settings": {
            "iterations": x_iterations,
            "n_samples_min": n_samples_min,
            "n_samples_max": n_samples_max,
            "m_range": [min_m, max_m]
        }
    }

    with open('ols_vs_wls_2node_results.pickle', 'wb') as f:
        pk.dump(results, f)

    print("\nResults saved to: ols_vs_wls_2node_results.pickle")

if __name__ == "__main__":
    main()
