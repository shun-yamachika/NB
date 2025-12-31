"""
Compare 4 fitting methods without using empirical standard deviations:
1. OLS: Ordinary Least Squares (no weights)
2. WLS-M: Weighted by bounce length m only
3. WLS-N: Weighted by trial count n only (Method 1)
4. WLS-MN: Weighted by both m and n
"""

import pickle as pk
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

def exp(m, A, f):
    """Exponential decay model: A * f^m"""
    return A * f**m

def compute_relative_uncertainty(popt, pcov):
    """Compute relative uncertainty: b/a where a=f, b=std(f)"""
    f_value = popt[1]
    f_std = np.sqrt(pcov[1, 1])
    return f_std / f_value

def estimate_std_model(full_data, endpoints):
    """
    Estimate std(m) = a + b*m using all 50 samples.
    Returns (intercept, slope) coefficients.
    """
    min_m, max_m = endpoints
    m_values = list(range(min_m, max_m + 1))
    fid_raw_data = full_data["decay data"][1]

    # Compute std for each m using all 50 samples
    stds = []
    for m in m_values:
        full_subset = fid_raw_data[m]  # All 50 trials
        stds.append(np.std(full_subset, ddof=1))

    # Fit linear model: std = intercept + slope*m
    slope, intercept, r_value, _, _ = linregress(m_values, stds)

    print(f"Estimated std model: std(m) = {intercept:.6f} + {slope:.6f} * m")
    print(f"  R² = {r_value**2:.4f}")

    return intercept, slope

def run_single_experiment(full_data, endpoints, n_samples_per_m, method="ols", std_model_params=None):
    """
    Run a single fitting experiment with different weighting schemes.

    Parameters:
    -----------
    method : str
        "ols": No weights
        "wls_m": Weighted by m only (std model)
        "wls_n": Weighted by n only (1/sqrt(n))
        "wls_mn": Weighted by both m and n
    std_model_params : tuple
        (intercept, slope) for std(m) = intercept + slope * m
    """
    min_m, max_m = endpoints
    m_values = list(range(min_m, max_m + 1))
    fid_raw_data = full_data["decay data"][1]

    # Subsample according to n_samples_per_m
    fid_means = []
    n_samples_list = []

    for m in m_values:
        n_samples = n_samples_per_m[m]
        subset = fid_raw_data[m][:n_samples]
        fid_means.append(np.mean(subset))
        n_samples_list.append(n_samples)

    # Compute weights based on method
    if method == "ols":
        # No weights
        popt, pcov = curve_fit(exp, m_values, fid_means)

    elif method == "wls_m":
        # Weight by m only: sigma = a + b*m
        if std_model_params is None:
            raise ValueError("std_model_params required for wls_m")
        intercept, slope = std_model_params
        sigma = intercept + slope * np.array(m_values)
        sigma = np.maximum(sigma, 1e-10)  # Avoid zero
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)

    elif method == "wls_n":
        # Weight by n only: sigma = 1/sqrt(n)
        sigma = 1.0 / np.sqrt(np.array(n_samples_list))
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)

    elif method == "wls_mn":
        # Weight by both m and n: sigma = (a + b*m) / sqrt(n)
        if std_model_params is None:
            raise ValueError("std_model_params required for wls_mn")
        intercept, slope = std_model_params
        sigma_m = intercept + slope * np.array(m_values)
        sigma_m = np.maximum(sigma_m, 1e-10)
        sigma = sigma_m / np.sqrt(np.array(n_samples_list))
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)

    else:
        raise ValueError(f"Unknown method: {method}")

    return compute_relative_uncertainty(popt, pcov), popt, pcov

def main():
    # Experimental parameters
    x_iterations = 10
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
    print()

    # Estimate std(m) model once using all data
    print("Estimating std(m) model from full dataset...")
    intercept, slope = estimate_std_model(full_data, endpoints)
    std_model_params = (intercept, slope)
    print()

    # Storage for results
    methods = {
        "OLS": {"method": "ols", "results": [], "params": []},
        "WLS-M": {"method": "wls_m", "results": [], "params": []},
        "WLS-N": {"method": "wls_n", "results": [], "params": []},
        "WLS-MN": {"method": "wls_mn", "results": [], "params": []}
    }

    # Run experiments
    print(f"Running {x_iterations} experiments...")
    print("="*80)

    for iteration in range(x_iterations):
        # Generate random trial counts for each m
        np.random.seed(iteration)
        n_samples_per_m = {m: np.random.randint(n_samples_min, n_samples_max + 1)
                          for m in m_values}

        print(f"\nIteration {iteration + 1}/{x_iterations}")
        print(f"  Trial counts: {list(n_samples_per_m.values())}")
        print()

        # Run each method
        for name, info in methods.items():
            method_type = info["method"]

            if method_type in ["wls_m", "wls_mn"]:
                rel_unc, popt, pcov = run_single_experiment(
                    full_data, endpoints, n_samples_per_m,
                    method=method_type, std_model_params=std_model_params
                )
            else:
                rel_unc, popt, pcov = run_single_experiment(
                    full_data, endpoints, n_samples_per_m,
                    method=method_type
                )

            info["results"].append(rel_unc)
            info["params"].append(popt)

            print(f"  {name:10s}: f={popt[1]:.6f}, rel_unc={rel_unc:.6f}")

    # Compute statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Number of iterations: {x_iterations}")
    print(f"Trial count range: [{n_samples_min}, {n_samples_max}]")
    print(f"Sequence length range: m ∈ [{min_m}, {max_m}]")
    print(f"std(m) model: std = {intercept:.6f} + {slope:.6f} * m")
    print()

    # Baseline: OLS
    ols_mean = np.mean(methods["OLS"]["results"])

    print(f"{'Method':<12} {'Mean f':<15} {'Mean Rel Unc':<20} {'Improvement vs OLS':<20}")
    print("-"*80)

    for name, info in methods.items():
        results = info["results"]
        params = info["params"]

        f_values = [p[1] for p in params]
        mean_f = np.mean(f_values)
        std_f = np.std(f_values)

        mean_unc = np.mean(results)
        std_unc = np.std(results)

        improvement = (ols_mean - mean_unc) / ols_mean * 100 if name != "OLS" else 0.0

        print(f"{name:<12} {mean_f:.6f}±{std_f:.6f}  {mean_unc:.6f}±{std_unc:.6f}     {improvement:+.2f}%")

    print("="*80)

    # Save results
    results_dict = {
        method: {
            "rel_uncertainties": info["results"],
            "fidelity_params": [p[1] for p in info["params"]],
            "mean_rel_unc": np.mean(info["results"]),
            "std_rel_unc": np.std(info["results"])
        }
        for method, info in methods.items()
    }
    results_dict["settings"] = {
        "iterations": x_iterations,
        "n_samples_min": n_samples_min,
        "n_samples_max": n_samples_max,
        "m_range": [min_m, max_m],
        "std_model": {"intercept": intercept, "slope": slope}
    }

    with open('compare_4methods_results.pickle', 'wb') as f:
        pk.dump(results_dict, f)

    print("\nResults saved to: compare_4methods_results.pickle")

if __name__ == "__main__":
    main()
