"""
Compare 5 fitting methods:
1. OLS: Ordinary Least Squares (no weights)
2. WLS-MN: Model-based (m+n, no empirical data)
3. WLS-Empirical: Empirical only (Method 2)
4. WLS-Combined: Model + Empirical (Shrinkage)
5. WLS-N: Trial count only (for reference, Method 1)
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

def shrinkage_factor(n, method="sqrt"):
    """
    Compute shrinkage factor alpha based on trial count n.

    alpha = 0: use model only
    alpha = 1: use empirical only

    Methods:
    - "sqrt": alpha = min(sqrt(n/30), 1)
    - "linear": alpha = min(n/30, 1)
    - "step": alpha = 1 if n >= 20 else 0.5
    """
    if method == "sqrt":
        return min(np.sqrt(n / 30), 1.0)
    elif method == "linear":
        return min(n / 30, 1.0)
    elif method == "step":
        return 1.0 if n >= 20 else 0.5
    else:
        raise ValueError(f"Unknown shrinkage method: {method}")

def run_single_experiment(full_data, endpoints, n_samples_per_m, method="ols",
                          std_model_params=None, shrinkage_method="sqrt"):
    """
    Run a single fitting experiment with different weighting schemes.

    Parameters:
    -----------
    method : str
        "ols": No weights
        "wls_n": Weighted by n only (1/sqrt(n))
        "wls_mn": Weighted by m and n (model-based)
        "wls_empirical": Weighted by empirical std
        "wls_combined": Weighted by combined model+empirical
    std_model_params : tuple
        (intercept, slope) for std(m) = intercept + slope * m
    shrinkage_method : str
        Method for computing shrinkage factor in wls_combined
    """
    min_m, max_m = endpoints
    m_values = list(range(min_m, max_m + 1))
    fid_raw_data = full_data["decay data"][1]

    # Subsample and compute empirical statistics
    fid_means = []
    fid_stds = []
    n_samples_list = []

    for m in m_values:
        n_samples = n_samples_per_m[m]
        subset = fid_raw_data[m][:n_samples]
        fid_means.append(np.mean(subset))
        fid_stds.append(np.std(subset, ddof=1))
        n_samples_list.append(n_samples)

    # Compute weights based on method
    if method == "ols":
        # No weights
        popt, pcov = curve_fit(exp, m_values, fid_means)

    elif method == "wls_n":
        # Weight by n only: sigma = 1/sqrt(n)
        sigma = 1.0 / np.sqrt(np.array(n_samples_list))
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)

    elif method == "wls_mn":
        # Weight by m and n: sigma = (a + b*m) / sqrt(n)
        if std_model_params is None:
            raise ValueError("std_model_params required for wls_mn")
        intercept, slope = std_model_params
        sigma_m = intercept + slope * np.array(m_values)
        sigma_m = np.maximum(sigma_m, 1e-10)
        sigma = sigma_m / np.sqrt(np.array(n_samples_list))
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)

    elif method == "wls_empirical":
        # Weight by empirical std: sigma = std(data) / sqrt(n)
        sigma = np.array(fid_stds) / np.sqrt(np.array(n_samples_list))
        sigma = np.where(sigma > 0, sigma, 1e-10)
        popt, pcov = curve_fit(exp, m_values, fid_means, sigma=sigma, absolute_sigma=True)

    elif method == "wls_combined":
        # Combined: shrinkage between model and empirical
        if std_model_params is None:
            raise ValueError("std_model_params required for wls_combined")
        intercept, slope = std_model_params

        # Model-based std
        sigma_model = intercept + slope * np.array(m_values)
        sigma_model = np.maximum(sigma_model, 1e-10)

        # Empirical std
        sigma_empirical = np.array(fid_stds)
        sigma_empirical = np.where(sigma_empirical > 0, sigma_empirical, 1e-10)

        # Shrinkage combination
        alphas = np.array([shrinkage_factor(n, shrinkage_method) for n in n_samples_list])
        sigma_combined = alphas * sigma_empirical + (1 - alphas) * sigma_model

        # Apply trial count correction
        sigma = sigma_combined / np.sqrt(np.array(n_samples_list))
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
        "WLS-N": {"method": "wls_n", "results": [], "params": []},
        "WLS-MN": {"method": "wls_mn", "results": [], "params": []},
        "WLS-Empirical": {"method": "wls_empirical", "results": [], "params": []},
        "WLS-Combined": {"method": "wls_combined", "results": [], "params": []}
    }

    # Run experiments
    print(f"Running {x_iterations} experiments...")
    print("="*90)

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

            if method_type in ["wls_mn", "wls_combined"]:
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

            print(f"  {name:16s}: f={popt[1]:.6f}, rel_unc={rel_unc:.6f}")

    # Compute statistics
    print("\n" + "="*90)
    print("RESULTS SUMMARY")
    print("="*90)
    print(f"Number of iterations: {x_iterations}")
    print(f"Trial count range: [{n_samples_min}, {n_samples_max}]")
    print(f"Sequence length range: m ∈ [{min_m}, {max_m}]")
    print(f"std(m) model: std = {intercept:.6f} + {slope:.6f} * m")
    print()

    # Baseline: OLS
    ols_mean = np.mean(methods["OLS"]["results"])

    print(f"{'Method':<18} {'Mean f':<15} {'Mean Rel Unc':<20} {'Improvement vs OLS':<20}")
    print("-"*90)

    for name, info in methods.items():
        results = info["results"]
        params = info["params"]

        f_values = [p[1] for p in params]
        mean_f = np.mean(f_values)
        std_f = np.std(f_values)

        mean_unc = np.mean(results)
        std_unc = np.std(results)

        improvement = (ols_mean - mean_unc) / ols_mean * 100 if name != "OLS" else 0.0

        print(f"{name:<18} {mean_f:.6f}±{std_f:.6f}  {mean_unc:.6f}±{std_unc:.6f}     {improvement:+.2f}%")

    print("="*90)

    # Additional analysis: Compare WLS-MN, WLS-Empirical, and WLS-Combined
    print("\nComparison of advanced methods:")
    print("-"*90)

    mn_mean = np.mean(methods["WLS-MN"]["results"])
    emp_mean = np.mean(methods["WLS-Empirical"]["results"])
    comb_mean = np.mean(methods["WLS-Combined"]["results"])

    print(f"WLS-MN (model only):        {mn_mean:.6f} (improvement: {(ols_mean - mn_mean) / ols_mean * 100:+.2f}%)")
    print(f"WLS-Empirical (data only):  {emp_mean:.6f} (improvement: {(ols_mean - emp_mean) / ols_mean * 100:+.2f}%)")
    print(f"WLS-Combined (model+data):  {comb_mean:.6f} (improvement: {(ols_mean - comb_mean) / ols_mean * 100:+.2f}%)")
    print()
    print(f"Combined vs Model-only:     {(mn_mean - comb_mean) / mn_mean * 100:+.2f}% additional improvement")
    print(f"Combined vs Empirical:      {(emp_mean - comb_mean) / emp_mean * 100:+.2f}% difference")
    print("="*90)

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

    with open('compare_5methods_results.pickle', 'wb') as f:
        pk.dump(results_dict, f)

    print("\nResults saved to: compare_5methods_results.pickle")

if __name__ == "__main__":
    main()
