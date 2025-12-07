import pickle as pk
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t

def exp(m, A, f):
    return A * f**m

# Load both datasets
with open('AB_decay_uniform_40.pickle', 'rb') as f:
    uniform_data = pk.load(f)

with open('AB_decay_fisher_corrected.pickle', 'rb') as f:
    fisher_data = pk.load(f)

# Fit both
def fit_and_analyze(data, label):
    endpoints = data["endpoints"]
    fid_means = data["decay data"][0]
    fid_data = data["decay data"][1]
    samples_per_bounce = data.get("samples_per_bounce", None)

    m_values = np.array(range(endpoints[0], endpoints[1]+1))
    fidelity_values = np.array([fid_means[i] for i in m_values])

    # Weighted fitting
    if samples_per_bounce is None:
        popt, pcov = curve_fit(exp, m_values, fidelity_values)
    else:
        std_errors = []
        for m in m_values:
            fidelity_samples = fid_data[m]
            sample_std = np.std(fidelity_samples, ddof=1)
            n_samples = samples_per_bounce.get(m, len(fidelity_samples))
            std_errors.append(sample_std / np.sqrt(n_samples))

        weights = np.array(std_errors)
        popt, pcov = curve_fit(exp, m_values, fidelity_values, sigma=weights, absolute_sigma=True)

    # Compute uncertainty
    h = t.ppf((1 + 0.95) / 2., 18 - 2)
    f_est = popt[1]
    f_uncertainty = h * np.sqrt(pcov[1,1])
    rel_uncertainty = (f_uncertainty / f_est) * 100

    total_samples = sum(len(fid_data[i]) for i in m_values)

    print(f"\n{label}:")
    print(f"  Estimated fidelity: {f_est:.4f} ± {f_uncertainty:.4f}")
    print(f"  Relative uncertainty: {rel_uncertainty:.2f}%")
    print(f"  Total samples: {total_samples}")

    return f_est, f_uncertainty, rel_uncertainty

print("="*80)
print("Comparison: Uniform 40 vs Fisher Corrected")
print("="*80)

f_uniform, unc_uniform, rel_uniform = fit_and_analyze(uniform_data, "Uniform 40")
f_fisher, unc_fisher, rel_fisher = fit_and_analyze(fisher_data, "Fisher Corrected")

print("\n" + "="*80)
print("Comparison Summary")
print("="*80)

improvement = ((rel_uniform - rel_fisher) / rel_uniform) * 100
print(f"\nRelative uncertainty improvement: {improvement:.2f}%")

if improvement > 0:
    print(f"✓ Fisher Corrected achieved {improvement:.2f}% better precision!")
else:
    print(f"✗ Fisher Corrected did not improve (worse by {abs(improvement):.2f}%)")

print("\nExpected improvement (theoretical): 15.84%")
print(f"Actual improvement: {improvement:.2f}%")

efficiency = (improvement / 15.84) * 100 if improvement > 0 else 0
print(f"Efficiency (actual/expected): {efficiency:.1f}%")

print("="*80)
