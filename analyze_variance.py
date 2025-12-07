"""
Analyze the actual variance from simulation results to understand
why Fisher-optimal design didn't perform as expected.
"""

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp(m, A, f):
    return A * f**m

def load_and_analyze(filename):
    """Load data and compute actual variances"""
    with open(filename, 'rb') as f:
        data = pk.load(f)
        endpoints = data["endpoints"]
        fid_means = data["decay data"][0]
        fid_data = data["decay data"][1]
        samples_per_bounce = data.get("samples_per_bounce", None)

        # Compute actual variances
        variances = {}
        std_devs = {}
        std_errors = {}

        for m in range(endpoints[0], endpoints[1] + 1):
            samples = fid_data[m]
            n_samples = len(samples)

            mean_val = np.mean(samples)
            variance = np.var(samples, ddof=1) if n_samples > 1 else 0
            std_dev = np.std(samples, ddof=1) if n_samples > 1 else 0
            std_error = std_dev / np.sqrt(n_samples) if n_samples > 1 else 0

            variances[m] = variance
            std_devs[m] = std_dev
            std_errors[m] = std_error

        return {
            'endpoints': endpoints,
            'means': fid_means,
            'data': fid_data,
            'variances': variances,
            'std_devs': std_devs,
            'std_errors': std_errors,
            'samples_per_bounce': samples_per_bounce
        }

# Load uniform 40 data (baseline)
print("="*80)
print("Analysis of Actual Variance in Simulation Results")
print("="*80)

uniform_results = load_and_analyze('AB_decay_uniform_40.pickle')
fisher_results = load_and_analyze('AB_decay_fisher_optimal.pickle')

# Fit the model to get A and f estimates
m_values = np.array(range(2, 21))
uniform_fid = np.array([uniform_results['means'][m] for m in m_values])
fisher_fid = np.array([fisher_results['means'][m] for m in m_values])

# Fit uniform data
popt_uniform, _ = curve_fit(exp, m_values, uniform_fid)
A_uniform, f_uniform = popt_uniform

print(f"\nUniform 40 fit: A = {A_uniform:.4f}, f = {f_uniform:.4f}")

# Calculate theoretical sensitivity df/dm for each bounce
def sensitivity(m, A, f):
    """Sensitivity to parameter f"""
    return A * m * f**(m-1)

# Compare actual variance with theoretical model
print("\n" + "="*80)
print("Variance Analysis: Actual vs Theoretical Model")
print("="*80)

print(f"\n{'Bounce':<8}{'Mean':<12}{'Actual Var':<15}{'Actual SD':<15}{'SE (n=40)':<15}{'Sensitivity':<15}")
print("-"*95)

for m in m_values:
    mean_val = uniform_results['means'][m]
    var_actual = uniform_results['variances'][m]
    sd_actual = uniform_results['std_devs'][m]
    se_actual = uniform_results['std_errors'][m]
    sens = sensitivity(m, A_uniform, f_uniform)

    print(f"{m:<8}{mean_val:<12.4f}{var_actual:<15.6f}{sd_actual:<15.4f}{se_actual:<15.4f}{sens:<15.4f}")

# Key insight: Check if variance is proportional to mean
print("\n" + "="*80)
print("Testing Variance Model Assumptions")
print("="*80)

means = [uniform_results['means'][m] for m in m_values]
variances = [uniform_results['variances'][m] for m in m_values]
std_devs = [uniform_results['std_devs'][m] for m in m_values]

# Test different variance models
# Model 1: Constant variance
const_var_model = np.mean(variances)

# Model 2: Proportional to mean (Poisson-like)
proportional_var = [m for m in means]

# Model 3: Binomial-like: var = p*(1-p)
binomial_var = [m * (1 - m) for m in means]

# Model 4: Actually fit the relationship
# var = a * mean^b
log_means = np.log([m for m in means if m > 0])
log_vars = np.log([v for v in variances if v > 0])

if len(log_means) > 5:
    coeffs = np.polyfit(log_means, log_vars, 1)
    b_exp = coeffs[0]  # Power exponent
    a_coeff = np.exp(coeffs[1])
    print(f"\nActual variance relationship: var ≈ {a_coeff:.6f} * mean^{b_exp:.3f}")
else:
    b_exp = 1
    a_coeff = 1

# Calculate Fisher Information with ACTUAL variances
print("\n" + "="*80)
print("Fisher Information with Actual Variances")
print("="*80)

print(f"\n{'Bounce':<8}{'Sensitivity^2':<18}{'Actual Var':<15}{'FI/sample':<15}{'n (Uniform)':<12}{'n (Fisher)':<12}")
print("-"*95)

total_FI_uniform = 0
total_FI_fisher = 0

for m in m_values:
    sens = sensitivity(m, A_uniform, f_uniform)
    var_actual = uniform_results['variances'][m]

    # Avoid division by zero
    if var_actual > 1e-10:
        FI_per_sample = (sens**2) / var_actual
    else:
        FI_per_sample = 0

    n_uniform = 40
    n_fisher = fisher_results['samples_per_bounce'].get(m, 40) if fisher_results['samples_per_bounce'] else 40

    FI_contrib_uniform = n_uniform * FI_per_sample
    FI_contrib_fisher = n_fisher * FI_per_sample

    total_FI_uniform += FI_contrib_uniform
    total_FI_fisher += FI_contrib_fisher

    print(f"{m:<8}{sens**2:<18.4f}{var_actual:<15.6f}{FI_per_sample:<15.2f}{n_uniform:<12}{n_fisher:<12}")

print("-"*95)
print(f"{'Total':<8}{'':<18}{'':<15}{'':<15}{total_FI_uniform:<12.2f}{total_FI_fisher:<12.2f}")

improvement = ((total_FI_fisher - total_FI_uniform) / total_FI_uniform) * 100
print(f"\nActual Fisher Information improvement: {improvement:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Mean fidelity vs bounce
ax1 = axes[0, 0]
ax1.errorbar(m_values, means, yerr=std_devs, fmt='o-', capsize=5, label='Uniform 40', color='blue')
ax1.set_xlabel('Bounce number m', fontsize=12)
ax1.set_ylabel('Mean fidelity', fontsize=12)
ax1.set_title('Fidelity Decay with Standard Deviation', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Variance vs Mean (log-log)
ax2 = axes[0, 1]
valid_idx = [i for i, (m, v) in enumerate(zip(means, variances)) if m > 0 and v > 0]
valid_means = [means[i] for i in valid_idx]
valid_vars = [variances[i] for i in valid_idx]

ax2.loglog(valid_means, valid_vars, 'o', markersize=8, label='Actual data', color='red')
# Plot fitted model
mean_range = np.logspace(np.log10(min(valid_means)), np.log10(max(valid_means)), 100)
fitted_var = a_coeff * mean_range**b_exp
ax2.loglog(mean_range, fitted_var, '--', linewidth=2,
           label=f'Fit: var = {a_coeff:.4f} × mean^{b_exp:.2f}', color='blue')
ax2.set_xlabel('Mean fidelity', fontsize=12)
ax2.set_ylabel('Variance', fontsize=12)
ax2.set_title('Variance vs Mean (log-log scale)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Coefficient of Variation (CV = std/mean)
ax3 = axes[1, 0]
cv = [sd / m if m > 0 else 0 for sd, m in zip(std_devs, means)]
ax3.plot(m_values, cv, 'o-', linewidth=2, markersize=8, color='green')
ax3.set_xlabel('Bounce number m', fontsize=12)
ax3.set_ylabel('Coefficient of Variation (SD/Mean)', fontsize=12)
ax3.set_title('Relative Noise Level Across Bounces', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=np.mean(cv), color='red', linestyle='--', label=f'Mean CV = {np.mean(cv):.3f}')
ax3.legend()

# Plot 4: Fisher Information per sample (actual)
ax4 = axes[1, 1]
FI_per_sample_actual = []
for m in m_values:
    sens = sensitivity(m, A_uniform, f_uniform)
    var_actual = uniform_results['variances'][m]
    if var_actual > 1e-10:
        FI_ps = (sens**2) / var_actual
    else:
        FI_ps = 0
    FI_per_sample_actual.append(FI_ps)

ax4.bar(m_values, FI_per_sample_actual, color='steelblue', alpha=0.7)
ax4.set_xlabel('Bounce number m', fontsize=12)
ax4.set_ylabel('Fisher Information per sample (actual variance)', fontsize=12)
ax4.set_title('Actual FI Distribution Across Bounces', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('variance_analysis.pdf', dpi=150)
print("\n" + "="*80)
print("Variance analysis plot saved as: variance_analysis.pdf")
print("="*80)

# Key findings summary
print("\n" + "="*80)
print("KEY FINDINGS: Why Fisher Optimal Didn't Improve Performance")
print("="*80)

print("\n1. Variance Scaling:")
print(f"   - Actual variance scales as: var ∝ mean^{b_exp:.3f}")
if abs(b_exp - 2) < 0.3:
    print("   - This is close to var ∝ mean² (constant relative error)")
    print("   - In this regime, ALL bounce numbers have similar information content!")
elif abs(b_exp - 1) < 0.3:
    print("   - This is close to var ∝ mean (Poisson-like)")
else:
    print(f"   - Custom scaling relationship")

print("\n2. Coefficient of Variation:")
cv_range = max(cv) - min(cv)
cv_mean = np.mean(cv)
print(f"   - Mean CV: {cv_mean:.4f}")
print(f"   - CV range: {cv_range:.4f}")
if cv_range / cv_mean < 0.5:
    print("   - CV is relatively constant across bounces")
    print("   - This means relative noise is uniform → all bounces equally valuable")

print("\n3. Actual Fisher Information improvement:")
print(f"   - Improvement: {improvement:.2f}%")
if improvement < 5:
    print("   - Very small improvement validates that uniform sampling is near-optimal")
    print("   - The variance structure doesn't strongly favor any particular bounce numbers")

print("\n4. Conclusion:")
if abs(b_exp - 2) < 0.3 or cv_range / cv_mean < 0.5:
    print("   - For this system, the noise characteristics are such that")
    print("     uniform sampling is nearly optimal!")
    print("   - Fisher-optimal design provides minimal benefit because")
    print("     the relative information content is similar across all bounces")
else:
    print("   - The Fisher optimal design should provide some benefit")
    print("   - Limited improvement may be due to:")
    print("     a) Finite sample statistics (760 samples isn't huge)")
    print("     b) Uncertainty in variance estimates")
    print("     c) Other experimental factors")

print("="*80)
