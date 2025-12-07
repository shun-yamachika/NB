import pickle as pk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

# Exponential decay function
def exp(m, A, f):
    return A * f**m

plt.close()

# Load all datasets
datasets = {
    'Uniform 20': 'AB_decay_uniform_20.pickle',
    'Uniform 40': 'AB_decay_uniform_40.pickle',
    'Center Heavy': 'AB_decay_weighted.pickle',
    'Linear Increasing': 'AB_decay_linear_increasing.pickle',
    'Linear Decreasing': 'AB_decay_linear_decreasing.pickle',
    'Endpoints Heavy': 'AB_decay_endpoints_heavy.pickle',
    'Fisher Optimal': 'AB_decay_fisher_optimal.pickle'
}

colors = {
    'Uniform 20': '#FF6B6B',
    'Uniform 40': '#4ECDC4',
    'Center Heavy': '#45B7D1',
    'Linear Increasing': '#96CEB4',
    'Linear Decreasing': '#FFEAA7',
    'Endpoints Heavy': '#DDA15E',
    'Fisher Optimal': '#FF1493'
}

# Set up figure with two subplots
fig = plt.figure(figsize=(16, 10))

# Main plot
ax1 = plt.subplot(2, 1, 1)

results = {}

# Process each dataset
for label, filename in datasets.items():
    with open(filename, 'rb') as f:
        data = pk.load(f)
        endpoints = data["endpoints"]
        fid_means = data["decay data"][0]
        fid_data = data["decay data"][1]
        alpha = data.get("alpha", 0.95)
        samples_per_bounce = data.get("samples_per_bounce", None)

        # Prepare data for fitting
        m_values = np.array(range(endpoints[0], endpoints[1]+1))
        fidelity_values = np.array([fid_means[i] for i in range(endpoints[0], endpoints[1]+1)])

        # Weighted least squares fitting
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

        # Store results
        results[label] = {
            'endpoints': endpoints,
            'fid_means': fid_means,
            'fid_data': fid_data,
            'popt': popt,
            'pcov': pcov,
            'alpha': alpha,
            'm_values': m_values,
            'fidelity_values': fidelity_values,
            'samples_per_bounce': samples_per_bounce
        }

        color = colors[label]

        # Plot sequence length averages (main plot)
        ax1.scatter(range(endpoints[0], endpoints[1]+1),
                   [fid_means[i] for i in range(endpoints[0], endpoints[1]+1)],
                   color=color, label=f'{label}', s=80, alpha=0.9, zorder=10)

        # Plot exponential fit
        ax1.plot(range(endpoints[0], endpoints[1]+1),
                [exp(m, popt[0], popt[1]) for m in range(endpoints[0], endpoints[1]+1)],
                color=color, alpha=0.7, linewidth=2.5, zorder=5)

# Compute studentized confidence interval
h = t.ppf((1 + 0.95) / 2., 18 - 2)

# Set axes labels for main plot
ax1.set_xlabel("Number of A $\\to$ B $\\to$ A bounces", fontsize=18)
ax1.set_ylabel("Sequence mean $b_m$", fontsize=18)
ax1.set_xticks(np.arange(2, 21, 2))
ax1.legend(loc='upper right', fontsize=13, ncol=2)
ax1.set_title("Comparison of Different Sampling Patterns", fontsize=22, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Second subplot: Sample distribution visualization
ax2 = plt.subplot(2, 1, 2)

# Plot sample distributions
offset = 0
bar_width = 0.11  # Adjusted for 7 patterns
bounce_numbers = range(2, 21)

for i, (label, color_val) in enumerate([(k, colors[k]) for k in ['Uniform 20', 'Uniform 40', 'Center Heavy', 'Linear Increasing', 'Linear Decreasing', 'Endpoints Heavy', 'Fisher Optimal']]):
    samples = results[label]['samples_per_bounce']
    if samples is None:
        # Uniform distribution
        if 'Uniform 20' in label:
            samples = {m: 20 for m in range(2, 21)}
        else:
            samples = {m: 40 for m in range(2, 21)}

    sample_counts = [samples.get(m, 0) for m in bounce_numbers]
    positions = [x + offset for x in bounce_numbers]
    ax2.bar(positions, sample_counts, bar_width, label=label, color=color_val, alpha=0.8)
    offset += bar_width

ax2.set_xlabel("Number of bounces", fontsize=16)
ax2.set_ylabel("Number of samples", fontsize=16)
ax2.set_title("Sample Distribution per Bounce Number", fontsize=20, fontweight='bold')
ax2.set_xticks([x + bar_width * 2.5 for x in bounce_numbers])
ax2.set_xticklabels(bounce_numbers)
ax2.legend(loc='best', fontsize=11, ncol=3)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save figure
fig.savefig("all_patterns_comparison.pdf", transparent=False, dpi=150)
print("Comparison figure saved as all_patterns_comparison.pdf")

# Print summary statistics
print("\n" + "="*80)
print("Summary of All Sampling Patterns")
print("="*80)

summary_data = []
for label in ['Uniform 20', 'Uniform 40', 'Center Heavy', 'Linear Increasing', 'Linear Decreasing', 'Endpoints Heavy', 'Fisher Optimal']:
    popt = results[label]['popt']
    pcov = results[label]['pcov']
    uncertainty = h * np.sqrt(pcov[1,1])

    endpoints = results[label]['endpoints']
    fid_data = results[label]['fid_data']
    total_samples = sum(len(fid_data[i]) for i in range(endpoints[0], endpoints[1]+1))

    rel_uncertainty = (uncertainty/popt[1])*100

    summary_data.append({
        'name': label,
        'fidelity': popt[1],
        'uncertainty': uncertainty,
        'rel_uncertainty': rel_uncertainty,
        'total_samples': total_samples
    })

# Sort by relative uncertainty (best first)
summary_data.sort(key=lambda x: x['rel_uncertainty'])

print(f"\n{'Rank':<6}{'Pattern':<20}{'Fidelity':<15}{'Uncertainty':<15}{'Rel. Unc.':<12}{'Samples':<10}")
print("-"*80)
for rank, data in enumerate(summary_data, 1):
    print(f"{rank:<6}{data['name']:<20}{data['fidelity']:.4f}{' ± ':<3}{data['uncertainty']:.4f}{'   ':<3}"
          f"{data['rel_uncertainty']:>6.2f}%{'   ':<3}{data['total_samples']:<10}")

print("\n" + "="*80)
print("Key Insights:")
print("-"*80)

# Find best pattern among 760-sample ones
best_760 = [d for d in summary_data if d['total_samples'] == 760][0]
print(f"Best performing pattern (760 samples): {best_760['name']}")
print(f"  - Relative uncertainty: {best_760['rel_uncertainty']:.2f}%")

# Compare with uniform 40
uniform_40 = [d for d in summary_data if d['name'] == 'Uniform 40'][0]
print(f"\nUniform 40 baseline (760 samples):")
print(f"  - Relative uncertainty: {uniform_40['rel_uncertainty']:.2f}%")

if best_760['name'] != 'Uniform 40':
    improvement = ((uniform_40['rel_uncertainty'] - best_760['rel_uncertainty']) / uniform_40['rel_uncertainty']) * 100
    print(f"\nImprovement over Uniform 40: {improvement:.1f}%")

print("="*80)
