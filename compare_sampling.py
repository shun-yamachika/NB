import pickle as pk
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非対話モードに設定
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

# Exponential decay function
def exp(m, A, f):
    return A * f**m

plt.close()

# Load all three datasets
datasets = {
    'Uniform 20': 'AB_decay_uniform_20.pickle',
    'Uniform 40': 'AB_decay_uniform_40.pickle',
    'Weighted': 'AB_decay_weighted.pickle'
}

colors = {
    'Uniform 20': 'red',
    'Uniform 40': 'blue',
    'Weighted': 'green'
}

# Set up figure
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()

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
            # Uniform sampling: use unweighted fit
            popt, pcov = curve_fit(exp, m_values, fidelity_values)
        else:
            # Empirical weights (based on actual sample standard deviation)
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
            'fidelity_values': fidelity_values
        }

        color = colors[label]

        # Plot sequence length averages
        ax.scatter(range(endpoints[0], endpoints[1]+1),
                  [fid_means[i] for i in range(endpoints[0], endpoints[1]+1)],
                  color=color, label=f'{label} (f={popt[1]:.3f})', s=50, alpha=0.8)

        # Plot individual sequence means (lighter, smaller)
        max_samples = max(len(fid_data[i]) for i in range(endpoints[0], endpoints[1]+1))
        for k in range(max_samples):
            bounce_nums = []
            fidelities = []
            for i in range(endpoints[0], endpoints[1]+1):
                if k < len(fid_data[i]):
                    bounce_nums.append(i)
                    fidelities.append(fid_data[i][k])
            if bounce_nums:
                ax.scatter(bounce_nums, fidelities, color=color, alpha=0.15, s=5)

        # Plot exponential fit
        ax.plot(range(endpoints[0], endpoints[1]+1),
               [exp(m, popt[0], popt[1]) for m in range(endpoints[0], endpoints[1]+1)],
               color=color, alpha=0.7, linewidth=2)

# Compute studentized confidence interval correction (using 18 data points)
h = t.ppf((1 + 0.95) / 2., 18 - 2)

# Add text with results
y_pos = 0.45
for label in ['Uniform 20', 'Uniform 40', 'Weighted']:
    popt = results[label]['popt']
    pcov = results[label]['pcov']
    color = colors[label]
    text = f"{label}: f = {popt[1]:.3f} ± {h*np.sqrt(pcov[1,1]):.3f}"
    ax.text(5.5, y_pos, text, fontsize=13, color=color)
    y_pos -= 0.04

# Set axes labels
ax.set_xlabel("Number of A $\\to$ B $\\to$ A bounces", fontsize=18)
ax.set_ylabel("Sequence mean $b_m$", fontsize=18)
ax.set_xticks(np.arange(2, 21, 2))
ax.legend(loc='best', fontsize=12)
ax.set_title("Comparison of Sampling Strategies", fontsize=20)

# Save figure
fig.savefig("sampling_comparison.pdf", transparent=True)
print("Comparison figure saved as sampling_comparison.pdf")

# Print summary statistics
print("\n" + "="*60)
print("Summary of Results")
print("="*60)
for label in ['Uniform 20', 'Uniform 40', 'Weighted']:
    popt = results[label]['popt']
    pcov = results[label]['pcov']
    uncertainty = h * np.sqrt(pcov[1,1])
    print(f"\n{label}:")
    print(f"  Estimated fidelity: {popt[1]:.4f} ± {uncertainty:.4f}")
    print(f"  Relative uncertainty: {(uncertainty/popt[1])*100:.2f}%")

    # Calculate total number of samples
    endpoints = results[label]['endpoints']
    fid_data = results[label]['fid_data']
    total_samples = sum(len(fid_data[i]) for i in range(endpoints[0], endpoints[1]+1))
    print(f"  Total samples: {total_samples}")
print("="*60)
