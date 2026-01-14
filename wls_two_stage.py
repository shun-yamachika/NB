"""
Two-stage WLS: データ駆動型モデル選択

Stage 1: 測定データから最適な分散モデルを自動選択
Stage 2: モデルベースの重みでWLSフィッティング
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ================================================================================
# データ読み込み
# ================================================================================
print("="*80)
print("Two-stage WLS with Automatic Model Selection")
print("="*80)

with open('AB_decay_200samples_main.pickle', 'rb') as f:
    data = pk.load(f)

fid_raw_data = data["decay data"][1]
all_m_values = sorted(fid_raw_data.keys())

m_range = (2, 40)
samples_per_m = 200

m_values = [m for m in all_m_values if m_range[0] <= m <= m_range[1]]
m_array = np.array(m_values)

print(f"\nData: m ∈ [{m_range[0]}, {m_range[1]}], n={samples_per_m}")

# データ準備
fid_means = []
fid_stds = []

for m in m_values:
    data_m = fid_raw_data[m][:samples_per_m]
    fid_means.append(np.mean(data_m))
    fid_stds.append(np.std(data_m, ddof=1))

fid_means = np.array(fid_means)
fid_stds = np.array(fid_stds)

# ================================================================================
# Stage 1: 最適分散モデルの自動選択
# ================================================================================
print("\n" + "="*80)
print("STAGE 1: Variance Model Selection")
print("="*80)

models = {}

# Model 1: Linear σ = a + b*m
slope, intercept, r_value, _, _ = linregress(m_array, fid_stds)
sigma_linear = intercept + slope * m_array
r2_linear = r_value**2
models['Linear'] = {
    'sigma': sigma_linear,
    'r2': r2_linear,
    'params': {'a': intercept, 'b': slope},
    'formula': f'σ(m) = {intercept:.5f} + {slope:.5f}·m'
}

# Model 2: Power law σ = a * m^b
log_m = np.log(m_array)
log_sigma = np.log(fid_stds)
slope_log, intercept_log, r_value_log, _, _ = linregress(log_m, log_sigma)
a_power = np.exp(intercept_log)
b_power = slope_log
sigma_power = a_power * (m_array ** b_power)
r2_power = r_value_log**2
models['Power law'] = {
    'sigma': sigma_power,
    'r2': r2_power,
    'params': {'a': a_power, 'b': b_power},
    'formula': f'σ(m) = {a_power:.5f}·m^{b_power:.3f}'
}

# Model 3: Quadratic σ = a + b*m + c*m²
coeffs = np.polyfit(m_array, fid_stds, 2)
sigma_quadratic = np.polyval(coeffs, m_array)
ss_res = np.sum((fid_stds - sigma_quadratic)**2)
ss_tot = np.sum((fid_stds - np.mean(fid_stds))**2)
r2_quad = 1 - ss_res/ss_tot
models['Quadratic'] = {
    'sigma': sigma_quadratic,
    'r2': r2_quad,
    'params': {'a': coeffs[2], 'b': coeffs[1], 'c': coeffs[0]},
    'formula': f'σ(m) = {coeffs[2]:.5f} + {coeffs[1]:.5f}·m + {coeffs[0]:.2e}·m²'
}

# Model 4: Constant (baseline) σ = const
sigma_constant = np.full_like(m_array, np.mean(fid_stds), dtype=float)
ss_res_const = np.sum((fid_stds - sigma_constant)**2)
r2_const = 1 - ss_res_const/ss_tot
models['Constant'] = {
    'sigma': sigma_constant,
    'r2': r2_const,
    'params': {'const': np.mean(fid_stds)},
    'formula': f'σ(m) = {np.mean(fid_stds):.5f}'
}

# 最適モデルを選択（R²が最大）
print("\nModel candidates:")
print(f"{'Model':<15} {'R²':<10} {'Formula'}")
print("-"*80)
for name, model_info in models.items():
    print(f"{name:<15} {model_info['r2']:<10.4f} {model_info['formula']}")

best_model_name = max(models, key=lambda k: models[k]['r2'])
best_model = models[best_model_name]

print(f"\n→ Selected model: {best_model_name} (R² = {best_model['r2']:.4f})")
print(f"  {best_model['formula']}")

sigma_model = best_model['sigma']

# ================================================================================
# 指数減衰モデル
# ================================================================================
def exponential_decay(m, A, f):
    return A * (f ** m)

# ================================================================================
# 3つの手法で比較
# ================================================================================
print("\n" + "="*80)
print("STAGE 2: Fitting Comparison")
print("="*80)

# 1. OLS
popt_ols, pcov_ols = curve_fit(exponential_decay, m_array, fid_means,
                                p0=[0.5, 0.9])
f_ols = popt_ols[1]
f_err_ols = np.sqrt(pcov_ols[1, 1])

# 2. WLS-SE (direct empirical)
standard_errors_empirical = fid_stds / np.sqrt(samples_per_m)
popt_wls_se, pcov_wls_se = curve_fit(exponential_decay, m_array, fid_means,
                                      sigma=standard_errors_empirical,
                                      absolute_sigma=True, p0=[0.5, 0.9])
f_wls_se = popt_wls_se[1]
f_err_wls_se = np.sqrt(pcov_wls_se[1, 1])

# 3. WLS-Two-stage (model-based)
standard_errors_model = sigma_model / np.sqrt(samples_per_m)
popt_wls_ts, pcov_wls_ts = curve_fit(exponential_decay, m_array, fid_means,
                                      sigma=standard_errors_model,
                                      absolute_sigma=True, p0=[0.5, 0.9])
f_wls_ts = popt_wls_ts[1]
f_err_wls_ts = np.sqrt(pcov_wls_ts[1, 1])

# ================================================================================
# 結果表示
# ================================================================================
print("\nFitting Results:")
print(f"{'Method':<20} {'f':<12} {'Uncertainty':<15} {'Rel. Unc. (%)':<15} {'Improvement'}")
print("-"*80)

rel_unc_ols = f_err_ols / f_ols * 100
rel_unc_wls_se = f_err_wls_se / f_wls_se * 100
rel_unc_wls_ts = f_err_wls_ts / f_wls_ts * 100

print(f"{'OLS':<20} {f_ols:<12.6f} {f_err_ols:<15.6f} {rel_unc_ols:<15.3f} baseline")
print(f"{'WLS-SE':<20} {f_wls_se:<12.6f} {f_err_wls_se:<15.6f} {rel_unc_wls_se:<15.3f} "
      f"+{(1-rel_unc_wls_se/rel_unc_ols)*100:.1f}%")
print(f"{'WLS-Two-stage':<20} {f_wls_ts:<12.6f} {f_err_wls_ts:<15.6f} {rel_unc_wls_ts:<15.3f} "
      f"+{(1-rel_unc_wls_ts/rel_unc_ols)*100:.1f}%")

# ================================================================================
# 重みの比較
# ================================================================================
weights_ols = np.ones_like(m_array)  # Equal weights
weights_wls_se = samples_per_m / (fid_stds ** 2)
weights_wls_ts = samples_per_m / (sigma_model ** 2)

# 正規化
weights_ols_norm = weights_ols / weights_ols.max()
weights_wls_se_norm = weights_wls_se / weights_wls_se.max()
weights_wls_ts_norm = weights_wls_ts / weights_wls_ts.max()

# ================================================================================
# プロット
# ================================================================================
print("\n" + "="*80)
print("Generating plots")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variance model selection
ax1 = axes[0, 0]
ax1.plot(m_array, fid_stds, 'o', markersize=8, color='black',
         label='Empirical σ(m)', alpha=0.7, zorder=5)
ax1.plot(m_array, sigma_model, '-', linewidth=3, color='red',
         label=f'Selected: {best_model_name}', alpha=0.8)

# Show other models in lighter colors
for name, model_info in models.items():
    if name != best_model_name:
        ax1.plot(m_array, model_info['sigma'], '--', linewidth=1.5,
                alpha=0.4, label=f'{name} (R²={model_info["r2"]:.3f})')

ax1.set_xlabel('Bounce Length (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Standard Deviation σ(m)', fontsize=12, fontweight='bold')
ax1.set_title('Stage 1: Variance Model Selection', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Weights comparison
ax2 = axes[0, 1]
x_pos = np.arange(len(m_array))
width = 0.25

ax2.bar(x_pos - width, weights_ols_norm, width, color='blue', alpha=0.7,
        label='OLS (equal)', edgecolor='black', linewidth=0.5)
ax2.bar(x_pos, weights_wls_se_norm, width, color='green', alpha=0.7,
        label='WLS-SE (empirical)', edgecolor='black', linewidth=0.5)
ax2.bar(x_pos + width, weights_wls_ts_norm, width, color='red', alpha=0.7,
        label='WLS-Two-stage (model)', edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Bounce Length (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Normalized Weight', fontsize=12, fontweight='bold')
ax2.set_title('Stage 2: Weight Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos[::3])
ax2.set_xticklabels(m_array[::3])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Uncertainty comparison
ax3 = axes[1, 0]
methods = ['OLS', 'WLS-SE', 'WLS-Two-stage']
uncertainties = [rel_unc_ols, rel_unc_wls_se, rel_unc_wls_ts]
colors = ['blue', 'green', 'red']

bars = ax3.bar(methods, uncertainties, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Relative Uncertainty (%)', fontsize=12, fontweight='bold')
ax3.set_title('Estimation Precision Comparison', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, unc in zip(bars, uncertainties):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{unc:.3f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Fit comparison
ax4 = axes[1, 1]
y_ols = exponential_decay(m_array, *popt_ols)
y_wls_se = exponential_decay(m_array, *popt_wls_se)
y_wls_ts = exponential_decay(m_array, *popt_wls_ts)

ax4.errorbar(m_array, fid_means, yerr=fid_stds/np.sqrt(samples_per_m),
            fmt='o', markersize=6, color='black', alpha=0.5, label='Data', capsize=3)
ax4.plot(m_array, y_ols, '-', linewidth=2, color='blue', alpha=0.7, label='OLS')
ax4.plot(m_array, y_wls_se, '-', linewidth=2, color='green', alpha=0.7, label='WLS-SE')
ax4.plot(m_array, y_wls_ts, '--', linewidth=2.5, color='red', alpha=0.8, label='WLS-Two-stage')

ax4.set_xlabel('Bounce Length (m)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
ax4.set_title('Decay Fits Comparison', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wls_two_stage.pdf', bbox_inches='tight', dpi=300)
print("\nPlot saved to: wls_two_stage.pdf")

# ================================================================================
# 詳細分析
# ================================================================================
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print(f"""
Two-stage WLS vs Direct Methods:

1. Variance Model Selection (Stage 1):
   Selected: {best_model_name} (R² = {best_model['r2']:.4f})
   Formula: {best_model['formula']}

2. Performance Comparison (Stage 2):

   OLS:            f = {f_ols:.6f} ± {f_err_ols:.6f} ({rel_unc_ols:.3f}%)
   WLS-SE:         f = {f_wls_se:.6f} ± {f_err_wls_se:.6f} ({rel_unc_wls_se:.3f}%)
   WLS-Two-stage:  f = {f_wls_ts:.6f} ± {f_err_wls_ts:.6f} ({rel_unc_wls_ts:.3f}%)

3. Key Insights:

   a) Model smoothing effect:
      WLS-Two-stage uses smoothed σ_model(m) instead of raw σ_empirical(m)
      → Less sensitive to sample noise in variance estimates

   b) Weight distribution:
      - WLS-SE: Directly from data (may have noise)
      - WLS-Two-stage: From smooth model (more stable)

   c) Performance vs WLS-SE:
      Difference: {abs(rel_unc_wls_ts - rel_unc_wls_se):.3f} percentage points
      {'WLS-Two-stage is better' if rel_unc_wls_ts < rel_unc_wls_se else
       'WLS-SE is better' if rel_unc_wls_se < rel_unc_wls_ts else
       'About the same'}

4. Advantages of Two-stage WLS:
   ✓ Data-driven model selection (not pre-fixed)
   ✓ Smooth weights (less noisy than direct empirical)
   ✓ Still respects empirical variance structure
   ✓ More robust with limited samples per m

5. When to use each method:
   - OLS: Homoscedastic data, simplicity
   - WLS-SE: Sufficient samples per m (n >> 10), trust empirical variance
   - WLS-Two-stage: Limited samples, want robustness, automatic adaptation
""")

print("="*80)
print("Analysis complete")
print("="*80)
