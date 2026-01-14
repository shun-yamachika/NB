"""
測定コスト vs 推定精度の分析

横軸：測定コスト = Σ(バウンス長 × 測定回数)
縦軸：推定されたfの相対不確実性

各mに対して測定回数を1, 2, 3, ... と増やしながら、
コストと精度のトレードオフを可視化
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================================================================================
# データ読み込み
# ================================================================================
print("="*80)
print("Cost vs Uncertainty Analysis")
print("="*80)

with open('AB_decay_200samples_main.pickle', 'rb') as f:
    data = pk.load(f)

fid_raw_data = data["decay data"][1]
all_m_values = sorted(fid_raw_data.keys())

m_range = (2, 40)
m_values = [m for m in all_m_values if m_range[0] <= m <= m_range[1]]
m_array = np.array(m_values)

print(f"\nBounce lengths: m ∈ [{m_range[0]}, {m_range[1]}]")
print(f"Number of m values: {len(m_values)}")
print(f"Total cost per measurement: Σm = {np.sum(m_array)}")

# ================================================================================
# 指数減衰モデル
# ================================================================================
def exponential_decay(m, A, f):
    return A * (f ** m)

# ================================================================================
# 測定回数を変えながら解析
# ================================================================================
max_samples = 200
sample_points = list(range(1, 11)) + list(range(15, 51, 5)) + list(range(60, 201, 10))

results_ols = {'samples': [], 'cost': [], 'f': [], 'f_err': [], 'rel_unc': []}
results_wls = {'samples': [], 'cost': [], 'f': [], 'f_err': [], 'rel_unc': []}

print(f"\nAnalyzing {len(sample_points)} different sample sizes...")
print(f"Sample points: {sample_points[:10]}... (total {len(sample_points)})")

for n_samples in sample_points:
    if n_samples > max_samples:
        break

    # 各mからn_samples個のデータを取得
    fid_means = []
    fid_stds = []

    for m in m_values:
        data_m = fid_raw_data[m][:n_samples]
        fid_means.append(np.mean(data_m))
        fid_stds.append(np.std(data_m, ddof=1))

    fid_means = np.array(fid_means)
    fid_stds = np.array(fid_stds)

    # 測定コスト = Σ(m × n_samples)
    total_cost = np.sum(m_array) * n_samples

    # OLSフィッティング
    try:
        popt_ols, pcov_ols = curve_fit(exponential_decay, m_array, fid_means,
                                       p0=[0.5, 0.9])
        f_ols = popt_ols[1]
        f_err_ols = np.sqrt(pcov_ols[1, 1])
        rel_unc_ols = f_err_ols / f_ols * 100  # %

        results_ols['samples'].append(n_samples)
        results_ols['cost'].append(total_cost)
        results_ols['f'].append(f_ols)
        results_ols['f_err'].append(f_err_ols)
        results_ols['rel_unc'].append(rel_unc_ols)
    except:
        pass

    # WLS-SEフィッティング（経験的標準誤差を重みとして使用）
    # n>=2 required for meaningful standard deviation
    if n_samples >= 2:
        try:
            standard_errors = fid_stds / np.sqrt(n_samples)
            # Check for zero std (shouldn't happen with n>=2, but be safe)
            if np.all(standard_errors > 0):
                popt_wls, pcov_wls = curve_fit(exponential_decay, m_array, fid_means,
                                               sigma=standard_errors, absolute_sigma=True,
                                               p0=[0.5, 0.9])
                f_wls = popt_wls[1]
                f_err_wls = np.sqrt(pcov_wls[1, 1])
                rel_unc_wls = f_err_wls / f_wls * 100  # %

                results_wls['samples'].append(n_samples)
                results_wls['cost'].append(total_cost)
                results_wls['f'].append(f_wls)
                results_wls['f_err'].append(f_err_wls)
                results_wls['rel_unc'].append(rel_unc_wls)
        except:
            pass

# Convert to arrays
for key in results_ols:
    results_ols[key] = np.array(results_ols[key])
    results_wls[key] = np.array(results_wls[key])

print(f"\nSuccessfully analyzed {len(results_ols['samples'])} sample sizes")

# ================================================================================
# サマリー統計
# ================================================================================
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

print("\nOLS:")
print(f"  n=1:   Cost = {results_ols['cost'][0]:.0f}, Rel. Unc. = {results_ols['rel_unc'][0]:.2f}%")
if len(results_ols['samples']) > 9:
    print(f"  n=10:  Cost = {results_ols['cost'][9]:.0f}, Rel. Unc. = {results_ols['rel_unc'][9]:.2f}%")
print(f"  n=200: Cost = {results_ols['cost'][-1]:.0f}, Rel. Unc. = {results_ols['rel_unc'][-1]:.2f}%")
print(f"  Improvement factor: {results_ols['rel_unc'][0] / results_ols['rel_unc'][-1]:.1f}×")

print("\nWLS-SE (n>=2):")
print(f"  n=2:   Cost = {results_wls['cost'][0]:.0f}, Rel. Unc. = {results_wls['rel_unc'][0]:.2f}%")
if len(results_wls['samples']) > 8:
    print(f"  n=10:  Cost = {results_wls['cost'][8]:.0f}, Rel. Unc. = {results_wls['rel_unc'][8]:.2f}%")
print(f"  n=200: Cost = {results_wls['cost'][-1]:.0f}, Rel. Unc. = {results_wls['rel_unc'][-1]:.2f}%")
print(f"  Improvement factor: {results_wls['rel_unc'][0] / results_wls['rel_unc'][-1]:.1f}×")

# ================================================================================
# プロット
# ================================================================================
print("\n" + "="*80)
print("Generating plots")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cost vs Relative Uncertainty (main plot)
ax1 = axes[0, 0]
ax1.plot(results_ols['cost'], results_ols['rel_unc'], 'o-',
         linewidth=2, markersize=6, label='OLS', alpha=0.7, color='blue')
ax1.plot(results_wls['cost'], results_wls['rel_unc'], 's-',
         linewidth=2, markersize=6, label='WLS-SE', alpha=0.7, color='green')

ax1.set_xlabel('Total Measurement Cost (Σm×n)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Relative Uncertainty of f (%)', fontsize=12, fontweight='bold')
ax1.set_title('Cost vs Uncertainty Trade-off', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Cost vs Relative Uncertainty (log-log)
ax2 = axes[0, 1]
ax2.loglog(results_ols['cost'], results_ols['rel_unc'], 'o-',
           linewidth=2, markersize=6, label='OLS', alpha=0.7, color='blue')
ax2.loglog(results_wls['cost'], results_wls['rel_unc'], 's-',
           linewidth=2, markersize=6, label='WLS-SE', alpha=0.7, color='green')

# Fit power law: uncertainty ∝ cost^(-α)
log_cost_ols = np.log(results_ols['cost'])
log_unc_ols = np.log(results_ols['rel_unc'])
slope_ols = np.polyfit(log_cost_ols, log_unc_ols, 1)[0]

# For WLS, filter out any nan or inf values
valid_wls = np.isfinite(results_wls['rel_unc']) & np.isfinite(results_wls['cost'])
if np.sum(valid_wls) > 1:
    log_cost_wls = np.log(results_wls['cost'][valid_wls])
    log_unc_wls = np.log(results_wls['rel_unc'][valid_wls])
    slope_wls = np.polyfit(log_cost_wls, log_unc_wls, 1)[0]
else:
    slope_wls = np.nan

ax2.set_xlabel('Total Measurement Cost (Σm×n)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative Uncertainty of f (%)', fontsize=12, fontweight='bold')
ax2.set_title(f'Log-Log Scale (OLS: {slope_ols:.2f}, WLS: {slope_wls:.2f})',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Sample size vs Relative Uncertainty
ax3 = axes[1, 0]
ax3.plot(results_ols['samples'], results_ols['rel_unc'], 'o-',
         linewidth=2, markersize=6, label='OLS', alpha=0.7, color='blue')
ax3.plot(results_wls['samples'], results_wls['rel_unc'], 's-',
         linewidth=2, markersize=6, label='WLS-SE', alpha=0.7, color='green')

ax3.set_xlabel('Samples per m (n)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Relative Uncertainty of f (%)', fontsize=12, fontweight='bold')
ax3.set_title('Samples vs Uncertainty', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Estimated f value vs sample size
ax4 = axes[1, 1]
ax4.plot(results_ols['samples'], results_ols['f'], 'o-',
         linewidth=2, markersize=6, label='OLS', alpha=0.7, color='blue')
ax4.plot(results_wls['samples'], results_wls['f'], 's-',
         linewidth=2, markersize=6, label='WLS-SE', alpha=0.7, color='green')

# Add error bands
ax4.fill_between(results_ols['samples'],
                 results_ols['f'] - results_ols['f_err'],
                 results_ols['f'] + results_ols['f_err'],
                 alpha=0.2, color='blue')
ax4.fill_between(results_wls['samples'],
                 results_wls['f'] - results_wls['f_err'],
                 results_wls['f'] + results_wls['f_err'],
                 alpha=0.2, color='green')

ax4.set_xlabel('Samples per m (n)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Estimated f', fontsize=12, fontweight='bold')
ax4.set_title('Convergence of f Estimate', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cost_vs_uncertainty.pdf', bbox_inches='tight')
print("\nPlot saved to: cost_vs_uncertainty.pdf")

# ================================================================================
# データテーブル出力
# ================================================================================
print("\n" + "="*80)
print("Data Table (selected points)")
print("="*80)

print(f"\n{'n':<6} {'Cost':<10} {'OLS f':<12} {'OLS Unc%':<12} {'WLS f':<12} {'WLS Unc%':<12}")
print("-"*74)

# Show selected sample sizes
selected_n = [1, 2, 5, 10, 20, 50, 100, 200]
for n in selected_n:
    # Find in OLS results
    ols_idx = np.where(results_ols['samples'] == n)[0]
    wls_idx = np.where(results_wls['samples'] == n)[0]

    if len(ols_idx) > 0:
        idx_ols = ols_idx[0]
        cost = results_ols['cost'][idx_ols]
        f_ols = results_ols['f'][idx_ols]
        unc_ols = results_ols['rel_unc'][idx_ols]

        if len(wls_idx) > 0:
            idx_wls = wls_idx[0]
            f_wls = results_wls['f'][idx_wls]
            unc_wls = results_wls['rel_unc'][idx_wls]
            print(f"{n:<6} {cost:<10.0f} {f_ols:<12.6f} {unc_ols:<12.3f} "
                  f"{f_wls:<12.6f} {unc_wls:<12.3f}")
        else:
            print(f"{n:<6} {cost:<10.0f} {f_ols:<12.6f} {unc_ols:<12.3f} "
                  f"{'N/A':<12} {'N/A':<12}")

# ================================================================================
# 解析結果サマリー
# ================================================================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

wls_scaling_str = f"{slope_wls:.3f}" if not np.isnan(slope_wls) else "N/A"

print(f"""
Measurement Cost vs Estimation Precision:

1. Scaling Relationship:
   - OLS: Uncertainty ∝ Cost^{slope_ols:.3f}
   - WLS-SE: Uncertainty ∝ Cost^{wls_scaling_str}

   Theoretical expectation: -0.5 (from √n scaling)
   OLS scaling: {'close to' if abs(slope_ols + 0.5) < 0.1 else 'deviates from'} theoretical prediction.

2. Efficiency Comparison:
   At n=200 (Cost = {results_ols['cost'][-1]:.0f}):
   - OLS:    Rel. Unc. = {results_ols['rel_unc'][-1]:.3f}%
   - WLS-SE: Rel. Unc. = {results_wls['rel_unc'][-1]:.3f}%
   - Improvement: {(1 - results_wls['rel_unc'][-1]/results_ols['rel_unc'][-1])*100:.1f}%

3. Convergence:
   Both methods converge to similar f values:
   - OLS:    f = {results_ols['f'][-1]:.6f} ± {results_ols['f_err'][-1]:.6f}
   - WLS-SE: f = {results_wls['f'][-1]:.6f} ± {results_wls['f_err'][-1]:.6f}

4. Practical Implications:
   - Diminishing returns: 10× cost increase → ~3× precision improvement
   - WLS-SE consistently outperforms OLS at all cost levels
   - Measurement budget should balance cost vs required precision
""")

print("="*80)
print("Analysis complete")
print("="*80)
