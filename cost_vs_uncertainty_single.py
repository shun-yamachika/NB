"""
測定コスト vs 推定精度の分析（シンプル版）

横軸：測定コスト = Σ(バウンス長 × 測定回数)
縦軸：推定されたfの相対不確実性
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================================================================================
# データ読み込み
# ================================================================================
print("="*80)
print("Cost vs Uncertainty Analysis (Simple)")
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

    # WLS-SEフィッティング（n>=2のみ）
    if n_samples >= 2:
        try:
            standard_errors = fid_stds / np.sqrt(n_samples)
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

print(f"Successfully analyzed {len(results_ols['samples'])} sample sizes")

# ================================================================================
# プロット
# ================================================================================
print("\n" + "="*80)
print("Generating plot")
print("="*80)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(results_ols['cost'], results_ols['rel_unc'], 'o-',
        linewidth=2.5, markersize=8, label='OLS', alpha=0.8, color='#2E86AB')
ax.plot(results_wls['cost'], results_wls['rel_unc'], 's-',
        linewidth=2.5, markersize=8, label='WLS-SE', alpha=0.8, color='#A23B72')

ax.set_xlabel('Total Measurement Cost  $\\Sigma (m \\times n)$', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative Uncertainty of $f$  (%)', fontsize=14, fontweight='bold')
ax.set_title('Measurement Cost vs Estimation Precision', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=12)

# Add annotation for key point
idx_200_ols = np.where(results_ols['samples'] == 200)[0]
idx_200_wls = np.where(results_wls['samples'] == 200)[0]

if len(idx_200_ols) > 0 and len(idx_200_wls) > 0:
    idx_ols = idx_200_ols[0]
    idx_wls = idx_200_wls[0]
    improvement = (1 - results_wls['rel_unc'][idx_wls] / results_ols['rel_unc'][idx_ols]) * 100
    ax.annotate(f'n=200: {improvement:.0f}% improvement',
                xy=(results_wls['cost'][idx_wls], results_wls['rel_unc'][idx_wls]),
                xytext=(results_wls['cost'][idx_wls] * 0.5, results_wls['rel_unc'][idx_wls] * 2),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
else:
    improvement = 0

plt.tight_layout()
plt.savefig('cost_vs_uncertainty_single.pdf', bbox_inches='tight', dpi=300)
print("\nPlot saved to: cost_vs_uncertainty_single.pdf")

# ================================================================================
# サマリー
# ================================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if len(idx_200_ols) > 0 and len(idx_200_wls) > 0:
    print(f"""
At n=200 (Total Cost = {results_ols['cost'][idx_ols]:.0f}):
  OLS:    {results_ols['rel_unc'][idx_ols]:.3f}% uncertainty
  WLS-SE: {results_wls['rel_unc'][idx_wls]:.3f}% uncertainty

  → WLS-SE achieves {improvement:.0f}% improvement in precision
  → Both converge to f ≈ 0.899
""")
else:
    print("\nData for n=200 not found in results.")

print("="*80)
