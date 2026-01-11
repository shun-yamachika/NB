"""
WLS-MN (Weighted Least Squares - Model-based) Implementation

モデルベースの重み付き最小二乗法でRBデータをフィッティング
σ(m) = a + b*m のモデルを使用して重み w(m,n) = n/(a+b*m)² を計算

参考: weighted_least_squares_summary.org
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ================================================================================
# モデルパラメータ（事前推定値）
# ================================================================================
# workspace2の実験結果から:
# std(m) = 0.030641 + 0.000982 * m (R² = 0.6012)

SIGMA_MODEL_A = 0.030641
SIGMA_MODEL_B = 0.000982

print("="*80)
print("WLS-MN Implementation: Model-based Weighted Least Squares")
print("="*80)
print(f"\nPre-estimated model parameters:")
print(f"  σ(m) = {SIGMA_MODEL_A:.6f} + {SIGMA_MODEL_B:.6f} * m")
print(f"  (Based on previous experimental data)")

# ================================================================================
# データ読み込み
# ================================================================================
print("\n" + "="*80)
print("Loading data")
print("="*80)

with open('AB_decay_200samples_main.pickle', 'rb') as f:
    data = pk.load(f)

fid_raw_data = data["decay data"][1]  # Fidelity data
all_m_values = sorted(fid_raw_data.keys())

print(f"\nData loaded:")
print(f"  Bounce lengths (m): {all_m_values[0]} to {all_m_values[-1]}")
print(f"  Number of m values: {len(all_m_values)}")
print(f"  Samples per m: 200")

# ================================================================================
# 実験設定
# ================================================================================
# m範囲とサンプル数を設定
m_range = (2, 20)
samples_per_m = 40  # 各mで使用するサンプル数

m_values = [m for m in all_m_values if m_range[0] <= m <= m_range[1]]
n_trials = samples_per_m

print(f"\nExperiment configuration:")
print(f"  m range: {m_range[0]} to {m_range[1]}")
print(f"  Samples per m: {samples_per_m}")
print(f"  Total data points: {len(m_values)}")

# ================================================================================
# データ準備
# ================================================================================
fid_means = []
fid_stds = []

for m in m_values:
    data_m = fid_raw_data[m][:samples_per_m]
    fid_means.append(np.mean(data_m))
    fid_stds.append(np.std(data_m, ddof=1))

fid_means = np.array(fid_means)
fid_stds = np.array(fid_stds)
m_array = np.array(m_values)

print(f"\nData statistics:")
print(f"  Mean fidelity range: {fid_means.min():.6f} to {fid_means.max():.6f}")
print(f"  Std range: {fid_stds.min():.6f} to {fid_stds.max():.6f}")

# ================================================================================
# モデル定義
# ================================================================================
def exponential_decay(m, A, f):
    """指数減衰モデル: y = A * f^m"""
    return A * (f ** m)

# ================================================================================
# Method 1: OLS (Ordinary Least Squares)
# ================================================================================
print("\n" + "="*80)
print("Method 1: OLS (Ordinary Least Squares)")
print("="*80)

# 重み付けなしでフィッティング
popt_ols, pcov_ols = curve_fit(
    exponential_decay,
    m_array,
    fid_means,
    p0=[0.95, 0.95],
    maxfev=10000
)

A_ols, f_ols = popt_ols
A_ols_err = np.sqrt(pcov_ols[0, 0])
f_ols_err = np.sqrt(pcov_ols[1, 1])

# 相対不確実性
rel_unc_ols = f_ols_err / f_ols

print(f"\nFitting results:")
print(f"  A = {A_ols:.6f} ± {A_ols_err:.6f}")
print(f"  f = {f_ols:.6f} ± {f_ols_err:.6f}")
print(f"  Relative uncertainty (f): {rel_unc_ols:.6f} ({rel_unc_ols*100:.4f}%)")

# ================================================================================
# Method 2: WLS-MN (Weighted Least Squares - Model-based)
# ================================================================================
print("\n" + "="*80)
print("Method 2: WLS-MN (Weighted Least Squares - Model-based)")
print("="*80)

# モデルベースの標準偏差を計算
sigma_model = SIGMA_MODEL_A + SIGMA_MODEL_B * m_array

# 試行回数で補正
sigma_mn = sigma_model / np.sqrt(n_trials)

# 重み: w(m,n) = n / σ²(m)
weights_mn = n_trials / (sigma_model ** 2)

print(f"\nWeights calculation:")
print(f"  σ_model(m=2)  = {sigma_model[m_array==2][0]:.6f}")
print(f"  σ_model(m=10) = {sigma_model[m_array==10][0]:.6f}")
print(f"  σ_model(m=20) = {sigma_model[m_array==20][0]:.6f}")
print(f"\n  Weight ratio: w(m=2)/w(m=20) = {weights_mn[m_array==2][0]/weights_mn[m_array==20][0]:.2f}")

# WLS-MNでフィッティング
popt_wls_mn, pcov_wls_mn = curve_fit(
    exponential_decay,
    m_array,
    fid_means,
    p0=[0.95, 0.95],
    sigma=sigma_mn,
    absolute_sigma=True,
    maxfev=10000
)

A_wls_mn, f_wls_mn = popt_wls_mn
A_wls_mn_err = np.sqrt(pcov_wls_mn[0, 0])
f_wls_mn_err = np.sqrt(pcov_wls_mn[1, 1])

# 相対不確実性
rel_unc_wls_mn = f_wls_mn_err / f_wls_mn

print(f"\nFitting results:")
print(f"  A = {A_wls_mn:.6f} ± {A_wls_mn_err:.6f}")
print(f"  f = {f_wls_mn:.6f} ± {f_wls_mn_err:.6f}")
print(f"  Relative uncertainty (f): {rel_unc_wls_mn:.6f} ({rel_unc_wls_mn*100:.4f}%)")

# ================================================================================
# 結果の比較
# ================================================================================
print("\n" + "="*80)
print("Comparison: OLS vs WLS-MN")
print("="*80)

improvement = (rel_unc_ols - rel_unc_wls_mn) / rel_unc_ols * 100

print(f"\n{'Method':<20} {'f':<15} {'Rel. Uncertainty':<20} {'Improvement'}")
print("-"*80)
print(f"{'OLS':<20} {f_ols:.6f}      {rel_unc_ols:.6f} ({rel_unc_ols*100:.4f}%)  baseline")
print(f"{'WLS-MN':<20} {f_wls_mn:.6f}      {rel_unc_wls_mn:.6f} ({rel_unc_wls_mn*100:.4f}%)  {improvement:+.2f}%")

if improvement > 0:
    print(f"\n✅ WLS-MN achieved {improvement:.2f}% improvement over OLS")
else:
    print(f"\n❌ WLS-MN performed {-improvement:.2f}% worse than OLS")

# ================================================================================
# 結果を保存
# ================================================================================
results = {
    'ols': {
        'A': A_ols,
        'f': f_ols,
        'A_err': A_ols_err,
        'f_err': f_ols_err,
        'rel_uncertainty': rel_unc_ols,
        'covariance': pcov_ols
    },
    'wls_mn': {
        'A': A_wls_mn,
        'f': f_wls_mn,
        'A_err': A_wls_mn_err,
        'f_err': f_wls_mn_err,
        'rel_uncertainty': rel_unc_wls_mn,
        'covariance': pcov_wls_mn,
        'sigma_model_a': SIGMA_MODEL_A,
        'sigma_model_b': SIGMA_MODEL_B,
        'weights': weights_mn
    },
    'data': {
        'm_values': m_values,
        'fid_means': fid_means,
        'fid_stds': fid_stds,
        'samples_per_m': samples_per_m
    }
}

with open('wls_mn_results.pickle', 'wb') as f:
    pk.dump(results, f)

print(f"\n" + "="*80)
print("Results saved to: wls_mn_results.pickle")
print("="*80)

# ================================================================================
# 可視化
# ================================================================================
print("\n" + "="*80)
print("Generating plots")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: フィッティング結果の比較
ax1 = axes[0, 0]
m_fine = np.linspace(m_array.min(), m_array.max(), 100)

ax1.errorbar(m_array, fid_means, yerr=fid_stds/np.sqrt(n_trials),
             fmt='o', color='black', label='Data', alpha=0.6, markersize=5)
ax1.plot(m_fine, exponential_decay(m_fine, A_ols, f_ols),
         'b--', linewidth=2, label=f'OLS: f={f_ols:.6f}±{f_ols_err:.6f}')
ax1.plot(m_fine, exponential_decay(m_fine, A_wls_mn, f_wls_mn),
         'r-', linewidth=2, label=f'WLS-MN: f={f_wls_mn:.6f}±{f_wls_mn_err:.6f}')

ax1.set_xlabel('Bounce Length (m)', fontsize=12)
ax1.set_ylabel('Fidelity', fontsize=12)
ax1.set_title('Fitting Comparison: OLS vs WLS-MN', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: 残差プロット
ax2 = axes[0, 1]
residuals_ols = fid_means - exponential_decay(m_array, A_ols, f_ols)
residuals_wls_mn = fid_means - exponential_decay(m_array, A_wls_mn, f_wls_mn)

ax2.scatter(m_array, residuals_ols, color='blue', alpha=0.6, s=50, label='OLS')
ax2.scatter(m_array, residuals_wls_mn, color='red', alpha=0.6, s=50, label='WLS-MN')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)

ax2.set_xlabel('Bounce Length (m)', fontsize=12)
ax2.set_ylabel('Residuals', fontsize=12)
ax2.set_title('Residuals Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: モデル標準偏差 vs 実測標準偏差
ax3 = axes[1, 0]
ax3.scatter(m_array, fid_stds, color='black', s=50, alpha=0.6, label='Empirical σ(m)')
ax3.plot(m_array, sigma_model, 'r-', linewidth=2,
         label=f'Model: σ(m) = {SIGMA_MODEL_A:.4f} + {SIGMA_MODEL_B:.6f}*m')

ax3.set_xlabel('Bounce Length (m)', fontsize=12)
ax3.set_ylabel('Standard Deviation', fontsize=12)
ax3.set_title('Model vs Empirical Standard Deviation', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: 重みの分布
ax4 = axes[1, 1]
ax4.bar(m_array, weights_mn / weights_mn.max(), color='red', alpha=0.7, width=0.8)
ax4.set_xlabel('Bounce Length (m)', fontsize=12)
ax4.set_ylabel('Normalized Weight', fontsize=12)
ax4.set_title('WLS-MN Weights Distribution', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('wls_mn_analysis.pdf', bbox_inches='tight')
print("\nPlot saved to: wls_mn_analysis.pdf")

# ================================================================================
# サマリー
# ================================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
WLS-MN Implementation Results:

Model Parameters:
  σ(m) = {SIGMA_MODEL_A:.6f} + {SIGMA_MODEL_B:.6f} * m

Data Configuration:
  m range: {m_range[0]} to {m_range[1]} ({len(m_values)} points)
  Samples per m: {samples_per_m}

Fitting Results:
  OLS:
    f = {f_ols:.6f} ± {f_ols_err:.6f}
    Relative uncertainty: {rel_unc_ols*100:.4f}%

  WLS-MN:
    f = {f_wls_mn:.6f} ± {f_wls_mn_err:.6f}
    Relative uncertainty: {rel_unc_wls_mn*100:.4f}%

Performance:
  Improvement: {improvement:+.2f}%
  {'✅ WLS-MN is better' if improvement > 0 else '❌ OLS is better'}

Theory:
  WLS-MN weights data points by their expected noise level:
  - Higher weights for low-noise points (small m)
  - Lower weights for high-noise points (large m)
  - Weights scale as w(m,n) = n / (a + b*m)²

Expected benefit:
  ~50% improvement (based on workspace2 results)
  Actual: {improvement:+.2f}%
""")

print("="*80)
print("Analysis complete")
print("="*80)
