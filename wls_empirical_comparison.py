"""
WLS-Empirical vs WLS-MN Comparison

事前データを使わない重み付け（WLS-Empirical）と
事前データを使う重み付け（WLS-MN）の比較

WLS-Empirical: 同じデータセットから計算した標準偏差で重み付け
WLS-MN: 事前推定したモデル σ(m) = a + b*m で重み付け
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
print("WLS-Empirical vs WLS-MN Comparison")
print("="*80)

with open('AB_decay_200samples_main.pickle', 'rb') as f:
    data = pk.load(f)

fid_raw_data = data["decay data"][1]  # Fidelity data
all_m_values = sorted(fid_raw_data.keys())

# ================================================================================
# 実験設定
# ================================================================================
m_range = (2, 20)
samples_per_m = 40

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
rel_unc_ols = f_ols_err / f_ols

print(f"\nFitting results:")
print(f"  A = {A_ols:.6f} ± {A_ols_err:.6f}")
print(f"  f = {f_ols:.6f} ± {f_ols_err:.6f}")
print(f"  Relative uncertainty (f): {rel_unc_ols:.6f} ({rel_unc_ols*100:.4f}%)")

# ================================================================================
# Method 2: WLS-Empirical (同じデータから計算した標準偏差で重み付け)
# ================================================================================
print("\n" + "="*80)
print("Method 2: WLS-Empirical (Empirical standard deviation)")
print("="*80)

# 実測の標準偏差を使用
sigma_empirical = fid_stds / np.sqrt(n_trials)

# 重み: w(m) = 1 / σ²(m)
# (curve_fitはsigma=σを渡すと、内部でw = 1/σ²として使う)
weights_empirical = 1.0 / (fid_stds ** 2)

print(f"\nWeights calculation:")
print(f"  σ_empirical(m=2)  = {fid_stds[m_array==2][0]:.6f}")
print(f"  σ_empirical(m=10) = {fid_stds[m_array==10][0]:.6f}")
print(f"  σ_empirical(m=20) = {fid_stds[m_array==20][0]:.6f}")
print(f"\n  Weight ratio: w(m=2)/w(m=20) = {weights_empirical[m_array==2][0]/weights_empirical[m_array==20][0]:.2f}")

# WLS-Empiricalでフィッティング
popt_wls_emp, pcov_wls_emp = curve_fit(
    exponential_decay,
    m_array,
    fid_means,
    p0=[0.95, 0.95],
    sigma=sigma_empirical,
    absolute_sigma=True,
    maxfev=10000
)

A_wls_emp, f_wls_emp = popt_wls_emp
A_wls_emp_err = np.sqrt(pcov_wls_emp[0, 0])
f_wls_emp_err = np.sqrt(pcov_wls_emp[1, 1])
rel_unc_wls_emp = f_wls_emp_err / f_wls_emp

print(f"\nFitting results:")
print(f"  A = {A_wls_emp:.6f} ± {A_wls_emp_err:.6f}")
print(f"  f = {f_wls_emp:.6f} ± {f_wls_emp_err:.6f}")
print(f"  Relative uncertainty (f): {rel_unc_wls_emp:.6f} ({rel_unc_wls_emp*100:.4f}%)")

# ================================================================================
# Method 3: WLS-MN (Model-based、事前データ使用)
# ================================================================================
print("\n" + "="*80)
print("Method 3: WLS-MN (Model-based with prior data)")
print("="*80)

# 事前推定したモデルパラメータ (workspace2から)
SIGMA_MODEL_A = 0.030641
SIGMA_MODEL_B = 0.000982

print(f"\nPre-estimated model parameters (from prior data):")
print(f"  σ(m) = {SIGMA_MODEL_A:.6f} + {SIGMA_MODEL_B:.6f} * m")

# モデルベースの標準偏差を計算
sigma_model = SIGMA_MODEL_A + SIGMA_MODEL_B * m_array
sigma_mn = sigma_model / np.sqrt(n_trials)

# 重み
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
rel_unc_wls_mn = f_wls_mn_err / f_wls_mn

print(f"\nFitting results:")
print(f"  A = {A_wls_mn:.6f} ± {A_wls_mn_err:.6f}")
print(f"  f = {f_wls_mn:.6f} ± {f_wls_mn_err:.6f}")
print(f"  Relative uncertainty (f): {rel_unc_wls_mn:.6f} ({rel_unc_wls_mn*100:.4f}%)")

# ================================================================================
# 結果の比較
# ================================================================================
print("\n" + "="*80)
print("Comparison: OLS vs WLS-Empirical vs WLS-MN")
print("="*80)

improvement_emp = (rel_unc_ols - rel_unc_wls_emp) / rel_unc_ols * 100
improvement_mn = (rel_unc_ols - rel_unc_wls_mn) / rel_unc_ols * 100

print(f"\n{'Method':<20} {'f':<15} {'Rel. Unc.':<15} {'Improvement'}")
print("-"*80)
print(f"{'OLS':<20} {f_ols:.6f}      {rel_unc_ols:.6f} ({rel_unc_ols*100:.4f}%)  baseline")
print(f"{'WLS-Empirical':<20} {f_wls_emp:.6f}      {rel_unc_wls_emp:.6f} ({rel_unc_wls_emp*100:.4f}%)  {improvement_emp:+.2f}%")
print(f"{'WLS-MN':<20} {f_wls_mn:.6f}      {rel_unc_wls_mn:.6f} ({rel_unc_wls_mn*100:.4f}%)  {improvement_mn:+.2f}%")

print("\n" + "="*80)
print("Key Observations:")
print("="*80)
print(f"\n1. WLS-Empirical vs OLS:")
print(f"   Improvement: {improvement_emp:+.2f}%")
print(f"   {'✅ Better' if improvement_emp > 0 else '❌ Worse'}")

print(f"\n2. WLS-MN vs OLS:")
print(f"   Improvement: {improvement_mn:+.2f}%")
print(f"   {'✅ Better' if improvement_mn > 0 else '❌ Worse'}")

print(f"\n3. WLS-Empirical vs WLS-MN:")
diff = ((rel_unc_wls_mn - rel_unc_wls_emp) / rel_unc_wls_emp) * 100
print(f"   WLS-MN is {diff:+.2f}% {'worse' if diff > 0 else 'better'} than WLS-Empirical")

# ================================================================================
# 標準偏差のモデルフィッティング（現在のデータから）
# ================================================================================
print("\n" + "="*80)
print("Standard Deviation Model from Current Data")
print("="*80)

# 現在のデータで σ(m) = a + b*m をフィッティング
slope, intercept, r_value, p_value, std_err = linregress(m_array, fid_stds)

print(f"\nLinear regression: σ(m) = a + b*m")
print(f"  a (intercept) = {intercept:.6f}")
print(f"  b (slope)     = {slope:.6f}")
print(f"  R²            = {r_value**2:.4f}")
print(f"  p-value       = {p_value:.6e}")

print(f"\nComparison with prior model:")
print(f"  Prior:   σ(m) = {SIGMA_MODEL_A:.6f} + {SIGMA_MODEL_B:.6f} * m  (R² from workspace2 = 0.6012)")
print(f"  Current: σ(m) = {intercept:.6f} + {slope:.6f} * m  (R² = {r_value**2:.4f})")

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
         'b--', linewidth=2, label=f'OLS: f={f_ols:.6f}±{f_ols_err:.6f}', alpha=0.7)
ax1.plot(m_fine, exponential_decay(m_fine, A_wls_emp, f_wls_emp),
         'g-', linewidth=2, label=f'WLS-Emp: f={f_wls_emp:.6f}±{f_wls_emp_err:.6f}', alpha=0.7)
ax1.plot(m_fine, exponential_decay(m_fine, A_wls_mn, f_wls_mn),
         'r-', linewidth=2, label=f'WLS-MN: f={f_wls_mn:.6f}±{f_wls_mn_err:.6f}', alpha=0.7)

ax1.set_xlabel('Bounce Length (m)', fontsize=12)
ax1.set_ylabel('Fidelity', fontsize=12)
ax1.set_title('Fitting Comparison: OLS vs WLS-Empirical vs WLS-MN', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: 相対不確実性の比較（棒グラフ）
ax2 = axes[0, 1]
methods = ['OLS', 'WLS-Empirical', 'WLS-MN']
uncertainties = [rel_unc_ols * 100, rel_unc_wls_emp * 100, rel_unc_wls_mn * 100]
colors = ['blue', 'green', 'red']

bars = ax2.bar(methods, uncertainties, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Relative Uncertainty (%)', fontsize=12)
ax2.set_title('Relative Uncertainty Comparison', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 値をバーの上に表示
for bar, unc in zip(bars, uncertainties):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{unc:.4f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: 標準偏差の比較（モデル vs 実測）
ax3 = axes[1, 0]
ax3.scatter(m_array, fid_stds, color='black', s=50, alpha=0.6, label='Empirical σ(m)', zorder=3)
ax3.plot(m_array, sigma_model, 'r-', linewidth=2,
         label=f'Prior model: σ(m) = {SIGMA_MODEL_A:.4f} + {SIGMA_MODEL_B:.6f}*m', alpha=0.7)
ax3.plot(m_array, intercept + slope * m_array, 'g--', linewidth=2,
         label=f'Current fit: σ(m) = {intercept:.4f} + {slope:.6f}*m (R²={r_value**2:.3f})', alpha=0.7)

ax3.set_xlabel('Bounce Length (m)', fontsize=12)
ax3.set_ylabel('Standard Deviation', fontsize=12)
ax3.set_title('Standard Deviation Models vs Empirical Data', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: 重みの比較
ax4 = axes[1, 1]
# 正規化された重み
weights_emp_norm = weights_empirical / weights_empirical.max()
weights_mn_norm = weights_mn / weights_mn.max()

width = 0.35
x_pos = np.arange(len(m_array))

ax4.bar(x_pos - width/2, weights_emp_norm, width, color='green', alpha=0.7, label='WLS-Empirical')
ax4.bar(x_pos + width/2, weights_mn_norm, width, color='red', alpha=0.7, label='WLS-MN')

ax4.set_xlabel('Bounce Length (m)', fontsize=12)
ax4.set_ylabel('Normalized Weight', fontsize=12)
ax4.set_title('Weights Distribution Comparison', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos[::2])  # Show every other m value
ax4.set_xticklabels(m_array[::2])
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('wls_empirical_comparison.pdf', bbox_inches='tight')
print("\nPlot saved to: wls_empirical_comparison.pdf")

# ================================================================================
# 結果を保存
# ================================================================================
results = {
    'ols': {
        'A': A_ols,
        'f': f_ols,
        'A_err': A_ols_err,
        'f_err': f_ols_err,
        'rel_uncertainty': rel_unc_ols
    },
    'wls_empirical': {
        'A': A_wls_emp,
        'f': f_wls_emp,
        'A_err': A_wls_emp_err,
        'f_err': f_wls_emp_err,
        'rel_uncertainty': rel_unc_wls_emp,
        'weights': weights_empirical,
        'sigma': fid_stds
    },
    'wls_mn': {
        'A': A_wls_mn,
        'f': f_wls_mn,
        'A_err': A_wls_mn_err,
        'f_err': f_wls_mn_err,
        'rel_uncertainty': rel_unc_wls_mn,
        'sigma_model_a': SIGMA_MODEL_A,
        'sigma_model_b': SIGMA_MODEL_B,
        'weights': weights_mn
    },
    'current_model': {
        'intercept': intercept,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value
    },
    'data': {
        'm_values': m_values,
        'fid_means': fid_means,
        'fid_stds': fid_stds,
        'samples_per_m': samples_per_m
    }
}

with open('wls_empirical_comparison_results.pickle', 'wb') as f:
    pk.dump(results, f)

print(f"Results saved to: wls_empirical_comparison_results.pickle")

# ================================================================================
# サマリー
# ================================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Comparison of Three Methods:

1. OLS (Ordinary Least Squares):
   - No weighting
   - f = {f_ols:.6f} ± {f_ols_err:.6f}
   - Relative uncertainty: {rel_unc_ols*100:.4f}%

2. WLS-Empirical (使用データから重み計算):
   - Weights from empirical σ(m)
   - f = {f_wls_emp:.6f} ± {f_wls_emp_err:.6f}
   - Relative uncertainty: {rel_unc_wls_emp*100:.4f}%
   - Improvement over OLS: {improvement_emp:+.2f}%
   - 統計的に正当（同じデータから計算）

3. WLS-MN (事前データのモデル使用):
   - Weights from prior model σ(m) = {SIGMA_MODEL_A:.6f} + {SIGMA_MODEL_B:.6f}*m
   - f = {f_wls_mn:.6f} ± {f_wls_mn_err:.6f}
   - Relative uncertainty: {rel_unc_wls_mn*100:.4f}%
   - Improvement over OLS: {improvement_mn:+.2f}%
   - 事前データを使用（統計的には問題あり）

Current Data Model:
   - σ(m) = {intercept:.6f} + {slope:.6f}*m
   - R² = {r_value**2:.4f}

Key Finding:
   WLS-Empirical vs WLS-MN: {diff:+.2f}% difference
   {'WLS-MN is better' if diff < 0 else 'WLS-Empirical is better' if diff > 0 else 'Almost identical'}

Statistical Validity:
   - WLS-Empirical: ✅ Statistically valid (no prior data used)
   - WLS-MN: ⚠️ Uses prior data (may underestimate uncertainty)
""")

print("="*80)
print("Analysis complete")
print("="*80)
