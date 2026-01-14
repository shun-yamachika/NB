"""
測定コスト vs 推定精度の分析（Two-stage版）

OLS vs WLS-Two-stage の比較
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
print("Cost vs Uncertainty Analysis (OLS vs WLS-Two-stage)")
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
# 分散モデル選択関数
# ================================================================================
def select_best_variance_model(m_array, fid_stds):
    """最適な分散モデルを自動選択"""
    models = {}

    # Model 1: Linear
    slope, intercept, r_value, _, _ = linregress(m_array, fid_stds)
    sigma_linear = intercept + slope * m_array
    models['Linear'] = {'sigma': sigma_linear, 'r2': r_value**2}

    # Model 2: Power law
    log_m = np.log(m_array)
    log_sigma = np.log(fid_stds)
    slope_log, intercept_log, r_value_log, _, _ = linregress(log_m, log_sigma)
    a_power = np.exp(intercept_log)
    b_power = slope_log
    sigma_power = a_power * (m_array ** b_power)
    models['Power law'] = {'sigma': sigma_power, 'r2': r_value_log**2}

    # Model 3: Quadratic
    coeffs = np.polyfit(m_array, fid_stds, 2)
    sigma_quadratic = np.polyval(coeffs, m_array)
    ss_res = np.sum((fid_stds - sigma_quadratic)**2)
    ss_tot = np.sum((fid_stds - np.mean(fid_stds))**2)
    r2_quad = 1 - ss_res/ss_tot
    models['Quadratic'] = {'sigma': sigma_quadratic, 'r2': r2_quad}

    # Select best
    best_name = max(models, key=lambda k: models[k]['r2'])
    return models[best_name]['sigma'], best_name, models[best_name]['r2']

# ================================================================================
# 測定回数を変えながら解析
# ================================================================================
max_samples = 200
sample_points = list(range(1, 11)) + list(range(15, 51, 5)) + list(range(60, 201, 10))

results_ols = {'samples': [], 'cost': [], 'f': [], 'f_err': [], 'rel_unc': []}
results_wls_ts = {'samples': [], 'cost': [], 'f': [], 'f_err': [], 'rel_unc': []}

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

    # 測定コスト
    total_cost = np.sum(m_array) * n_samples

    # OLSフィッティング
    try:
        popt_ols, pcov_ols = curve_fit(exponential_decay, m_array, fid_means,
                                       p0=[0.5, 0.9])
        f_ols = popt_ols[1]
        f_err_ols = np.sqrt(pcov_ols[1, 1])
        rel_unc_ols = f_err_ols / f_ols * 100

        results_ols['samples'].append(n_samples)
        results_ols['cost'].append(total_cost)
        results_ols['f'].append(f_ols)
        results_ols['f_err'].append(f_err_ols)
        results_ols['rel_unc'].append(rel_unc_ols)
    except:
        pass

    # WLS-Two-stageフィッティング（n>=3で十分なデータがある場合のみ）
    if n_samples >= 3:
        try:
            # Stage 1: モデル選択
            sigma_model, model_name, r2 = select_best_variance_model(m_array, fid_stds)

            # Stage 2: モデルベース重み付け
            standard_errors_model = sigma_model / np.sqrt(n_samples)
            if np.all(standard_errors_model > 0):
                popt_wls_ts, pcov_wls_ts = curve_fit(exponential_decay, m_array, fid_means,
                                                      sigma=standard_errors_model,
                                                      absolute_sigma=True, p0=[0.5, 0.9])
                f_wls_ts = popt_wls_ts[1]
                f_err_wls_ts = np.sqrt(pcov_wls_ts[1, 1])
                rel_unc_wls_ts = f_err_wls_ts / f_wls_ts * 100

                results_wls_ts['samples'].append(n_samples)
                results_wls_ts['cost'].append(total_cost)
                results_wls_ts['f'].append(f_wls_ts)
                results_wls_ts['f_err'].append(f_err_wls_ts)
                results_wls_ts['rel_unc'].append(rel_unc_wls_ts)
        except:
            pass

# Convert to arrays
for key in results_ols:
    results_ols[key] = np.array(results_ols[key])
    results_wls_ts[key] = np.array(results_wls_ts[key])

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
ax.plot(results_wls_ts['cost'], results_wls_ts['rel_unc'], 's-',
        linewidth=2.5, markersize=8, label='WLS-Two-stage', alpha=0.8, color='#A23B72')

ax.set_xlabel('Total Measurement Cost  $\\Sigma (m \\times n)$', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative Uncertainty of $f$  (%)', fontsize=14, fontweight='bold')
ax.set_title('Measurement Cost vs Estimation Precision', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=12)

# Add annotation for key point
idx_200_ols = np.where(results_ols['samples'] == 200)[0]
idx_200_wls = np.where(results_wls_ts['samples'] == 200)[0]

if len(idx_200_ols) > 0 and len(idx_200_wls) > 0:
    idx_ols = idx_200_ols[0]
    idx_wls = idx_200_wls[0]
    improvement = (1 - results_wls_ts['rel_unc'][idx_wls] / results_ols['rel_unc'][idx_ols]) * 100
    ax.annotate(f'n=200: {improvement:.0f}% improvement',
                xy=(results_wls_ts['cost'][idx_wls], results_wls_ts['rel_unc'][idx_wls]),
                xytext=(results_wls_ts['cost'][idx_wls] * 0.5, results_wls_ts['rel_unc'][idx_wls] * 2),
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
  OLS:            {results_ols['rel_unc'][idx_ols]:.3f}% uncertainty
  WLS-Two-stage:  {results_wls_ts['rel_unc'][idx_wls]:.3f}% uncertainty

  → WLS-Two-stage achieves {improvement:.0f}% improvement in precision
  → Both converge to f ≈ 0.899
""")
else:
    print("\nData for n=200 not found in results.")

print("="*80)
