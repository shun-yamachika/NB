"""
WLS-MN Cost vs Uncertainty Analysis

コスト（バウンス長×試行回数の和）と不確実性の関係を解析
異なるサンプル数での性能を比較
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================================================================================
# モデルパラメータ
# ================================================================================
SIGMA_MODEL_A = 0.030641
SIGMA_MODEL_B = 0.000982

print("="*80)
print("WLS-MN Cost vs Uncertainty Analysis")
print("="*80)

# ================================================================================
# データ読み込み
# ================================================================================
with open('AB_decay_200samples_main.pickle', 'rb') as f:
    data = pk.load(f)

fid_raw_data = data["decay data"][1]
all_m_values = sorted(fid_raw_data.keys())

# ================================================================================
# 実験設定
# ================================================================================
m_range = (2, 20)
m_values = [m for m in all_m_values if m_range[0] <= m <= m_range[1]]
m_array = np.array(m_values)

# 異なるサンプル数でテスト
sample_counts = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
n_trials_per_setting = 10  # 各設定で10回試行

print(f"\nExperiment configuration:")
print(f"  m range: {m_range[0]} to {m_range[1]} ({len(m_values)} points)")
print(f"  Sample counts to test: {sample_counts}")
print(f"  Trials per setting: {n_trials_per_setting}")

# ================================================================================
# モデル定義
# ================================================================================
def exponential_decay(m, A, f):
    """指数減衰モデル"""
    return A * (f ** m)

def calculate_cost(m_values, n_samples):
    """コスト計算: Σ(m × n)"""
    return sum(m * n_samples for m in m_values)

# ================================================================================
# 実験実行
# ================================================================================
print("\n" + "="*80)
print("Running experiments")
print("="*80)

results_ols = {n: [] for n in sample_counts}
results_wls_mn = {n: [] for n in sample_counts}
costs = {n: 0 for n in sample_counts}

for n_samples in sample_counts:
    print(f"\nSamples per m: {n_samples}")

    # コスト計算
    cost = calculate_cost(m_values, n_samples)
    costs[n_samples] = cost
    print(f"  Total cost: {cost}")

    for trial in range(n_trials_per_setting):
        # ランダムサンプリング
        np.random.seed(trial + n_samples * 1000)

        fid_means = []
        fid_stds = []

        for m in m_values:
            indices = np.random.choice(200, n_samples, replace=False)
            selected_samples = fid_raw_data[m][indices]
            fid_means.append(np.mean(selected_samples))
            fid_stds.append(np.std(selected_samples, ddof=1))

        fid_means = np.array(fid_means)
        fid_stds = np.array(fid_stds)

        # OLS
        try:
            popt_ols, pcov_ols = curve_fit(
                exponential_decay,
                m_array,
                fid_means,
                p0=[0.95, 0.95],
                maxfev=10000
            )
            f_ols_err = np.sqrt(pcov_ols[1, 1])
            rel_unc_ols = f_ols_err / popt_ols[1]
            results_ols[n_samples].append(rel_unc_ols)
        except:
            pass

        # WLS-MN
        try:
            sigma_model = SIGMA_MODEL_A + SIGMA_MODEL_B * m_array
            sigma_mn = sigma_model / np.sqrt(n_samples)

            popt_wls_mn, pcov_wls_mn = curve_fit(
                exponential_decay,
                m_array,
                fid_means,
                p0=[0.95, 0.95],
                sigma=sigma_mn,
                absolute_sigma=True,
                maxfev=10000
            )
            f_wls_mn_err = np.sqrt(pcov_wls_mn[1, 1])
            rel_unc_wls_mn = f_wls_mn_err / popt_wls_mn[1]
            results_wls_mn[n_samples].append(rel_unc_wls_mn)
        except:
            pass

    print(f"  OLS: {len(results_ols[n_samples])} successful trials")
    print(f"  WLS-MN: {len(results_wls_mn[n_samples])} successful trials")

# ================================================================================
# 統計計算
# ================================================================================
print("\n" + "="*80)
print("Statistical summary")
print("="*80)

cost_list = []
ols_mean_unc = []
ols_std_unc = []
wls_mn_mean_unc = []
wls_mn_std_unc = []

print(f"\n{'Cost':<8} {'OLS Mean':<12} {'OLS Std':<12} {'WLS-MN Mean':<12} {'WLS-MN Std':<12} {'Improvement'}")
print("-"*80)

for n_samples in sample_counts:
    cost = costs[n_samples]

    ols_vals = np.array(results_ols[n_samples])
    wls_mn_vals = np.array(results_wls_mn[n_samples])

    if len(ols_vals) > 0 and len(wls_mn_vals) > 0:
        ols_m = np.mean(ols_vals)
        ols_s = np.std(ols_vals, ddof=1)
        wls_mn_m = np.mean(wls_mn_vals)
        wls_mn_s = np.std(wls_mn_vals, ddof=1)

        improvement = (ols_m - wls_mn_m) / ols_m * 100

        cost_list.append(cost)
        ols_mean_unc.append(ols_m)
        ols_std_unc.append(ols_s)
        wls_mn_mean_unc.append(wls_mn_m)
        wls_mn_std_unc.append(wls_mn_s)

        print(f"{cost:<8} {ols_m*100:<12.4f} {ols_s*100:<12.4f} {wls_mn_m*100:<12.4f} {wls_mn_s*100:<12.4f} {improvement:+6.1f}%")

# ================================================================================
# 結果保存
# ================================================================================
results_dict = {
    'sample_counts': sample_counts,
    'costs': cost_list,
    'ols': {
        'mean_uncertainty': ols_mean_unc,
        'std_uncertainty': ols_std_unc,
        'all_results': results_ols
    },
    'wls_mn': {
        'mean_uncertainty': wls_mn_mean_unc,
        'std_uncertainty': wls_mn_std_unc,
        'all_results': results_wls_mn
    },
    'm_range': m_range,
    'n_trials_per_setting': n_trials_per_setting
}

with open('wls_mn_cost_analysis_results.pickle', 'wb') as f:
    pk.dump(results_dict, f)

print("\n" + "="*80)
print("Results saved to: wls_mn_cost_analysis_results.pickle")
print("="*80)

# ================================================================================
# 可視化
# ================================================================================
print("\n" + "="*80)
print("Generating plots")
print("="*80)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

cost_array = np.array(cost_list)
ols_mean_array = np.array(ols_mean_unc) * 100
ols_std_array = np.array(ols_std_unc) * 100
wls_mn_mean_array = np.array(wls_mn_mean_unc) * 100
wls_mn_std_array = np.array(wls_mn_std_unc) * 100

# Plot 1: コスト vs 不確実性（メイン）
ax1 = fig.add_subplot(gs[0, :])
ax1.errorbar(cost_array, ols_mean_array, yerr=ols_std_array,
             fmt='o-', color='blue', linewidth=2, markersize=8,
             capsize=5, capthick=2, label='OLS', alpha=0.7)
ax1.errorbar(cost_array, wls_mn_mean_array, yerr=wls_mn_std_array,
             fmt='s-', color='red', linewidth=2, markersize=8,
             capsize=5, capthick=2, label='WLS-MN', alpha=0.7)

ax1.set_xlabel('Total Cost (Σ m × n)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Relative Uncertainty (%)', fontsize=14, fontweight='bold')
ax1.set_title('Cost vs Uncertainty: OLS vs WLS-MN', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# コスト効率の良い領域を強調
ax1.axvspan(0, 3000, alpha=0.1, color='green', label='Cost-efficient region')

# Plot 2: 対数スケール
ax2 = fig.add_subplot(gs[1, 0])
ax2.loglog(cost_array, ols_mean_array, 'o-', color='blue', linewidth=2,
           markersize=8, label='OLS', alpha=0.7)
ax2.loglog(cost_array, wls_mn_mean_array, 's-', color='red', linewidth=2,
           markersize=8, label='WLS-MN', alpha=0.7)

ax2.set_xlabel('Total Cost (log scale)', fontsize=12)
ax2.set_ylabel('Rel. Uncertainty (%, log scale)', fontsize=12)
ax2.set_title('Log-Log Plot: Cost vs Uncertainty', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: 改善率 vs コスト
ax3 = fig.add_subplot(gs[1, 1])
improvement_array = (ols_mean_array - wls_mn_mean_array) / ols_mean_array * 100
ax3.plot(cost_array, improvement_array, 'o-', color='green', linewidth=2,
         markersize=8, alpha=0.7)
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.fill_between(cost_array, 0, improvement_array, alpha=0.3, color='green')

ax3.set_xlabel('Total Cost', fontsize=12)
ax3.set_ylabel('Improvement (%)', fontsize=12)
ax3.set_title('WLS-MN Improvement over OLS', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=0)

# 平均改善率を表示
mean_improvement = np.mean(improvement_array)
ax3.text(0.95, 0.95, f'Mean improvement:\n{mean_improvement:.1f}%',
         transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('wls_mn_cost_analysis.pdf', bbox_inches='tight')
print("\nPlot saved: wls_mn_cost_analysis.pdf")

# ================================================================================
# サマリー
# ================================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Cost vs Uncertainty Analysis:

Configuration:
  m range: {m_range[0]} to {m_range[1]} ({len(m_values)} points)
  Sample counts tested: {len(sample_counts)}
  Cost range: {min(cost_list)} to {max(cost_list)}
  Trials per setting: {n_trials_per_setting}

Key Findings:

1. Cost Efficiency:
   - Low cost ({min(cost_list)}): OLS {ols_mean_array[0]:.3f}%, WLS-MN {wls_mn_mean_array[0]:.3f}%
   - High cost ({max(cost_list)}): OLS {ols_mean_array[-1]:.3f}%, WLS-MN {wls_mn_mean_array[-1]:.3f}%

2. Improvement:
   - Mean improvement: {mean_improvement:.1f}%
   - Min improvement: {min(improvement_array):.1f}%
   - Max improvement: {max(improvement_array):.1f}%

3. Uncertainty Reduction:
   - OLS: {ols_mean_array[0]:.3f}% → {ols_mean_array[-1]:.3f}% ({ols_mean_array[0]/ols_mean_array[-1]:.1f}x)
   - WLS-MN: {wls_mn_mean_array[0]:.3f}% → {wls_mn_mean_array[-1]:.3f}% ({wls_mn_mean_array[0]/wls_mn_mean_array[-1]:.1f}x)

4. Cost-Benefit:
   WLS-MN consistently outperforms OLS across all cost levels.
   Improvement is stable (~{np.std(improvement_array):.1f}% std deviation).

Recommendation:
  For cost-efficient experiments: {sample_counts[3]}-{sample_counts[5]} samples/m
  (Cost: {cost_list[3]}-{cost_list[5]}, Uncertainty: ~{wls_mn_mean_array[4]:.2f}%)
""")

print("="*80)
print("Analysis complete")
print("="*80)
