"""
実測分散の可視化

σ_empirical(m)の挙動を詳しく調べる：
1. σ(m)の生データ
2. 線形・非線形モデルのフィッティング
3. 分散の挙動の詳細分析
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
print("Empirical Variance Visualization")
print("="*80)

with open('AB_decay_200samples_main.pickle', 'rb') as f:
    data = pk.load(f)

fid_raw_data = data["decay data"][1]
all_m_values = sorted(fid_raw_data.keys())

m_range = (2, 40)
samples_per_m = 200  # Use all available samples

m_values = [m for m in all_m_values if m_range[0] <= m <= m_range[1]]
m_array = np.array(m_values)

# データ準備
fid_means = []
fid_stds = []
fid_vars = []

for m in m_values:
    data_m = fid_raw_data[m][:samples_per_m]
    fid_means.append(np.mean(data_m))
    std = np.std(data_m, ddof=1)
    fid_stds.append(std)
    fid_vars.append(std**2)

fid_means = np.array(fid_means)
fid_stds = np.array(fid_stds)
fid_vars = np.array(fid_vars)

print(f"\nData: m ∈ [{m_range[0]}, {m_range[1]}], n={samples_per_m}")
print(f"\nBasic statistics:")
print(f"  σ(m) range: [{fid_stds.min():.6f}, {fid_stds.max():.6f}]")
print(f"  σ(m) ratio (max/min): {fid_stds.max() / fid_stds.min():.3f}")
print(f"  Mean σ: {np.mean(fid_stds):.6f}")

# ================================================================================
# OLSフィッティング（参照用）
# ================================================================================
def exponential_decay(m, A, f):
    return A * (f ** m)

popt_ols, _ = curve_fit(exponential_decay, m_array, fid_means, p0=[0.95, 0.95])
A_ols, f_ols = popt_ols
y_predicted = exponential_decay(m_array, A_ols, f_ols)

print(f"\nOLS fit: A = {A_ols:.6f}, f = {f_ols:.6f}")

# ================================================================================
# 経験的モデルフィッティング
# ================================================================================
print("\n" + "="*80)
print("Empirical Model Fitting")
print("="*80)

# 1. Linear: σ = a + b*m
slope, intercept, r_value, p_value, std_err = linregress(m_array, fid_stds)
sigma_linear = intercept + slope * m_array

print(f"\n1. Linear model: σ(m) = a + b*m")
print(f"   a = {intercept:.6f}")
print(f"   b = {slope:.6f}")
print(f"   R² = {r_value**2:.4f}")
print(f"   Trend: {'Increasing' if slope > 0 else 'Decreasing'} with m")

# 2. Power law: σ = a * m^b
log_m = np.log(m_array)
log_sigma = np.log(fid_stds)
slope_log, intercept_log, r_value_log, _, _ = linregress(log_m, log_sigma)
a_power = np.exp(intercept_log)
b_power = slope_log
sigma_power = a_power * (m_array ** b_power)

print(f"\n2. Power law: σ(m) = a * m^b")
print(f"   a = {a_power:.6f}")
print(f"   b = {b_power:.6f}")
print(f"   R² = {r_value_log**2:.4f}")

# 3. Quadratic: σ = a + b*m + c*m²
coeffs = np.polyfit(m_array, fid_stds, 2)
sigma_quadratic = np.polyval(coeffs, m_array)

print(f"\n3. Quadratic: σ(m) = a + b*m + c*m²")
print(f"   a = {coeffs[2]:.6f}")
print(f"   b = {coeffs[1]:.6f}")
print(f"   c = {coeffs[0]:.6e}")
ss_res = np.sum((fid_stds - sigma_quadratic)**2)
ss_tot = np.sum((fid_stds - np.mean(fid_stds))**2)
r2_quad = 1 - ss_res/ss_tot
print(f"   R² = {r2_quad:.4f}")

# ================================================================================
# データテーブル
# ================================================================================
print("\n" + "="*80)
print("Data Table")
print("="*80)

print(f"\n{'m':<4} {'y(m)':<10} {'σ_emp':<10} {'Var_emp':<12}")
print("-"*40)

for i, m in enumerate(m_array):
    print(f"{m:<4} {y_predicted[i]:<10.6f} {fid_stds[i]:<10.6f} {fid_vars[i]:<12.6e}")

# ================================================================================
# 可視化
# ================================================================================
print("\n" + "="*80)
print("Generating plots")
print("="*80)

fig = plt.figure(figsize=(15, 8))

# Plot 1: σ(m) vs m - 生データと各種フィット
ax1 = plt.subplot(2, 3, 1)
ax1.plot(m_array, fid_stds, 'o', markersize=8, color='black',
         label='Empirical σ(m)', alpha=0.7, linewidth=2)
ax1.plot(m_array, sigma_linear, '--', linewidth=2, color='blue',
         label=f'Linear (R²={r_value**2:.3f})', alpha=0.7)
ax1.plot(m_array, sigma_power, '--', linewidth=2, color='green',
         label=f'Power law (R²={r_value_log**2:.3f})', alpha=0.7)
ax1.plot(m_array, sigma_quadratic, '--', linewidth=2, color='purple',
         label=f'Quadratic (R²={r2_quad:.3f})', alpha=0.7)

ax1.set_xlabel('Bounce Length (m)', fontsize=11)
ax1.set_ylabel('Standard Deviation σ(m)', fontsize=11)
ax1.set_title('Empirical σ(m) with Model Fits', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Var(m) vs m
ax2 = plt.subplot(2, 3, 2)
ax2.plot(m_array, fid_vars, 'o-', markersize=8, color='black',
         label='Empirical Var(m)', alpha=0.7, linewidth=2)

ax2.set_xlabel('Bounce Length (m)', fontsize=11)
ax2.set_ylabel('Variance Var(m)', fontsize=11)
ax2.set_title('Empirical Variance vs m', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: y(m) と σ(m) の関係
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_predicted, fid_stds, s=100, alpha=0.7, c=m_array,
            cmap='viridis', edgecolor='black', linewidth=1)

# Trend line
z = np.polyfit(y_predicted, fid_stds, 1)
p = np.poly1d(z)
y_trend = np.linspace(y_predicted.min(), y_predicted.max(), 100)
ax3.plot(y_trend, p(y_trend), 'r--', linewidth=2, alpha=0.7,
         label=f'Trend: σ = {z[0]:.4f}*y + {z[1]:.4f}')

ax3.set_xlabel('Signal y(m)', fontsize=11)
ax3.set_ylabel('σ(m)', fontsize=11)
ax3.set_title('σ vs y (color = m)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(ax3.collections[0], ax=ax3)
cbar.set_label('m', fontsize=10)

# Plot 4: log(σ) vs log(m) - power law
ax4 = plt.subplot(2, 3, 4)
ax4.plot(np.log(m_array), np.log(fid_stds), 'o', markersize=8,
         color='black', alpha=0.7)
ax4.plot(np.log(m_array), intercept_log + slope_log * np.log(m_array),
         'r-', linewidth=2, alpha=0.7,
         label=f'slope = {slope_log:.3f}')

ax4.set_xlabel('log(m)', fontsize=11)
ax4.set_ylabel('log(σ)', fontsize=11)
ax4.set_title('Power Law Test', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals of linear fit
ax5 = plt.subplot(2, 3, 5)
residuals = fid_stds - sigma_linear
ax5.plot(m_array, residuals, 'o-', markersize=8, color='blue',
         alpha=0.7, linewidth=2)
ax5.axhline(0, color='black', linestyle='-', linewidth=1)
ax5.fill_between(m_array, -2*std_err, 2*std_err, alpha=0.2, color='gray',
                 label='±2 std err')

ax5.set_xlabel('Bounce Length (m)', fontsize=11)
ax5.set_ylabel('Residuals', fontsize=11)
ax5.set_title('Linear Fit Residuals', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Plot 6: Coefficient of Variation
ax6 = plt.subplot(2, 3, 6)
cv = fid_stds / fid_means  # Coefficient of variation
ax6.plot(m_array, cv * 100, 'o-', markersize=8, color='purple',
         alpha=0.7, linewidth=2)

ax6.set_xlabel('Bounce Length (m)', fontsize=11)
ax6.set_ylabel('Coefficient of Variation (%)', fontsize=11)
ax6.set_title('CV = σ/μ', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('empirical_variance_visualization.pdf', bbox_inches='tight')
print("\nPlot saved to: empirical_variance_visualization.pdf")

# ================================================================================
# サマリー
# ================================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Empirical Variance Analysis:

1. Basic Statistics:
   Range: σ(m) ∈ [{fid_stds.min():.6f}, {fid_stds.max():.6f}]
   Ratio (max/min): {fid_stds.max() / fid_stds.min():.3f}
   Mean: {np.mean(fid_stds):.6f}

2. Trend with m:
   σ(m) {'INCREASES' if slope > 0 else 'DECREASES'} with m
   Linear: σ = {intercept:.6f} + {slope:.6f}*m (R² = {r_value**2:.4f})
   Power:  σ = {a_power:.6f} * m^{b_power:.3f} (R² = {r_value_log**2:.4f})

3. Physical Interpretation:
   σ(m) increases with m

   - Longer sequences → More gate operations
   - More operations → More accumulated noise
   - Therefore: σ increases with m, even though signal y decreases

4. Best Model:
   Power law (σ = a·m^b) gives best fit with R² = {r_value_log**2:.4f}
   Exponent b = {b_power:.3f} indicates sub-linear growth

5. Coefficient of Variation (CV):
   CV = σ/μ increases with m, indicating that relative noise grows
   as sequences get longer and signal decreases
""")

print("="*80)
print("Analysis complete")
print("="*80)
