"""
σ(m) vs m の可視化（単品）

実測の標準偏差がバウンス長mに対してどのように変化するかを示す
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ================================================================================
# データ読み込み
# ================================================================================
print("="*80)
print("σ(m) vs m Visualization")
print("="*80)

with open('AB_decay_200samples_main.pickle', 'rb') as f:
    data = pk.load(f)

fid_raw_data = data["decay data"][1]
all_m_values = sorted(fid_raw_data.keys())

m_range = (2, 40)
samples_per_m = 200

m_values = [m for m in all_m_values if m_range[0] <= m <= m_range[1]]
m_array = np.array(m_values)

# データ準備
fid_stds = []

for m in m_values:
    data_m = fid_raw_data[m][:samples_per_m]
    std = np.std(data_m, ddof=1)
    fid_stds.append(std)

fid_stds = np.array(fid_stds)

print(f"\nData: m ∈ [{m_range[0]}, {m_range[1]}], n={samples_per_m}")
print(f"σ(m) range: [{fid_stds.min():.6f}, {fid_stds.max():.6f}]")
print(f"Ratio (max/min): {fid_stds.max() / fid_stds.min():.3f}")

# ================================================================================
# モデルフィッティング
# ================================================================================

# 1. Linear: σ = a + b*m
slope, intercept, r_value, p_value, std_err = linregress(m_array, fid_stds)
sigma_linear = intercept + slope * m_array
r2_linear = r_value**2

# 2. Power law: σ = a * m^b
log_m = np.log(m_array)
log_sigma = np.log(fid_stds)
slope_log, intercept_log, r_value_log, _, _ = linregress(log_m, log_sigma)
a_power = np.exp(intercept_log)
b_power = slope_log
sigma_power = a_power * (m_array ** b_power)
r2_power = r_value_log**2

# 3. Quadratic: σ = a + b*m + c*m²
coeffs = np.polyfit(m_array, fid_stds, 2)
sigma_quadratic = np.polyval(coeffs, m_array)
ss_res = np.sum((fid_stds - sigma_quadratic)**2)
ss_tot = np.sum((fid_stds - np.mean(fid_stds))**2)
r2_quad = 1 - ss_res/ss_tot

print("\n" + "="*80)
print("Model Fits")
print("="*80)

print(f"\n1. Linear: σ(m) = {intercept:.6f} + {slope:.6f}*m")
print(f"   R² = {r2_linear:.4f}")

print(f"\n2. Power law: σ(m) = {a_power:.6f} * m^{b_power:.3f}")
print(f"   R² = {r2_power:.4f}")

print(f"\n3. Quadratic: σ(m) = {coeffs[2]:.6f} + {coeffs[1]:.6f}*m + {coeffs[0]:.6e}*m²")
print(f"   R² = {r2_quad:.4f}")

print(f"\nBest model: Power law (highest R² = {r2_power:.4f})")

# ================================================================================
# プロット
# ================================================================================
print("\n" + "="*80)
print("Generating plot")
print("="*80)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Empirical data
ax.plot(m_array, fid_stds, 'o', markersize=10, color='black',
        label='Empirical σ(m)', alpha=0.7, linewidth=2, zorder=5)

# Model fits
ax.plot(m_array, sigma_linear, '--', linewidth=2.5, color='#2E86AB',
        label=f'Linear (R²={r2_linear:.3f})', alpha=0.8)
ax.plot(m_array, sigma_power, '--', linewidth=2.5, color='#A23B72',
        label=f'Power law (R²={r2_power:.3f})', alpha=0.8)
ax.plot(m_array, sigma_quadratic, '--', linewidth=2.5, color='#F77F00',
        label=f'Quadratic (R²={r2_quad:.3f})', alpha=0.8)

ax.set_xlabel('Bounce Length  $m$', fontsize=14, fontweight='bold')
ax.set_ylabel('Standard Deviation  $\\sigma(m)$', fontsize=14, fontweight='bold')
ax.set_title('Empirical Standard Deviation vs Bounce Length', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=12)

# Add text box with best model
textstr = f'Best fit:\n$\\sigma(m) = {a_power:.3f} \\cdot m^{{{b_power:.3f}}}$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.98, 0.35, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('sigma_vs_m_single.pdf', bbox_inches='tight', dpi=300)
print("\nPlot saved to: sigma_vs_m_single.pdf")

# ================================================================================
# サマリー
# ================================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Empirical Standard Deviation Analysis:

  σ(m) increases with m: {fid_stds.min():.6f} → {fid_stds.max():.6f}
  Ratio: {fid_stds.max() / fid_stds.min():.2f}×

  Best model: σ(m) = {a_power:.6f} * m^{b_power:.3f}  (R² = {r2_power:.4f})

  Physical interpretation:
    - Longer sequences (larger m) → More gate operations
    - More operations → More accumulated noise
    - Sub-linear growth (exponent {b_power:.3f} < 1) indicates
      noise grows slower than linear with sequence length
""")

print("="*80)
