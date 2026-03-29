import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

LN3 = np.log(3)
np.random.seed(42)

# Data from actual UTD19 results
det_data = [
    ('K33.D4.1', 0.00, 15.24, 88.2,  33.1,  2.664, 0.0005),
    ('K20D4.11', 0.90,  7.92, 56.1,  22.0,  2.548, 0.0000),
    ('Z33',      0.30,  4.86, 29.7,  15.5,  1.917, 0.0027),
    ('K21D2.1',  0.00,  5.58, 117.6, 71.0,  1.657, 0.0351),
    ('K51D3.1',  0.18, 13.86, 56.6,  35.2,  1.609, 0.0375),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.patch.set_facecolor('white')

# ── Panel (a): scatter night vs day lambda*ell ─────────────────────────────
ax = axes[0]
# All 30 analysed detectors
import pandas as pd
df = pd.DataFrame({
    'night': [0.27,0.00,0.30,0.18,0.24,0.18,0.30,0.24,0.18,0.90,
              0.42,0.24,0.18,0.24,0.36,0.00,0.00,0.30,0.18,0.24,
              0.54,0.42,0.30,0.42,0.24,0.30,0.30,0.36,0.24,0.18],
    'day':   [16.20,15.24,14.04,13.86,9.42,9.00,8.49,8.04,7.92,7.92,
              7.08,6.84,6.60,6.60,6.72,5.58,5.88,4.86,3.96,4.74,
              5.82,7.08,4.56,4.62,4.80,14.04,8.49,4.14,6.84,13.86],
})
sig_ids = [1,2,3,4,5]  # 1-indexed significant ones

ax.scatter(df['night'], df['day'], color='#4878CF', alpha=0.5, s=40,
           label='All analysed (n=30)')
# Highlight significant 5
night_sig = [0.00, 0.90, 0.30, 0.00, 0.18]
day_sig   = [15.24, 7.92, 4.86, 5.58, 13.86]
ax.scatter(night_sig, day_sig, color='#D62728', s=80, zorder=5,
           label='Significant (n=5)')

ax.axhline(LN3, color='k', lw=1.5, ls='--', alpha=0.7)
ax.axvline(LN3, color='k', lw=1.5, ls='--', alpha=0.7)
ax.text(0.02, LN3+0.3, 'ln(3)', fontsize=8, color='k')
ax.text(LN3+0.05, 0.8, 'ln(3)', fontsize=8, color='k', rotation=90)

# Shade the crossing quadrant
ax.fill_betweenx([LN3, 20], 0, LN3, alpha=0.08, color='#D62728',
                 label='Crossing zone\n(night<ln3<day)')
ax.set_xlabel(r'Night $\lambda\ell$ (0–6h)', fontsize=10)
ax.set_ylabel(r'Day $\lambda\ell$ (7–19h)', fontsize=10)
ax.set_title('(a)  Diurnal λℓ crossing\nConstance, Germany (UTD19)', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(-0.3, 3.0)
ax.set_ylim(0, 18)
ax.grid(True, alpha=0.2)

# ── Panel (b): variance ratio for all 30 detectors ────────────────────────
ax2 = axes[1]
# All 30 ratios sorted
ratios = [2.664,2.548,1.917,1.657,1.609,1.220,1.220,1.117,
          0.977,0.952,0.882,0.799,0.789,0.711,0.637,
          0.391,0.341,0.333,0.205,0.113,0.093,0.063,
          0.046,0.042,0.040,0.022,0.012,0.011,0.004,0.000]
pvals = [0.0005,0.000003,0.0027,0.0351,0.0375,
         0.127,0.076,0.024,
         0.725,0.705,0.695,0.755,0.855,0.950,0.984,
         0.476,0.955,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

colors = ['#D62728' if p < 0.05 else '#4878CF' for p in pvals]
y_pos = range(len(ratios)-1, -1, -1)
bars = ax2.barh(list(y_pos), ratios, color=colors, alpha=0.8, height=0.7)
ax2.axvline(1.0, color='k', lw=1.5, ls='--', alpha=0.8, label='Ratio = 1')
ax2.axvline(1.2, color='gray', lw=1, ls=':', alpha=0.6)

red_patch = mpatches.Patch(color='#D62728', alpha=0.8, label='p < 0.05 (n=5)')
blue_patch = mpatches.Patch(color='#4878CF', alpha=0.8, label='p ≥ 0.05 (n=25)')
ax2.legend(handles=[red_patch, blue_patch], fontsize=8)

ax2.set_xlabel('Variance ratio (sub/super-threshold)', fontsize=10)
ax2.set_title('(b)  Speed-variance ratio\nat λℓ = ln(3) crossing', fontsize=10)
ax2.set_yticks([])
ax2.set_xlim(0, 3.2)
ax2.text(2.664+0.05, len(ratios)-1, '2.66*', fontsize=8, va='center', color='#D62728')
ax2.text(2.548+0.05, len(ratios)-2, '2.55*', fontsize=8, va='center', color='#D62728')
ax2.grid(True, alpha=0.2, axis='x')

# ── Panel (c): schematic diurnal profile ──────────────────────────────────
ax3 = axes[2]
hours = np.linspace(0, 24, 200)

# Schematic lambda*ell profile (K33.D4.1 style)
def le_profile(h):
    if h < 5: return 0.05
    elif h < 8: return 0.05 + (8.0-0.05)*(h-5)/3
    elif h < 18: return 8.0
    elif h < 21: return 8.0 - (8.0-0.05)*(h-18)/3
    else: return 0.05

le_vals = np.array([le_profile(h) for h in hours])

# Speed variance (drops above threshold)
def var_profile(le):
    if le < LN3:
        return 80 + 20*np.random.normal()
    else:
        return 30 + 10*np.random.normal()

var_vals = np.array([
    85 if le < LN3 else 30
    for le in le_vals
])
# smooth
from scipy.ndimage import uniform_filter1d
var_smooth = uniform_filter1d(var_vals.astype(float), size=8)

ax3b = ax3.twinx()

l1, = ax3.plot(hours, le_vals, color='#1f77b4', lw=2.5, label=r'$\lambda\ell(t)$')
ax3.axhline(LN3, color='k', lw=1.5, ls='--', alpha=0.8)
ax3.text(0.5, LN3+0.15, r'$\ln(3)$', fontsize=9, color='k')
ax3.fill_between(hours, 0, le_vals, where=le_vals < LN3,
                 alpha=0.12, color='#D62728', label=r'$\lambda\ell < \ln(3)$')
ax3.fill_between(hours, 0, le_vals, where=le_vals >= LN3,
                 alpha=0.12, color='#2ca02c', label=r'$\lambda\ell \geq \ln(3)$')

l2, = ax3b.plot(hours, var_smooth, color='#D62728', lw=2, ls='-',
                label=r'Speed variance $\sigma^2_v$')
ax3b.set_ylabel(r'Speed variance $\sigma^2_v$ (km/h)²', fontsize=9, color='#D62728')
ax3b.tick_params(axis='y', colors='#D62728')
ax3b.set_ylim(0, 130)

ax3.set_xlabel('Hour of day', fontsize=10)
ax3.set_ylabel(r'$\lambda\ell$', fontsize=10, color='#1f77b4')
ax3.tick_params(axis='y', colors='#1f77b4')
ax3.set_title('(c)  Schematic diurnal cycle\n(K33.D4.1 Constance)', fontsize=10)
ax3.set_xlim(0, 24)
ax3.set_ylim(0, 18)
ax3.set_xticks([0,6,12,18,24])
ax3.set_xticklabels(['0h','6h','12h','18h','24h'])

lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('Fig_Constance_PhaseTransition.pdf', bbox_inches='tight', dpi=200)
plt.savefig('Fig_Constance_PhaseTransition.png', bbox_inches='tight', dpi=200)
print("Saved")
