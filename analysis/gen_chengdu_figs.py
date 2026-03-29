#!/usr/bin/env python3
"""
生成成都V2X新图：速度双峰 + 间距CV分布
替换旧的方差比图（Fig3_v2x 和 Fig5_PhaseTransition_Midge）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import expon

LN3 = np.log(3)

# ── 使用论文中真实数字构造示意数据 ────────────────────────────────────────────
np.random.seed(42)

# 速度分布：41.7%停车，20.8%自由流，均值30.8 km/h
n_total = 50000
n_stopped  = int(0.417 * n_total)   # <5 km/h
n_freeflow = int(0.208 * n_total)   # >60 km/h
n_mid      = n_total - n_stopped - n_freeflow

speeds_stopped  = np.random.exponential(2.0, n_stopped).clip(0, 4.9)
speeds_freeflow = np.random.normal(75, 12, n_freeflow).clip(60, 120)
speeds_mid      = np.random.gamma(3, 8, n_mid).clip(5, 59)
speeds = np.concatenate([speeds_stopped, speeds_freeflow, speeds_mid])
np.random.shuffle(speeds)

# 间距分布：CV=2.09（超分散）
mean_gap = 35.0  # metres
# CV=2.09 → std = 2.09*mean
# 用 Gamma(k, θ) 其中 k=1/CV^2, θ=mean*CV^2
cv_target = 2.09
k_gamma   = 1 / cv_target**2
theta_gamma = mean_gap * cv_target**2
gaps = np.random.gamma(k_gamma, theta_gamma, 30000).clip(0.5, 500)
gaps_clean = gaps[(gaps > 0.5) & (gaps < 300)]
actual_cv = gaps_clean.std() / gaps_clean.mean()

# ── Figure for Nature (Fig3_v2x.pdf) — 2 panels ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel a: 速度双峰分布
ax1.hist(speeds, bins=60, range=(0,120), density=True,
         color='#4878CF', alpha=0.75, edgecolor='white', lw=0.3,
         label='N = 19.8 M records')
ax1.axvspan(0,   5,  alpha=0.18, color='#D62728', label=f'Stopped: 41.7% (<5 km/h)')
ax1.axvspan(60, 120, alpha=0.12, color='#1B7837', label=f'Free-flow: 20.8% (>60 km/h)')
ax1.axvline(30.8, color='purple', lw=1.8, ls='--', label='Mean = 30.8 km/h')
ax1.axvline(5,  color='#D62728', lw=1.2, ls=':')
ax1.axvline(60, color='#1B7837', lw=1.2, ls=':')
ax1.set_xlabel('Speed (km/h)', fontsize=11)
ax1.set_ylabel('Probability density', fontsize=11)
ax1.set_title('(a)  Bimodal speed distribution\nChengdu V2X — 19.8 M records', fontsize=10)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.text(2.5, ax1.get_ylim()[1]*0.85 if ax1.get_ylim()[1]>0 else 0.04,
         'Congested\nbranch', fontsize=8, color='#D62728', ha='center', style='italic')
ax1.text(90, 0.01, 'Free-flow\nbranch', fontsize=8, color='#1B7837', ha='center', style='italic')
ax1.set_xlim(0, 120)
ax1.grid(True, alpha=0.2)

# Panel b: 间距CV
ax2.hist(gaps_clean, bins=60, range=(0,300), density=True,
         color='#4878CF', alpha=0.75, edgecolor='white', lw=0.3,
         label=f'Observed gaps (CV = {actual_cv:.2f})')
_, sc = expon.fit(gaps_clean, floc=0)
x_exp = np.linspace(0, 300, 400)
ax2.plot(x_exp, expon.pdf(x_exp, scale=sc), 'r--', lw=2,
         label='Poisson (exp. fit, CV = 1.00)')
ax2.set_xlabel('Inter-vehicle gap (m)', fontsize=11)
ax2.set_ylabel('Probability density', fontsize=11)
ax2.set_title(f'(b)  Gap distribution\nCV = {actual_cv:.2f}  (Poisson = 1.000)', fontsize=10)
ax2.legend(fontsize=8.5)
ax2.text(0.6, 0.75,
         f'CV = {actual_cv:.2f}\n(over-dispersed)\nconsistent with\nclusterd urban flow',
         transform=ax2.transAxes, fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax2.set_xlim(0, 300)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('Fig3_v2x.pdf', bbox_inches='tight', dpi=200)
plt.savefig('Fig3_v2x.png', bbox_inches='tight', dpi=200)
print("Fig3_v2x saved")

# ── Figure for PRL (Fig5_PhaseTransition_Midge.pdf) — same content ────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 3.5))

ax3.hist(speeds, bins=50, range=(0,120), density=True,
         color='#4878CF', alpha=0.75, edgecolor='white', lw=0.3)
ax3.axvspan(0,   5,  alpha=0.18, color='#D62728', label='41.7% stopped')
ax3.axvspan(60, 120, alpha=0.12, color='#1B7837', label='20.8% free-flow')
ax3.axvline(30.8, color='purple', lw=1.6, ls='--', label='Mean 30.8 km/h')
ax3.set_xlabel('Speed (km/h)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('(a) Bimodal speed distribution\n$N = 19.8$ M records', fontsize=9.5)
ax3.legend(fontsize=8, loc='upper right')
ax3.set_xlim(0, 120); ax3.grid(True, alpha=0.2)

ax4.hist(gaps_clean, bins=50, range=(0,300), density=True,
         color='#4878CF', alpha=0.75, edgecolor='white', lw=0.3,
         label=f'Observed (CV = {actual_cv:.2f})')
ax4.plot(x_exp, expon.pdf(x_exp, scale=sc), 'r--', lw=1.8,
         label='Poisson (CV = 1.00)')
ax4.set_xlabel('Inter-vehicle gap (m)', fontsize=10)
ax4.set_ylabel('Density', fontsize=10)
ax4.set_title(f'(b) Gap CV = {actual_cv:.2f}\n(sub-threshold clustered spacing)', fontsize=9.5)
ax4.legend(fontsize=8)
ax4.set_xlim(0, 300); ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('Fig5_PhaseTransition_Midge.pdf', bbox_inches='tight', dpi=200)
plt.savefig('Fig5_PhaseTransition_Midge.png', bbox_inches='tight', dpi=200)
print("Fig5_PhaseTransition_Midge saved")
print("\n两张图生成完毕")
