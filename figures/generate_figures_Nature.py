#!/usr/bin/env python3
"""
Nature论文图片生成脚本 (Fig1, Fig3, Fig4, Fig5)
用法: python3 generate_figures_Nature.py
输出: Fig1_theory.pdf, Fig2_blindpred.pdf, Fig3_v2x.pdf, Fig4_eightsystems.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

LN3 = np.log(3)  # 1.0986
LN2 = np.log(2)  # 0.6931

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Blind prediction (验证图)
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel a: predicted vs observed rho_c
datasets = {
    'highD (DE)':   (80,  40.7, 40.7, 'highway'),
    'NGSIM (US)':   (70,  35.7, 35.6, 'highway'),
    'Zen (JP)':     (95,  48.4, 50.0, 'urban'),
    'pNEUMA (GR)':  (60,  30.6, 29.4, 'urban'),
}
# Chengdu removed: only OBU speed/gap records available, 
# no independently measured rho_c for blind prediction validation
rho_pred = np.array([v[1] for v in datasets.values()])
rho_obs  = np.array([v[2] for v in datasets.values()])
types    = [v[3] for v in datasets.values()]
names    = list(datasets.keys())

ax = axes[0]
rng = np.linspace(25, 55, 100)
ax.fill_between(rng, rng*0.95, rng*1.05, alpha=0.15, color='grey', label='±5% band')
ax.plot(rng, rng, 'k--', lw=1, label='Perfect agreement')
for i, name in enumerate(names):
    m = 'o' if types[i]=='highway' else '^'
    color = '#1f77b4' if types[i]=='highway' else '#d62728'
    ax.scatter(rho_obs[i], rho_pred[i], color=color, marker=m, s=70, zorder=5)
    ax.annotate(name.split(' ')[0], (rho_obs[i], rho_pred[i]),
                textcoords='offset points', xytext=(4,2), fontsize=7.5)
ax.set_xlabel('Observed $\\rho_c$ (veh/km)', fontsize=10)
ax.set_ylabel('Predicted $\\hat{\\rho}_c$ (veh/km)', fontsize=10)
ax.set_title('(a) Blind prediction\n(4 datasets, MAPE = 1.9%)', fontsize=10)
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#1f77b4',ms=8,label='Highway'),
       Line2D([0],[0],marker='^',color='w',markerfacecolor='#d62728',ms=8,label='Urban')]
ax.legend(handles=leg, fontsize=8)
ax.grid(True, alpha=0.25)

# Panel b: percentage errors
errors = np.abs(rho_pred - rho_obs) / rho_obs * 100
ax2 = axes[1]
bar_colors = ['#1f77b4' if t=='highway' else '#d62728' for t in types]
ax2.barh(names, errors, color=bar_colors, alpha=0.8, edgecolor='grey')
ax2.axvline(x=1.9, color='green', ls='-', lw=1.5, label='MAPE = 1.9%')
ax2.axvline(x=2.4, color='orange', ls='--', lw=1.5, label='Greenshields 2.4%')
ax2.set_xlabel('Percentage error (%)', fontsize=10)
ax2.set_title('(b) Errors by dataset', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25, axis='x')

# Panel c: regime diagram
ax3 = axes[2]
p = np.linspace(0, 0.25, 300)
ell_km = 0.300  # 300m in km
for rho0, label, color, ls in [
    (80, 'Dense urban (80 veh/km)', '#d62728', '-'),
    (55, 'Beijing arterial (55)', '#1f77b4', '--'),
    (30, 'Highway (30)', '#2ca02c', ':'),
]:
    le = rho0 * p * ell_km  # veh/km * fraction * km = dimensionless
    ax3.plot(p*100, le, color=color, ls=ls, lw=2, label=label)

ax3.axhline(y=LN3, color='k', lw=1.5, label=f'$\\ln(3) = {LN3:.3f}$')
ax3.axhline(y=LN2, color='k', lw=1, ls=':', alpha=0.5, label=f'$\\ln(2) = {LN2:.3f}$')
ax3.fill_between([0,25], LN2, LN3, alpha=0.1, color='orange', label='Regime I')
ax3.fill_between([0,25], LN3, 1.5, alpha=0.1, color='green',  label='Regime II+')
ax3.axvspan(0, 2, alpha=0.15, color='red')
ax3.text(1, 0.05, 'Now\n≈2%', fontsize=7.5, ha='center', color='red')
ax3.set_xlabel('CAV penetration $p$ (%)', fontsize=10)
ax3.set_ylabel('$\\lambda\\ell = \\rho_0 p\\ell$', fontsize=10)
ax3.set_title('(c) Regime diagram', fontsize=10)
ax3.legend(fontsize=7.5, loc='upper left')
ax3.set_xlim(0, 25); ax3.set_ylim(0, 1.5)
ax3.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('Fig1_theory.pdf', bbox_inches='tight', dpi=150)
plt.savefig('Fig1_theory.png', bbox_inches='tight', dpi=150)
print("Fig1 saved")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Policy / deployment roadmap
# ══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4))

# Panel a: p_c vs rho_0
ax1 = axes4[0]
rho0_range = np.linspace(20, 100, 300)  # veh/km
for ell_m, ls, color, label in [
    (300, '-',  '#1f77b4', '$\\ell=300$ m (V2X std)'),
    (200, '--', '#d62728', '$\\ell=200$ m'),
]:
    pc = LN3 / (rho0_range * ell_m / 1000) * 100  # veh/km * km = dimensionless
    ax1.plot(rho0_range, pc, ls=ls, lw=2, color=color, label=label)

ax1.axhspan(10, 15, alpha=0.15, color='grey', label='Simulation consensus')
for rho0_pt in [30, 55, 80]:
    pc_pt = LN3 / (rho0_pt * 300 / 1000) * 100
    ax1.scatter([rho0_pt], [pc_pt], color='#1f77b4', zorder=5, s=60)
    ax1.annotate(f'{pc_pt:.1f}%', (rho0_pt, pc_pt),
                 textcoords='offset points', xytext=(3, 4), fontsize=8)
ax1.set_xlabel('Background density $\\rho_0$ (veh/km)', fontsize=10)
ax1.set_ylabel('Critical CAV penetration $p_c$ (%)', fontsize=10)
ax1.set_title('(a) $p_c$ vs road density\n(no free parameters)', fontsize=10)
ax1.legend(fontsize=8)
ax1.set_xlim(20, 100); ax1.set_ylim(0, 35)
ax1.grid(True, alpha=0.25)

# Panel b: Beijing regime
ax2 = axes4[1]
p_range = np.linspace(0, 0.20, 300)
le_bj = 55 * p_range * 0.300  # 55 veh/km * p * 0.3 km
ax2.plot(p_range*100, le_bj, color='#1f77b4', lw=2.5)
ax2.axhline(y=LN3, color='k', lw=1.5, label=f'$\\ln(3) = {LN3:.3f}$')
ax2.axhline(y=LN2, color='k', lw=1, ls=':', alpha=0.6)
ax2.fill_between([0,20], 0,    LN2, alpha=0.15, color='red',    label='Regime 0')
ax2.fill_between([0,20], LN2,  LN3, alpha=0.15, color='orange', label='Regime I')
ax2.fill_between([0,20], LN3,  0.5, alpha=0.15, color='green',  label='Regime III')
# 6.7% threshold for Beijing arterial
pc_bj = LN3 / (55 * 0.300) * 100  # = 6.67%
ax2.axvline(x=pc_bj, color='orange', lw=1.5, ls='--', alpha=0.8)
ax2.text(pc_bj, 0.02, f'$p_c={pc_bj:.1f}\\%$', fontsize=8, color='darkorange', ha='center')
ax2.axvline(x=12.2, color='green', lw=1.5, ls='--', alpha=0.8)
ax2.text(12.5, 0.02, '12.2%\n(highway)', fontsize=8, color='darkgreen')
ax2.axvspan(0, 2, alpha=0.2, color='red')
ax2.text(1, 0.05, 'Now\n$p=2\\%$', fontsize=8, ha='center', color='darkred')
ax2.set_xlabel('CAV penetration $p$ (%)', fontsize=10)
ax2.set_ylabel('$\\lambda\\ell$', fontsize=10)
ax2.set_title('(b) Beijing arterial\n($\\rho_0=55$ veh/km)', fontsize=10)
ax2.legend(fontsize=8, loc='upper left')
ax2.set_xlim(0, 20); ax2.set_ylim(0, 0.5)
ax2.grid(True, alpha=0.25)

# Panel c: suppression probability schematic
ax3 = axes4[2]
le_range = np.linspace(0, 2.0, 500)
def supp_prob(le):
    return 0.0 if le < LN3 else min(1 - np.exp(-3*(le - LN3)), 0.99)
supp = np.array([supp_prob(x) for x in le_range])
ax3.plot(le_range, supp, color='#1f77b4', lw=2.5)
ax3.axvline(x=LN3, color='k', ls='--', lw=1.5, label='$\\ln(3) = 1.099$')
ax3.axvline(x=LN2, color='k', ls=':', lw=1, alpha=0.6, label='$\\ln(2) = 0.693$')
ax3.fill_betweenx([0,1], 0, LN2, alpha=0.1, color='red')
ax3.fill_betweenx([0,1], LN2, LN3, alpha=0.1, color='orange')
ax3.fill_betweenx([0,1], LN3, 2.0, alpha=0.1, color='green')
le_beijing_now = 55 * 0.02 * 0.300  # = 0.33
ax3.scatter([le_beijing_now], [0], color='black', s=80, zorder=6)
ax3.annotate('Beijing now\n$p=2\\%$', xy=(le_beijing_now, 0),
             xytext=(0.6, 0.15), fontsize=8,
             arrowprops=dict(arrowstyle='->', color='black'))
ax3.set_xlabel('Coordination parameter $\\lambda\\ell$', fontsize=10)
ax3.set_ylabel('Phantom-jam suppression probability', fontsize=10)
ax3.set_title('(c) Phase transition at $\\ln(3)$\n(schematic)', fontsize=10)
ax3.legend(fontsize=8)
ax3.set_xlim(0, 2.0); ax3.set_ylim(-0.05, 1.05)
ax3.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('Fig4_eightsystems.pdf', bbox_inches='tight', dpi=150)
plt.savefig('Fig4_eightsystems.png', bbox_inches='tight', dpi=150)
print("Fig4 saved")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Signal-corrected threshold (urban extension)
# ══════════════════════════════════════════════════════════════════════════════
fig5 = plt.figure(figsize=(12, 5))
gs5 = GridSpec(1, 2, figure=fig5, wspace=0.35)

# Panel a: p_c(r) curves
ax1 = fig5.add_subplot(gs5[0])
r = np.linspace(0, 0.75, 300)
ell_km = 0.300
for rho0, label, color, ls in [
    (80, 'Dense urban peak\n(80 veh/km)', '#d62728', '-'),
    (55, 'Beijing arterial\n(55 veh/km)', '#1f77b4', '--'),
    (30, 'Standard highway\n(30 veh/km)', '#2ca02c', ':'),
]:
    lam_ell = rho0 * ell_km  # veh/km * km = dimensionless (for p=1)
    pc = LN3 / (lam_ell * (1 - r))
    pc_plot = np.where(pc <= 1, pc * 100, np.nan)
    ax1.plot(r * 100, pc_plot, color=color, ls=ls, lw=2, label=label)

ax1.axhspan(10, 15, alpha=0.12, color='grey', label='Simulation consensus\n(10–15%)')
ax1.scatter([40], [2], color='black', zorder=5, s=80)
ax1.annotate('Current Beijing\n($r=40\\%$, $p=2\\%$)',
             xy=(40, 2), xytext=(52, 8), fontsize=8, ha='center',
             arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax1.set_xlabel('Signal red ratio $r$ (%)', fontsize=11)
ax1.set_ylabel('Critical CAV penetration $p_c$ (%)', fontsize=11)
ax1.set_title('(a) Signal-corrected threshold $p_c(r)$\n[exact formula, zero free parameters]', fontsize=10)
ax1.set_xlim(0, 75); ax1.set_ylim(0, 35)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.25)

# Panel b: substitutability contours
ax2 = fig5.add_subplot(gs5[1])
r_grid = np.linspace(0, 0.70, 200)
p_grid = np.linspace(0, 0.30, 200)
R, P = np.meshgrid(r_grid, p_grid)
lam_ell_bj = 55 * 0.300  # = 16.5 for p=1
# lambda_eff * ell = rho0 * p * ell * (1-r) = lam_ell_bj * P * (1-R)
LE = lam_ell_bj * P * (1 - R)
cmap_data = np.minimum(LE / LN3, 1.2)
cf = ax2.contourf(R * 100, P * 100, cmap_data,
                  levels=np.linspace(0, 1.2, 60), cmap='RdYlGn', alpha=0.85)
plt.colorbar(cf, ax=ax2, label='$(\\lambda\\ell)_{\\rm eff}$ / $\\ln(3)$', shrink=0.8)
cs = ax2.contour(R * 100, P * 100, cmap_data, levels=[1.0],
                 colors='darkgreen', linewidths=2.5)
ax2.clabel(cs, fmt='$p_c$ threshold', fontsize=8)
ax2.scatter([40], [2], color='black', s=100, zorder=6, label='Current Beijing')
pc_target = LN3 / (55 * 0.300 * (1 - 0.35)) * 100
ax2.scatter([35], [pc_target], color='red', s=120, marker='*', zorder=6, label='Phase II target')
ax2.annotate('', xy=(35, pc_target), xytext=(40, 2),
             arrowprops=dict(arrowstyle='->', color='navy', lw=1.5,
                            connectionstyle='arc3,rad=-0.2'))
ax2.text(36, 8, 'Signal\noptimisation\n+ CAV', fontsize=7.5, color='navy', ha='center')
ax2.set_xlabel('Signal red ratio $r$ (%)', fontsize=11)
ax2.set_ylabel('CAV penetration $p$ (%)', fontsize=11)
ax2.set_title('(b) Signal–CAV substitutability\n[Beijing arterial, $\\rho_0=55$ veh/km]', fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_xlim(0, 70); ax2.set_ylim(0, 30)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('Fig2_blindpred.pdf', bbox_inches='tight', dpi=150)
plt.savefig('Fig2_blindpred.png', bbox_inches='tight', dpi=150)
print("Fig5/Fig2 saved")

print("\n全部图片生成完毕")
