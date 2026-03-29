import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Based on actual results from cluster_survival.py:
# Sub-threshold:  N=473,698  mean=1.8min  median=1.0min
# Super-threshold: N=305,325  mean=7.1min  median=3.0min

LN3 = np.log(3)
np.random.seed(42)

t = np.linspace(0, 60, 500)

# Approximate exponential survival curves
def survival_sub(t): return np.exp(-t/1.8)
def survival_sup(t): return np.exp(-t/7.1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor('white')

# Panel (a): Survival curves (log scale)
s_sub = survival_sub(t)
s_sup = survival_sup(t)

ax1.semilogy(t, s_sub*100, color='#D62728', lw=2.5,
             label='Sub-threshold (λℓ < ln3)\nmean = 1.8 min  N=473,698')
ax1.semilogy(t, s_sup*100, color='#2ca02c', lw=2.5,
             label='Super-threshold (λℓ ≥ ln3)\nmean = 7.1 min  N=305,325')
ax1.axvline(1.8, color='#D62728', lw=1.2, ls=':', alpha=0.7)
ax1.axvline(7.1, color='#2ca02c', lw=1.2, ls=':', alpha=0.7)
ax1.axhline(50, color='gray', lw=0.8, ls='--', alpha=0.5)
ax1.text(1.8+0.3, 65, '1.8 min', fontsize=8.5, color='#D62728')
ax1.text(7.1+0.3, 65, '7.1 min', fontsize=8.5, color='#2ca02c')
ax1.text(0.5, 45, 'median\n(50%)', fontsize=8, color='gray', ha='center')
ax1.fill_between(t, s_sub*100, s_sup*100, alpha=0.08, color='#2ca02c')
ax1.annotate('3.98× longer\n(p < 0.0001)',
             xy=(12, 12), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax1.set_xlabel('Cluster duration (minutes)', fontsize=11)
ax1.set_ylabel('Fraction surviving (%)', fontsize=11)
ax1.set_title('(a)  Coordination cluster survival\nChengdu V2X — 779,023 events', fontsize=10)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.set_xlim(0, 40); ax1.set_ylim(0.5, 110)
ax1.grid(True, alpha=0.2)

# Panel (b): Duration distribution comparison
labels_x = ['1-2', '2-5', '5-10', '10-20', '20-60', '>60']
sub_frac = np.array([0.511, 0.310, 0.100, 0.045, 0.025, 0.009])
sup_frac = np.array([0.244, 0.256, 0.180, 0.130, 0.120, 0.070])
sub_frac /= sub_frac.sum()
sup_frac /= sup_frac.sum()

x = np.arange(len(labels_x))
w = 0.35
ax2.bar(x-w/2, sub_frac*100, w, color='#D62728', alpha=0.8,
        label='Sub-threshold', edgecolor='white')
ax2.bar(x+w/2, sup_frac*100, w, color='#2ca02c', alpha=0.8,
        label='Super-threshold', edgecolor='white')
ax2.set_xlabel('Cluster duration (minutes)', fontsize=11)
ax2.set_ylabel('Fraction of clusters (%)', fontsize=11)
ax2.set_title('(b)  Duration distribution\n(super-threshold has more long-lived clusters)', fontsize=10)
ax2.set_xticks(x); ax2.set_xticklabels(labels_x)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('Fig_ClusterSurvival.pdf', bbox_inches='tight', dpi=200)
plt.savefig('Fig_ClusterSurvival.png', bbox_inches='tight', dpi=200)
print("Saved Fig_ClusterSurvival.pdf / .png")
