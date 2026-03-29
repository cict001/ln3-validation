import os, sys, glob
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else r"D:\BaiduNetdiskDownload\data\data"
OUT_DIR  = os.path.join(os.path.expanduser("~"), "Downloads", "chengdu_results")
os.makedirs(OUT_DIR, exist_ok=True)

LN3      = np.log(3)
ELL_KM   = 0.30
ELL_DEG  = ELL_KM / 111.0   # 300m in degrees latitude
LAT_STEP = 0.00135

print("="*65)
print("Coordination Cluster Survival Analysis")
print("For each 1-min snapshot: find clusters where N >= 3")
print("Track cluster lifetime across consecutive snapshots")
print("="*65)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_files:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR,"**","*.csv"), recursive=True))
print("Files: %d" % len(csv_files))

# Track active clusters: key=(lat_bin) -> {start_time, max_size, lam_ell}
active_clusters = {}   # lat_bin -> (t_start, n_veh, lam_ell)
cluster_lifetimes = [] # completed clusters: (lifetime_min, max_n, lam_ell, crossed_threshold)

# Store snapshots sorted by time across files
# Process file by file, maintain state
prev_t = None
prev_bins = set()

all_snapshots = []   # (t_bin, lat_bin, n_veh, lam_ell)

print("Collecting snapshots...")
for fi, fpath in enumerate(csv_files):
    try:
        df = pd.read_csv(fpath, low_memory=False,
                         usecols=['alarm_time','latitude','vehicle_id'],
                         dtype={'latitude':float,'vehicle_id':str})
    except: continue

    try:
        df['ts'] = pd.to_datetime(df['alarm_time'])
        df = df[df['ts'].dt.year == 2026]
    except: continue
    df = df[(df['latitude']>29.5)&(df['latitude']<31.5)]
    if df.empty: continue

    df['lat_bin'] = (df['latitude']/LAT_STEP).round()*LAT_STEP
    df['t_bin']   = df['ts'].dt.floor('1min')

    for (t_bin, lat_bin), grp in df.groupby(['t_bin','lat_bin']):
        n_veh   = grp['vehicle_id'].nunique()
        if n_veh < 2: continue
        lam_ell = n_veh * ELL_KM
        all_snapshots.append((t_bin, lat_bin, n_veh, lam_ell))

    if fi % 20 == 0:
        print("  [%d/%d] %d snapshots" % (fi+1, len(csv_files), len(all_snapshots)), end='\r')

print("\nTotal snapshots: %d" % len(all_snapshots))
df_snap = pd.DataFrame(all_snapshots, columns=['t_bin','lat_bin','n_veh','lam_ell'])
df_snap = df_snap.sort_values(['lat_bin','t_bin'])

# ── Cluster lifetime tracking per road segment ────────────────────────────────
print("Computing cluster lifetimes...")

cluster_events = []

for lat_bin, seg in df_snap.groupby('lat_bin'):
    seg = seg.sort_values('t_bin').reset_index(drop=True)
    times  = seg['t_bin'].values
    n_vehs = seg['n_veh'].values
    le_arr = seg['lam_ell'].values

    # Find runs of consecutive minutes with N >= 3
    in_cluster = n_vehs >= 3
    i = 0
    while i < len(in_cluster):
        if in_cluster[i]:
            j = i
            while j < len(in_cluster) and in_cluster[j]:
                # Check consecutive (within 2 minutes gap allowed)
                if j > i:
                    gap = (times[j]-times[j-1]) / np.timedelta64(1,'m')
                    if gap > 2:
                        break
                j += 1
            # Cluster from i to j-1
            duration = (times[j-1]-times[i]) / np.timedelta64(1,'m') + 1
            max_n    = int(n_vehs[i:j].max())
            mean_le  = float(le_arr[i:j].mean())
            above    = bool(mean_le >= LN3)
            cluster_events.append({
                'lat_bin':    lat_bin,
                't_start':    times[i],
                'duration_m': float(duration),
                'max_n':      max_n,
                'mean_lam_ell': mean_le,
                'above_ln3':  above,
            })
            i = j
        else:
            i += 1

print("Cluster events found: %d" % len(cluster_events))

# ── Analysis ──────────────────────────────────────────────────────────────────
df_cl = pd.DataFrame(cluster_events)

lines = []
lines.append("="*65)
lines.append("Coordination Cluster Survival Analysis")
lines.append("Definition: run of consecutive 1-min snapshots with N >= 3")
lines.append("="*65)
lines.append("Total cluster events: %d" % len(df_cl))
lines.append("")

if len(df_cl) > 10:
    below = df_cl[~df_cl['above_ln3']]['duration_m']
    above = df_cl[ df_cl['above_ln3']]['duration_m']

    lines.append("Cluster duration statistics:")
    lines.append("%-30s %8s %10s %10s %8s" %
                 ("Group","N","Mean(min)","Median(min)","Max(min)"))
    lines.append("-"*65)
    for label, arr in [("Sub-threshold (le<ln3)", below),
                        ("Super-threshold (le>=ln3)", above)]:
        if len(arr) > 0:
            lines.append("%-30s %8d %10.1f %10.1f %8.1f" %
                         (label, len(arr), arr.mean(), arr.median(), arr.max()))

    lines.append("")
    lines.append("Prediction: super-threshold clusters last LONGER")
    lines.append("(more vehicles = more coordination = sustained cluster)")
    if len(below) > 5 and len(above) > 5:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(above, below, alternative='greater')
        ratio = above.mean() / below.mean() if below.mean() > 0 else 0
        lines.append("Duration ratio (super/sub): %.2fx" % ratio)
        lines.append("Mann-Whitney p = %.4f  %s" %
                     (p, "CONFIRMED" if p < 0.05 else "not significant"))

    lines.append("")
    lines.append("Max cluster size distribution:")
    for n in [3, 4, 5, 6, 8, 10]:
        cnt = (df_cl['max_n'] >= n).sum()
        lines.append("  N >= %2d: %d clusters (%.1f%%)" %
                     (n, cnt, cnt/len(df_cl)*100))

    lines.append("")
    lines.append("Theoretical significance:")
    lines.append("  N >= 3 clusters exist: BIDIRECTIONAL coordination possible")
    lines.append("  Their persistence (duration) tests whether ln(3) threshold")
    lines.append("  sustains cooperation, not just enables it momentarily.")

    # Lifetime distribution for power law check
    all_dur = df_cl['duration_m'].values
    lines.append("")
    lines.append("Duration distribution (check for power law):")
    for threshold in [1, 2, 5, 10, 20, 60]:
        cnt = (all_dur >= threshold).sum()
        lines.append("  duration >= %3d min: %d  (%.1f%%)" %
                     (threshold, cnt, cnt/len(df_cl)*100))

txt = '\n'.join(lines)
print('\n' + txt)

out_txt = os.path.join(OUT_DIR, "cluster_survival.txt")
out_csv = os.path.join(OUT_DIR, "cluster_survival.csv")
with open(out_txt, 'w', encoding='utf-8') as f: f.write(txt)
df_cl.to_csv(out_csv, index=False)
print("\nSaved:")
print("  " + out_txt)
print("  " + out_csv)
